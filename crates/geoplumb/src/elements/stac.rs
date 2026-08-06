//! stac collection source: pulls search the api lazily, in whole-degree
//! blocks cached for the source's lifetime, so coverage is not bound to
//! the bbox given at open. open searches that bbox once to anchor the
//! grid, crs and band count on the most recent item. items are filtered
//! to the anchor crs and band counts, searches follow `next` links to the
//! end, each item's cog assets open lazily over http range requests, and
//! a pull mosaics its items most-recent-first, band by band. a search may
//! name several assets (collections like sentinel-2 spread bands across
//! per-band cogs): an item is kept only when every asset is present, and
//! a pull reads them all, bands concatenated in asset order. item reads
//! run in parallel, capped across every pull in the process.
//!
//! a pull may carry its own interval, which overrides the search's
//! `datetime` for that pull: blocks are searched, and their item sets
//! kept, per interval, so one source serves any number of instants. the
//! rfc 3339 side of `TimeInterval` lives here, the only place that speaks
//! it

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, LazyLock, Mutex};

use crate::caps::{
    CapsPattern, CapsSet, Constraint, Crs, Dtype, RasterPattern, ResRange, SetField,
};
use crate::chunk::{Chunk, RasterChunk};
use crate::element::Source;
use crate::elements::cog::{HttpRange, band_count, read_chunk};
use crate::error::{Error, Result};
use crate::window::{Bbox, GridSpec, TimeInterval, WindowReq};
use futures::future::BoxFuture;
use terrano_core::CogReader;

/// lon/lat side of one search block. two degrees keeps a glo-30 block
/// (one item per degree square) far under the default page limit
const BLOCK_DEG: f64 = 2.0;

/// cold block searches one pull may trigger. a window past this cap is
/// asking for a mosaic of thousands of items anyway, fail loud instead
const MAX_BLOCK_SEARCHES: usize = 32;

/// features one search may accumulate across its pages, for the same
/// reason: past this the mosaic is not something a pull should be doing
const MAX_SEARCH_ITEMS: usize = 1000;

/// item reads in flight across every pull in the process. each read is
/// blocking http offloaded to the runtime's blocking pool, so without a
/// shared cap n concurrent pulls would stack n windows' worth of threads
const MAX_PARALLEL_ITEM_READS: usize = 64;

static ITEM_READ_PERMITS: LazyLock<tokio::sync::Semaphore> =
    LazyLock::new(|| tokio::sync::Semaphore::new(MAX_PARALLEL_ITEM_READS));

/// readers an item keeps per asset once a read returns them. each pooled
/// reader holds its own http client and client thread, so an uncapped
/// pool retains one thread per concurrent read it ever saw
const POOLED_READERS_PER_ASSET: usize = 4;

/// epoch milliseconds from an rfc 3339 timestamp, the form stac item and
/// search datetimes take: a date, `T`, a time, optional fractional
/// seconds past the second, and `Z` or a `+hh:mm` utc offset
pub fn parse_rfc3339(s: &str) -> Result<i64> {
    parse_ms(s).ok_or_else(|| Error::Source(format!("not an rfc 3339 timestamp: {s}")))
}

/// the inverse, always utc, fractional seconds only when there are any
pub fn format_rfc3339(ms: i64) -> String {
    let (y, m, d) = civil_from_days(ms.div_euclid(86_400_000));
    let ms_of_day = ms.rem_euclid(86_400_000);
    let (s, milli) = (ms_of_day / 1000, ms_of_day % 1000);
    let (h, mi, sec) = (s / 3600, s / 60 % 60, s % 60);
    let date = format!("{y:04}-{m:02}-{d:02}T{h:02}:{mi:02}:{sec:02}");
    match milli {
        0 => format!("{date}Z"),
        _ => format!("{date}.{milli:03}Z"),
    }
}

/// the rfc 3339 face of the engine's time interval. it sits here because
/// the search string is the only place geoplumb speaks rfc 3339
impl TimeInterval {
    /// an rfc 3339 `start/end` interval, the form stac's `datetime` takes
    pub fn parse(s: &str) -> Result<TimeInterval> {
        let (start, end) = s
            .split_once('/')
            .ok_or_else(|| Error::Source(format!("not an rfc 3339 interval: {s}")))?;
        let interval = TimeInterval::new(parse_rfc3339(start)?, parse_rfc3339(end)?);
        if interval.end_ms < interval.start_ms {
            return Err(Error::Source(format!(
                "time interval {s} ends before it starts"
            )));
        }
        Ok(interval)
    }

    /// the `datetime` a search carrying this interval sends. stac compares
    /// it closed at both ends, so an item stamped exactly at `end_ms` can
    /// come back even though the interval excludes that instant
    pub fn to_stac_datetime(self) -> String {
        format!(
            "{}/{}",
            format_rfc3339(self.start_ms),
            format_rfc3339(self.end_ms)
        )
    }
}

fn parse_ms(s: &str) -> Option<i64> {
    let num = |v: &str| match !v.is_empty() && v.bytes().all(|b| b.is_ascii_digit()) {
        true => v.parse::<i64>().ok(),
        false => None,
    };
    let (date, rest) = s.split_at_checked(10)?;
    let mut parts = date.split('-');
    let (y, mo, d) = (
        num(parts.next()?)?,
        num(parts.next()?)?,
        num(parts.next()?)?,
    );
    if parts.next().is_some() {
        return None;
    }
    let rest = rest.strip_prefix(['T', 't'])?;
    let (time, offset_ms) = match rest.strip_suffix(['Z', 'z']) {
        Some(time) => (time, 0),
        None => {
            let (time, offset) = rest.split_at_checked(rest.len().checked_sub(6)?)?;
            let sign = match offset.as_bytes()[0] {
                b'+' => 1,
                b'-' => -1,
                _ => return None,
            };
            let (oh, om) = offset[1..].split_once(':')?;
            (time, sign * (num(oh)? * 3600 + num(om)? * 60) * 1000)
        }
    };
    let mut parts = time.split(':');
    let (h, mi) = (num(parts.next()?)?, num(parts.next()?)?);
    let seconds = parts.next()?;
    if parts.next().is_some() {
        return None;
    }
    let (sec, frac) = match seconds.split_once('.') {
        None => (num(seconds)?, 0),
        Some((sec, frac)) => {
            if frac.is_empty() || !frac.bytes().all(|b| b.is_ascii_digit()) {
                return None;
            }
            // truncate or pad to milliseconds, the engine's resolution
            let milli: String = frac.chars().chain("000".chars()).take(3).collect();
            (num(sec)?, num(&milli)?)
        }
    };
    if !(1..=12).contains(&mo)
        || !(1..=days_in_month(y, mo)).contains(&d)
        || h > 23
        || mi > 59
        || sec > 60
    {
        return None;
    }
    Some((days_from_civil(y, mo, d) * 86_400 + h * 3600 + mi * 60 + sec) * 1000 + frac - offset_ms)
}

fn days_in_month(y: i64, m: i64) -> i64 {
    let leap = y % 4 == 0 && (y % 100 != 0 || y % 400 == 0);
    match m {
        2 if leap => 29,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    }
}

/// days since 1970-01-01 in the proleptic gregorian calendar, and its
/// inverse. howard hinnant's `days_from_civil` / `civil_from_days`
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// how a pull combines the items covering one window. `datetime`, or the
/// pull's own interval where it has one, picks the interval and this
/// picks what the interval collapses to. the composite is fixed at the
/// source either way, only the interval moves per pull
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Composite {
    /// most recent item wins each pixel, older ones fill its nodata
    #[default]
    Latest,
    Mean,
    Median,
    Min,
    Max,
}

#[derive(Clone)]
pub struct StacSearch {
    /// stac api root, e.g. `https://earth-search.aws.element84.com/v1`
    pub api: String,
    pub collection: String,
    /// asset keys holding the cogs, e.g. `["data"]`, or `["red", "nir"]`
    /// for a collection with one cog per band. items missing any listed
    /// asset are skipped, pulls concatenate bands in this order
    pub assets: Vec<String>,
    /// lon/lat anchor bbox: min lon, min lat, max lon, max lat. searched
    /// once at open to pick the grid and crs, pulls search past it lazily
    pub bbox: [f64; 4],
    /// rfc 3339 instant or interval, verbatim stac `datetime`
    pub datetime: Option<String>,
    /// items per search page. a search follows `next` links until the api
    /// runs out, so this is page size only, not a coverage limit
    pub limit: u32,
    /// what a pull does with the items covering one window
    pub composite: Composite,
}

impl StacSearch {
    pub fn new(api: &str, collection: &str, asset: &str, bbox: [f64; 4]) -> Self {
        StacSearch {
            api: api.into(),
            collection: collection.into(),
            assets: vec![asset.into()],
            bbox,
            datetime: None,
            limit: 100,
            composite: Composite::default(),
        }
    }
}

struct Found {
    /// one href per search asset, in asset order
    hrefs: Vec<String>,
    datetime: String,
    /// lon/lat footprint straight from the stac feature
    bbox: [f64; 4],
    epsg: u32,
    /// band count each asset declares, `None` when it declares none
    bands: Vec<Option<u16>>,
}

struct Item {
    /// one href per search asset, in asset order
    hrefs: Vec<String>,
    datetime: String,
    /// footprint in the source crs, for skipping items a pull misses
    bbox: Bbox,
    /// per-asset pools of opened readers: a read takes one out, opens
    /// fresh when the pool is empty, and puts it back after, so
    /// concurrent chunks of one pull do not serialize on an item
    readers: Vec<Mutex<Vec<CogReader<HttpRange>>>>,
}

struct Inner {
    search: StacSearch,
    client: reqwest::blocking::Client,
    crs: Crs,
    /// per-asset band counts of the anchor item's cogs, which every kept
    /// item shares
    asset_bands: Vec<u16>,
    /// their sum, the band count a pull produces
    bands: u16,
    /// lon/lat to source crs for item footprints, `None` when it is 4326
    to_native: Option<projicio_core::Transform>,
    /// source crs to lon/lat for pull windows, `None` when it is 4326
    to_lonlat: Option<projicio_core::Transform>,
    /// blocks searched, per pull time. `None` is the search's own
    /// `datetime`, the only time there was before pulls carried one
    searched: Mutex<HashSet<(i32, i32, Option<TimeInterval>)>>,
    /// every item seen so far, keyed by href so blocks and pull times
    /// share one copy, its pooled readers included
    items: Mutex<HashMap<String, Arc<Item>>>,
    /// the hrefs each pull time's searches matched, so two times over one
    /// block do not pool their items into one mosaic
    matched: Mutex<HashMap<Option<TimeInterval>, HashSet<String>>>,
}

pub struct StacSrc {
    inner: Arc<Inner>,
    origin_x: f64,
    origin_y: f64,
    base_resolution: f64,
    bands: u16,
    crs: Crs,
}

/// stac asset hrefs on aws are often `s3://bucket/key`, publicly readable
/// at the corresponding https endpoint
pub fn s3_to_https(href: &str) -> String {
    match href.strip_prefix("s3://").and_then(|r| r.split_once('/')) {
        Some((bucket, key)) => format!("https://{bucket}.s3.amazonaws.com/{key}"),
        None => href.to_string(),
    }
}

/// one search, following the response's `next` links until the api runs
/// out of pages, so coverage never depends on `limit`
fn search_page(
    client: &reqwest::blocking::Client,
    search: &StacSearch,
    bbox: [f64; 4],
    time: Option<TimeInterval>,
) -> Result<Vec<Found>> {
    let fail = |detail: String| Error::Source(detail);
    let [minx, miny, maxx, maxy] = bbox;
    let mut url = format!(
        "{}/search?collections={}&bbox={minx},{miny},{maxx},{maxy}&limit={}",
        search.api, search.collection, search.limit
    );
    // a pull's interval replaces the search's own for this search
    let datetime = time
        .map(TimeInterval::to_stac_datetime)
        .or_else(|| search.datetime.clone());
    if let Some(dt) = datetime {
        url.push_str(&format!("&datetime={dt}"));
    }

    let mut found = Vec::new();
    let mut seen = 0usize;
    loop {
        let text = client
            .get(&url)
            .header(reqwest::header::ACCEPT, "application/geo+json")
            .send()
            .and_then(|r| r.error_for_status())
            .and_then(|r| r.text())
            .map_err(|e| fail(format!("stac search failed: {e}")))?;
        let body: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| fail(format!("stac search returned invalid json: {e}")))?;

        let features = body["features"]
            .as_array()
            .ok_or_else(|| fail("stac search response has no features".into()))?;
        seen += features.len();
        if seen > MAX_SEARCH_ITEMS {
            return Err(fail(format!(
                "stac search over bbox {minx},{miny},{maxx},{maxy} passed {MAX_SEARCH_ITEMS} items across its pages, narrow the search"
            )));
        }
        for f in features {
            if let Some(item) = parse_found(f, &search.assets) {
                found.push(item);
            }
        }

        let next = body["links"].as_array().and_then(|links| {
            links
                .iter()
                .find(|l| l["rel"].as_str() == Some("next"))
                .and_then(|l| l["href"].as_str())
        });
        match next {
            Some(href) => url = href.to_string(),
            None => break,
        }
    }
    Ok(found)
}

fn parse_found(f: &serde_json::Value, assets: &[String]) -> Option<Found> {
    // every asset or none: a partial item cannot serve a full band stack
    let hrefs: Vec<String> = assets
        .iter()
        .map(|a| f["assets"][a]["href"].as_str().map(s3_to_https))
        .collect::<Option<_>>()?;
    let datetime = f["properties"]["datetime"]
        .as_str()
        .unwrap_or("")
        .to_string();
    let bbox: Vec<f64> = f["bbox"]
        .as_array()
        .map(|a| a.iter().filter_map(|v| v.as_f64()).collect())
        .unwrap_or_default();
    if bbox.len() < 4 {
        return None;
    }
    let epsg = f["properties"]["proj:epsg"]
        .as_u64()
        .or_else(|| {
            assets
                .iter()
                .find_map(|a| f["assets"][a]["proj:epsg"].as_u64())
        })
        .unwrap_or(0) as u32;
    // raster:bands is the per-asset one, eo:bands the spectral list, both
    // one entry per band. an asset declaring neither is not filtered
    let bands = assets
        .iter()
        .map(|a| {
            ["raster:bands", "eo:bands"]
                .iter()
                .find_map(|k| f["assets"][a][k].as_array())
                .and_then(|arr| u16::try_from(arr.len()).ok())
        })
        .collect();
    Some(Found {
        hrefs,
        datetime,
        bbox: [bbox[0], bbox[1], bbox[2], bbox[3]],
        epsg,
        bands,
    })
}

impl StacSrc {
    /// searches the anchor bbox and opens the most recent item's cogs to
    /// anchor the grid. blocking http, call it off the async runtime
    pub fn open(search: &StacSearch) -> Result<StacSrc> {
        if search.assets.is_empty() {
            return Err(Error::Source("stac search names no assets".into()));
        }
        let client = reqwest::blocking::Client::new();
        let mut found = search_page(&client, search, search.bbox, None)?;
        if found.is_empty() {
            return Err(Error::Source(format!(
                "stac search matched no items with assets {:?}",
                search.assets
            )));
        }
        // most recent first, so it anchors crs and grid
        found.sort_by(|a, b| b.datetime.cmp(&a.datetime));
        let epsg = found[0].epsg;
        let crs = Crs(epsg);
        let anchor_key = found[0].hrefs[0].clone();

        let (to_native, to_lonlat) = if epsg == 4326 {
            (None, None)
        } else {
            let auth = crs.authority();
            let proj = |from: &str, to: &str| {
                projicio_core::Transform::new(from, to)
                    .map_err(|e| Error::Projection(e.to_string()))
            };
            (
                Some(proj("EPSG:4326", &auth)?),
                Some(proj(&auth, "EPSG:4326")?),
            )
        };

        // the anchor cogs open before the filter runs, their band counts
        // are what the other items are held to
        let readers = found[0]
            .hrefs
            .iter()
            .map(|href| CogReader::open(HttpRange::new(href)))
            .collect::<core::result::Result<Vec<_>, _>>()?;
        let meta = readers[0].meta().clone();
        for (reader, asset) in readers.iter().zip(&search.assets) {
            let res = reader.meta().pixel_width;
            if res != meta.pixel_width {
                return Err(Error::Source(format!(
                    "asset {asset} has pixel size {res}, asset {} has {}: \
                     assets on different resolutions cannot stack into one raster",
                    search.assets[0], meta.pixel_width
                )));
            }
        }
        let asset_bands = readers
            .iter()
            .map(band_count)
            .collect::<Result<Vec<u16>>>()?;
        let bands = asset_bands.iter().sum();

        let inner = Inner {
            search: search.clone(),
            client,
            crs,
            asset_bands,
            bands,
            to_native,
            to_lonlat,
            searched: Mutex::new(HashSet::new()),
            items: Mutex::new(HashMap::new()),
            matched: Mutex::new(HashMap::new()),
        };
        // the anchor search ran on the search's own datetime, so its
        // items belong to the `None` time
        inner.insert(found, None)?;
        let anchor = inner
            .items
            .lock()
            .unwrap()
            .get(&anchor_key)
            .cloned()
            .ok_or_else(|| {
                Error::Source(format!(
                    "anchor item {anchor_key} declares a band count its cogs do not have"
                ))
            })?;
        for (pool, reader) in anchor.readers.iter().zip(readers) {
            pool.lock().unwrap().push(reader);
        }
        Ok(StacSrc {
            inner: Arc::new(inner),
            origin_x: meta.origin_x,
            origin_y: meta.origin_y,
            base_resolution: meta.pixel_width,
            bands,
            crs,
        })
    }

    /// items discovered so far, across the open search and every block
    pub fn item_count(&self) -> usize {
        self.inner.items.lock().unwrap().len()
    }

    /// the crs all kept items share, from the most recent anchor item
    pub fn crs(&self) -> Crs {
        self.crs
    }

    /// the band count all kept items share: the anchor item's cog bands,
    /// summed across the search's assets
    pub fn bands(&self) -> u16 {
        self.bands
    }
}

impl Inner {
    fn native_bbox(&self, b: &[f64; 4]) -> Result<Bbox> {
        match &self.to_native {
            None => Ok(Bbox::new(b[0], b[1], b[2], b[3])),
            Some(t) => {
                let fail = |detail: String| Error::Source(detail);
                let (x0, y0) = t.convert(b[0], b[1]).map_err(|e| fail(e.to_string()))?;
                let (x1, y1) = t.convert(b[2], b[3]).map_err(|e| fail(e.to_string()))?;
                Ok(Bbox::new(x0.min(x1), y0.min(y1), x0.max(x1), y0.max(y1)))
            }
        }
    }

    fn lonlat_bbox(&self, b: &Bbox) -> Result<[f64; 4]> {
        match &self.to_lonlat {
            None => Ok([b.min_x, b.min_y, b.max_x, b.max_y]),
            Some(t) => {
                let fail = |detail: String| Error::Source(detail);
                let (x0, y0) = t
                    .convert(b.min_x, b.min_y)
                    .map_err(|e| fail(e.to_string()))?;
                let (x1, y1) = t
                    .convert(b.max_x, b.max_y)
                    .map_err(|e| fail(e.to_string()))?;
                Ok([x0.min(x1), y0.min(y1), x0.max(x1), y0.max(y1)])
            }
        }
    }

    /// keep items matching the anchor's crs and per-asset band counts,
    /// one shared copy per item, keyed by its first asset's href, and
    /// record which of them `time`'s searches matched
    fn insert(&self, found: Vec<Found>, time: Option<TimeInterval>) -> Result<()> {
        for f in found {
            if f.epsg != self.crs.0 {
                continue;
            }
            if f.bands
                .iter()
                .zip(&self.asset_bands)
                .any(|(declared, anchor)| declared.is_some_and(|b| b != *anchor))
            {
                continue;
            }
            let key = f.hrefs[0].clone();
            if !self.items.lock().unwrap().contains_key(&key) {
                let item = Arc::new(Item {
                    bbox: self.native_bbox(&f.bbox)?,
                    readers: f.hrefs.iter().map(|_| Mutex::new(Vec::new())).collect(),
                    hrefs: f.hrefs,
                    datetime: f.datetime,
                });
                self.items.lock().unwrap().insert(key.clone(), item);
            }
            self.matched
                .lock()
                .unwrap()
                .entry(time)
                .or_default()
                .insert(key);
        }
        Ok(())
    }

    /// search every block the lon/lat window touches that has not been
    /// searched at this time yet. concurrent pulls may race a block, the
    /// href dedup in `insert` makes that harmless. the cold-block cap
    /// applies per time, the way it applies per source
    fn ensure_coverage(&self, w: &[f64; 4], time: Option<TimeInterval>) -> Result<()> {
        let block = |v: f64, lo: f64, hi: f64| (v.clamp(lo, hi) / BLOCK_DEG).floor() as i32;
        let (ix0, ix1) = (block(w[0], -180.0, 180.0), block(w[2], -180.0, 180.0));
        let (iy0, iy1) = (block(w[1], -90.0, 90.0), block(w[3], -90.0, 90.0));
        let mut missing = Vec::new();
        {
            let searched = self.searched.lock().unwrap();
            for iy in iy0..=iy1 {
                for ix in ix0..=ix1 {
                    if !searched.contains(&(ix, iy, time)) {
                        missing.push((ix, iy));
                    }
                }
            }
        }
        if missing.len() > MAX_BLOCK_SEARCHES {
            return Err(Error::Source(format!(
                "window needs {} cold stac block searches, the cap is {MAX_BLOCK_SEARCHES}",
                missing.len()
            )));
        }
        for (ix, iy) in missing {
            let bbox = [
                (f64::from(ix) * BLOCK_DEG).max(-180.0),
                (f64::from(iy) * BLOCK_DEG).max(-90.0),
                (f64::from(ix + 1) * BLOCK_DEG).min(180.0),
                (f64::from(iy + 1) * BLOCK_DEG).min(90.0),
            ];
            let found = search_page(&self.client, &self.search, bbox, time)?;
            self.insert(found, time)?;
            self.searched.lock().unwrap().insert((ix, iy, time));
        }
        Ok(())
    }

    /// one item's window across every asset, bands concatenated in asset
    /// order, each cog opening on first use
    fn read_item(&self, item: &Item, req: &WindowReq) -> Result<RasterChunk> {
        let mut bands = Vec::with_capacity(usize::from(self.bands));
        for (k, pool) in item.readers.iter().enumerate() {
            let taken = pool.lock().unwrap().pop();
            let mut reader = match taken {
                Some(r) => r,
                None => CogReader::open(HttpRange::new(&item.hrefs[k]))?,
            };
            let meta = reader.meta().clone();
            let asset_bands = band_count(&reader)?;
            if asset_bands != self.asset_bands[k] {
                return Err(Error::Source(format!(
                    "stac item asset {} has {asset_bands} bands, the anchor item has {}",
                    item.hrefs[k], self.asset_bands[k]
                )));
            }
            let chunk = read_chunk(&mut reader, req, meta.origin_x, meta.origin_y, self.crs)?;
            {
                let mut pool = pool.lock().unwrap();
                if pool.len() < POOLED_READERS_PER_ASSET {
                    pool.push(reader);
                }
            }
            bands.extend(chunk.bands.into_bands());
        }
        Ok(RasterChunk {
            bands: terrano_core::BandedRaster::new(bands).map_err(Error::Terrano)?,
            bbox: req.bbox,
            resolution: req.resolution,
            crs: self.crs,
        })
    }

    /// every chunk reduced per pixel per band, one chunk per item. a
    /// reducer needs the whole stack, the caller reads it first
    fn reduce_chunks(
        &self,
        chunks: &[RasterChunk],
        req: &WindowReq,
        op: Composite,
    ) -> Result<RasterChunk> {
        let (cols, rows) = (chunks[0].width(), chunks[0].height());
        let mut bands = Vec::with_capacity(usize::from(self.bands));
        for bi in 0..usize::from(self.bands) {
            let planes: Vec<&[f64]> = chunks
                .iter()
                .map(|c| c.bands.band(bi).expect("equal band counts").data())
                .collect();
            let mut values = Vec::with_capacity(planes.len());
            let mut data = Vec::with_capacity(cols * rows);
            for p in 0..cols * rows {
                values.clear();
                values.extend(planes.iter().map(|pl| pl[p]).filter(|v| v.is_finite()));
                data.push(reduce_values(&mut values, op));
            }
            bands.push(
                terrano_core::Raster::from_vec(cols, rows, data, req.resolution, f64::NAN)
                    .map_err(Error::Terrano)?,
            );
        }
        Ok(RasterChunk {
            bands: terrano_core::BandedRaster::new(bands).map_err(Error::Terrano)?,
            bbox: req.bbox,
            resolution: req.resolution,
            crs: self.crs,
        })
    }

    /// nothing intersects: an all-nodata window on the source grid
    fn empty(&self, req: &WindowReq) -> Result<RasterChunk> {
        let cols = (req.bbox.width() / req.resolution).round() as usize;
        let rows = (req.bbox.height() / req.resolution).round() as usize;
        let bands = (0..self.bands)
            .map(|_| {
                terrano_core::Raster::from_vec(
                    cols,
                    rows,
                    vec![f64::NAN; cols * rows],
                    req.resolution,
                    f64::NAN,
                )
            })
            .collect::<core::result::Result<Vec<_>, _>>()
            .map_err(Error::Terrano)?;
        Ok(RasterChunk {
            bands: terrano_core::BandedRaster::new(bands).map_err(Error::Terrano)?,
            bbox: req.bbox,
            resolution: req.resolution,
            crs: self.crs,
        })
    }

    /// search coverage for the window at the pull's time, then every item
    /// that time matched which touches the window. most recent first, so
    /// Latest fills in the right order. the reducers do not care, they
    /// share the sort anyway
    fn intersecting_items(&self, req: &WindowReq) -> Result<Vec<Arc<Item>>> {
        self.ensure_coverage(&self.lonlat_bbox(&req.bbox)?, req.time)?;
        let hrefs = self
            .matched
            .lock()
            .unwrap()
            .get(&req.time)
            .cloned()
            .unwrap_or_default();
        let all = self.items.lock().unwrap();
        let mut items: Vec<Arc<Item>> = hrefs
            .iter()
            .filter_map(|href| all.get(href))
            .filter(|i| i.bbox.intersects(&req.bbox))
            .cloned()
            .collect();
        drop(all);
        items.sort_by(|a, b| b.datetime.cmp(&a.datetime));
        Ok(items)
    }
}

/// one window read per item, `MAX_PARALLEL_ITEM_READS` in flight across
/// the whole process, results in item order
async fn read_items(
    inner: &Arc<Inner>,
    items: &[Arc<Item>],
    req: &WindowReq,
) -> Result<Vec<RasterChunk>> {
    let reads = items.iter().map(|item| {
        let inner = inner.clone();
        let item = item.clone();
        let req = *req;
        async move {
            let _permit = ITEM_READ_PERMITS
                .acquire()
                .await
                .expect("semaphore is never closed");
            crate::engine::offload(move || inner.read_item(&item, &req)).await
        }
    });
    futures::future::try_join_all(reads).await
}

/// most-recent-first fill: each chunk patches the nodata the newer ones
/// left
fn fill_latest(mut merged: Option<RasterChunk>, chunks: Vec<RasterChunk>) -> Option<RasterChunk> {
    for chunk in chunks {
        match &mut merged {
            None => merged = Some(chunk),
            Some(out) => {
                for (bi, src) in chunk.bands.bands().iter().enumerate() {
                    let dst = out.bands.band_mut(bi).expect("equal band counts");
                    for (o, v) in dst.data_mut().iter_mut().zip(src.data()) {
                        if o.is_nan() {
                            *o = *v;
                        }
                    }
                }
            }
        }
    }
    merged
}

/// a pixel is complete only when every band has a value there
fn complete(chunk: &RasterChunk) -> bool {
    !chunk
        .bands
        .bands()
        .iter()
        .any(|b| b.data().iter().any(|v| v.is_nan()))
}

/// the fill walks in waves so the early exit survives parallel reads: it
/// stops after the first complete wave, wasting at most one wave of reads
async fn latest_window(
    inner: &Arc<Inner>,
    items: &[Arc<Item>],
    req: &WindowReq,
) -> Result<Option<RasterChunk>> {
    let mut merged: Option<RasterChunk> = None;
    for wave in items.chunks(MAX_PARALLEL_ITEM_READS) {
        let chunks = read_items(inner, wave, req).await?;
        merged = crate::engine::offload(move || fill_latest(merged, chunks)).await;
        if merged.as_ref().is_some_and(complete) {
            break;
        }
    }
    Ok(merged)
}

async fn reduce_window(
    inner: &Arc<Inner>,
    items: &[Arc<Item>],
    req: &WindowReq,
    op: Composite,
) -> Result<Option<RasterChunk>> {
    let chunks = read_items(inner, items, req).await?;
    if chunks.is_empty() {
        return Ok(None);
    }
    let inner = inner.clone();
    let req = *req;
    crate::engine::offload(move || inner.reduce_chunks(&chunks, &req, op).map(Some)).await
}

/// reduce one pixel's finite values across items. `vals` holds only finite
/// values, so min and max cannot fold a NaN into the answer, and a pixel
/// no item had a value at stays nodata
fn reduce_values(vals: &mut [f64], op: Composite) -> f64 {
    if vals.is_empty() {
        return f64::NAN;
    }
    match op {
        // items arrive most-recent-first, so the first value is the latest
        Composite::Latest => vals[0],
        Composite::Mean => vals.iter().sum::<f64>() / vals.len() as f64,
        Composite::Min => vals.iter().copied().fold(f64::INFINITY, f64::min),
        Composite::Max => vals.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        Composite::Median => {
            vals.sort_by(f64::total_cmp);
            let mid = vals.len() / 2;
            if vals.len() % 2 == 1 {
                vals[mid]
            } else {
                (vals[mid - 1] + vals[mid]) / 2.0
            }
        }
    }
}

impl Source for StacSrc {
    fn constraint(&self) -> Constraint {
        Constraint::Produces(CapsSet::one(CapsPattern::Raster(RasterPattern {
            dtype: SetField::one(Dtype::F64),
            bands: SetField::one(self.bands),
            crs: SetField::one(self.crs),
            resolution: ResRange::at_least(self.base_resolution),
            chunk_px: SetField::Any,
        })))
    }

    fn grid(&self) -> GridSpec {
        GridSpec {
            origin_x: self.origin_x,
            origin_y: self.origin_y,
            base_resolution: self.base_resolution,
            chunk_px: 256,
        }
    }

    fn time_varying(&self) -> bool {
        true
    }

    fn read<'a>(&'a self, req: &'a WindowReq) -> BoxFuture<'a, Result<Chunk>> {
        let inner = self.inner.clone();
        let req = *req;
        Box::pin(async move {
            let items = {
                let inner = inner.clone();
                crate::engine::offload(move || inner.intersecting_items(&req)).await?
            };
            let merged = match inner.search.composite {
                Composite::Latest => latest_window(&inner, &items, &req).await?,
                op => reduce_window(&inner, &items, &req, op).await?,
            };
            match merged {
                Some(chunk) => Ok(Chunk::Raster(chunk)),
                None => crate::engine::offload(move || inner.empty(&req).map(Chunk::Raster)).await,
            }
        })
    }
}
