//! stac collection source: pulls search the api lazily, in whole-degree
//! blocks cached for the source's lifetime, so coverage is not bound to
//! the bbox given at open. open searches that bbox once to anchor the
//! grid and crs on the most recent item. items are filtered to the
//! anchor crs, each item's cog asset opens lazily over http range
//! requests, and a pull mosaics its items most-recent-first

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use crate::caps::{
    CapsPattern, CapsSet, Constraint, Crs, Dtype, RasterPattern, ResRange, SetField,
};
use crate::chunk::{Chunk, RasterChunk};
use crate::element::Source;
use crate::elements::cog::{HttpRange, read_chunk};
use crate::error::{Error, Result};
use crate::window::{Bbox, GridSpec, WindowReq};
use futures::future::BoxFuture;
use terrano_core::CogReader;

/// lon/lat side of one search block. two degrees keeps a glo-30 block
/// (one item per degree square) far under the default page limit
const BLOCK_DEG: f64 = 2.0;

/// cold block searches one pull may trigger. a window past this cap is
/// asking for a mosaic of thousands of items anyway, fail loud instead
const MAX_BLOCK_SEARCHES: usize = 32;

#[derive(Clone)]
pub struct StacSearch {
    /// stac api root, e.g. `https://earth-search.aws.element84.com/v1`
    pub api: String,
    pub collection: String,
    /// asset key holding the cog, e.g. `data`
    pub asset: String,
    /// lon/lat anchor bbox: min lon, min lat, max lon, max lat. searched
    /// once at open to pick the grid and crs, pulls search past it lazily
    pub bbox: [f64; 4],
    /// rfc 3339 instant or interval, verbatim stac `datetime`
    pub datetime: Option<String>,
    /// items per search page. a block search filling a whole page fails
    /// rather than serving partial coverage
    pub limit: u32,
}

impl StacSearch {
    pub fn new(api: &str, collection: &str, asset: &str, bbox: [f64; 4]) -> Self {
        StacSearch {
            api: api.into(),
            collection: collection.into(),
            asset: asset.into(),
            bbox,
            datetime: None,
            limit: 100,
        }
    }
}

struct Found {
    href: String,
    datetime: String,
    /// lon/lat footprint straight from the stac feature
    bbox: [f64; 4],
    epsg: u32,
}

struct Item {
    href: String,
    datetime: String,
    /// footprint in the source crs, for skipping items a pull misses
    bbox: Bbox,
    reader: Mutex<Option<CogReader<HttpRange>>>,
}

struct Inner {
    search: StacSearch,
    client: reqwest::blocking::Client,
    crs: Crs,
    /// lon/lat to source crs for item footprints, `None` when it is 4326
    to_native: Option<projicio_core::Transform>,
    /// source crs to lon/lat for pull windows, `None` when it is 4326
    to_lonlat: Option<projicio_core::Transform>,
    searched: Mutex<HashSet<(i32, i32)>>,
    /// every item seen so far, keyed by href so blocks share one copy
    items: Mutex<HashMap<String, Arc<Item>>>,
}

pub struct StacSrc {
    inner: Arc<Inner>,
    origin_x: f64,
    origin_y: f64,
    base_resolution: f64,
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

/// one search request, first page only
fn search_page(
    client: &reqwest::blocking::Client,
    search: &StacSearch,
    bbox: [f64; 4],
) -> Result<Vec<Found>> {
    let fail = |detail: String| Error::Source(detail);
    let [minx, miny, maxx, maxy] = bbox;
    let mut url = format!(
        "{}/search?collections={}&bbox={minx},{miny},{maxx},{maxy}&limit={}",
        search.api, search.collection, search.limit
    );
    if let Some(dt) = &search.datetime {
        url.push_str(&format!("&datetime={dt}"));
    }
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
    let mut found = Vec::new();
    for f in features {
        let Some(href) = f["assets"][&search.asset]["href"].as_str() else {
            continue;
        };
        let datetime = f["properties"]["datetime"]
            .as_str()
            .unwrap_or("")
            .to_string();
        let bbox: Vec<f64> = f["bbox"]
            .as_array()
            .map(|a| a.iter().filter_map(|v| v.as_f64()).collect())
            .unwrap_or_default();
        if bbox.len() < 4 {
            continue;
        }
        let epsg = f["properties"]["proj:epsg"]
            .as_u64()
            .or_else(|| f["assets"][&search.asset]["proj:epsg"].as_u64())
            .unwrap_or(0) as u32;
        found.push(Found {
            href: s3_to_https(href),
            datetime,
            bbox: [bbox[0], bbox[1], bbox[2], bbox[3]],
            epsg,
        });
    }
    Ok(found)
}

impl StacSrc {
    /// searches the anchor bbox and opens the most recent item's cog to
    /// anchor the grid. blocking http, call it off the async runtime
    pub fn open(search: &StacSearch) -> Result<StacSrc> {
        let client = reqwest::blocking::Client::new();
        let mut found = search_page(&client, search, search.bbox)?;
        if found.is_empty() {
            return Err(Error::Source(format!(
                "stac search matched no items with asset {:?}",
                search.asset
            )));
        }
        // most recent first, so it anchors crs and grid
        found.sort_by(|a, b| b.datetime.cmp(&a.datetime));
        let epsg = found[0].epsg;
        let crs = Crs(epsg);
        let anchor_href = found[0].href.clone();

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

        let inner = Inner {
            search: search.clone(),
            client,
            crs,
            to_native,
            to_lonlat,
            searched: Mutex::new(HashSet::new()),
            items: Mutex::new(HashMap::new()),
        };
        inner.insert(found)?;
        let anchor = inner
            .items
            .lock()
            .unwrap()
            .get(&anchor_href)
            .cloned()
            .expect("anchor inserted above");
        let reader = CogReader::open(HttpRange::new(&anchor.href))?;
        let meta = reader.meta().clone();
        *anchor.reader.lock().unwrap() = Some(reader);
        Ok(StacSrc {
            inner: Arc::new(inner),
            origin_x: meta.origin_x,
            origin_y: meta.origin_y,
            base_resolution: meta.pixel_width,
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

    /// keep crs-matching items, one shared copy per href
    fn insert(&self, found: Vec<Found>) -> Result<()> {
        for f in found {
            if f.epsg != self.crs.0 {
                continue;
            }
            if self.items.lock().unwrap().contains_key(&f.href) {
                continue;
            }
            let item = Arc::new(Item {
                bbox: self.native_bbox(&f.bbox)?,
                href: f.href.clone(),
                datetime: f.datetime,
                reader: Mutex::new(None),
            });
            self.items.lock().unwrap().insert(f.href, item);
        }
        Ok(())
    }

    /// search every block the lon/lat window touches that has not been
    /// searched yet. concurrent pulls may race a block, the href dedup in
    /// `insert` makes that harmless
    fn ensure_coverage(&self, w: &[f64; 4]) -> Result<()> {
        let block = |v: f64, lo: f64, hi: f64| (v.clamp(lo, hi) / BLOCK_DEG).floor() as i32;
        let (ix0, ix1) = (block(w[0], -180.0, 180.0), block(w[2], -180.0, 180.0));
        let (iy0, iy1) = (block(w[1], -90.0, 90.0), block(w[3], -90.0, 90.0));
        let mut missing = Vec::new();
        {
            let searched = self.searched.lock().unwrap();
            for iy in iy0..=iy1 {
                for ix in ix0..=ix1 {
                    if !searched.contains(&(ix, iy)) {
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
            let found = search_page(&self.client, &self.search, bbox)?;
            if found.len() >= self.search.limit as usize {
                return Err(Error::Source(format!(
                    "stac block search filled a whole page of {} items, coverage would be partial",
                    found.len()
                )));
            }
            self.insert(found)?;
            self.searched.lock().unwrap().insert((ix, iy));
        }
        Ok(())
    }

    fn read_sync(&self, req: &WindowReq) -> Result<RasterChunk> {
        self.ensure_coverage(&self.lonlat_bbox(&req.bbox)?)?;
        let mut items: Vec<Arc<Item>> = self
            .items
            .lock()
            .unwrap()
            .values()
            .filter(|i| i.bbox.intersects(&req.bbox))
            .cloned()
            .collect();
        // most recent first, so it wins the mosaic
        items.sort_by(|a, b| b.datetime.cmp(&a.datetime));

        let mut merged: Option<RasterChunk> = None;
        for item in items {
            let mut slot = item.reader.lock().unwrap();
            if slot.is_none() {
                *slot = Some(CogReader::open(HttpRange::new(&item.href))?);
            }
            let reader = slot.as_mut().expect("opened above");
            let meta = reader.meta().clone();
            let chunk = read_chunk(reader, req, meta.origin_x, meta.origin_y, self.crs)?;
            drop(slot);
            match &mut merged {
                None => merged = Some(chunk),
                Some(out) => {
                    let add = chunk.bands.band(0).expect("single band").data().to_vec();
                    let band = out.bands.band_mut(0).expect("single band");
                    for (o, v) in band.data_mut().iter_mut().zip(add) {
                        if o.is_nan() {
                            *o = v;
                        }
                    }
                }
            }
            if let Some(out) = &merged {
                let band = out.bands.band(0).expect("single band");
                if !band.data().iter().any(|v| v.is_nan()) {
                    break;
                }
            }
        }
        match merged {
            Some(chunk) => Ok(chunk),
            // nothing intersects: an all-nodata window on the source grid
            None => {
                let cols = (req.bbox.width() / req.resolution).round() as usize;
                let rows = (req.bbox.height() / req.resolution).round() as usize;
                let band = terrano_core::Raster::from_vec(
                    cols,
                    rows,
                    vec![f64::NAN; cols * rows],
                    req.resolution,
                    f64::NAN,
                )
                .map_err(Error::Terrano)?;
                Ok(RasterChunk {
                    bands: terrano_core::BandedRaster::new(vec![band]).map_err(Error::Terrano)?,
                    bbox: req.bbox,
                    resolution: req.resolution,
                    crs: self.crs,
                })
            }
        }
    }
}

impl Source for StacSrc {
    fn constraint(&self) -> Constraint {
        Constraint::Produces(CapsSet::one(CapsPattern::Raster(RasterPattern {
            dtype: SetField::one(Dtype::F64),
            bands: SetField::one(1),
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

    fn read<'a>(&'a self, req: &'a WindowReq) -> BoxFuture<'a, Result<Chunk>> {
        let inner = self.inner.clone();
        let req = *req;
        Box::pin(async move {
            crate::engine::offload(move || inner.read_sync(&req).map(Chunk::Raster)).await
        })
    }
}
