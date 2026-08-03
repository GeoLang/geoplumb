//! stac collection source: one search at open resolves the matching
//! items, each item's cog asset opens lazily over http range requests,
//! and a pull mosaics the items most-recent-first onto the source grid.
//! items are filtered to the first (most recent) item's crs, and only the
//! first search page is used

use std::sync::{Arc, Mutex};

use crate::caps::{
    CapsPattern, CapsSet, Constraint, Crs, Dtype, RasterPattern, ResRange, SetField,
};
use crate::chunk::RasterChunk;
use crate::element::Source;
use crate::elements::cog::{HttpRange, read_chunk};
use crate::error::{Error, Result};
use crate::window::{Bbox, GridSpec, WindowReq};
use futures::future::BoxFuture;
use terrano_core::CogReader;

pub struct StacSearch {
    /// stac api root, e.g. `https://earth-search.aws.element84.com/v1`
    pub api: String,
    pub collection: String,
    /// asset key holding the cog, e.g. `data`
    pub asset: String,
    /// lon/lat search bbox: min lon, min lat, max lon, max lat
    pub bbox: [f64; 4],
    /// rfc 3339 instant or interval, verbatim stac `datetime`
    pub datetime: Option<String>,
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

struct Item {
    href: String,
    /// footprint in the source crs, for skipping items a pull misses
    bbox: Bbox,
    reader: Mutex<Option<CogReader<HttpRange>>>,
}

struct Inner {
    items: Vec<Item>,
    crs: Crs,
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

impl StacSrc {
    /// searches the api and opens the most recent item's cog to anchor the
    /// grid. blocking http, call it off the async runtime
    pub fn open(search: &StacSearch) -> Result<StacSrc> {
        let fail = |detail: String| Error::Source(detail);
        let [minx, miny, maxx, maxy] = search.bbox;
        let mut url = format!(
            "{}/search?collections={}&bbox={minx},{miny},{maxx},{maxy}&limit={}",
            search.api, search.collection, search.limit
        );
        if let Some(dt) = &search.datetime {
            url.push_str(&format!("&datetime={dt}"));
        }
        let text = reqwest::blocking::Client::new()
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
        let mut found: Vec<(String, String, [f64; 4], u32)> = Vec::new();
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
            found.push((
                s3_to_https(href),
                datetime,
                [bbox[0], bbox[1], bbox[2], bbox[3]],
                epsg,
            ));
        }
        if found.is_empty() {
            return Err(fail(format!(
                "stac search matched no items with asset {:?}",
                search.asset
            )));
        }
        // most recent first, so it wins the mosaic and anchors crs and grid
        found.sort_by(|a, b| b.1.cmp(&a.1));
        let epsg = found[0].3;
        found.retain(|f| f.3 == epsg);
        let crs = Crs(epsg);

        let to_native = if epsg == 4326 {
            None
        } else {
            Some(
                projicio_core::Transform::new("EPSG:4326", &crs.authority())
                    .map_err(|e| Error::Projection(e.to_string()))?,
            )
        };
        let native_bbox = |b: &[f64; 4]| -> Result<Bbox> {
            match &to_native {
                None => Ok(Bbox::new(b[0], b[1], b[2], b[3])),
                Some(t) => {
                    let (x0, y0) = t.convert(b[0], b[1]).map_err(|e| fail(e.to_string()))?;
                    let (x1, y1) = t.convert(b[2], b[3]).map_err(|e| fail(e.to_string()))?;
                    Ok(Bbox::new(x0.min(x1), y0.min(y1), x0.max(x1), y0.max(y1)))
                }
            }
        };

        let items: Vec<Item> = found
            .iter()
            .map(|f| {
                Ok(Item {
                    href: f.0.clone(),
                    bbox: native_bbox(&f.2)?,
                    reader: Mutex::new(None),
                })
            })
            .collect::<Result<_>>()?;

        let first = CogReader::open(HttpRange::new(&items[0].href))?;
        let meta = first.meta().clone();
        *items[0].reader.lock().unwrap() = Some(first);
        Ok(StacSrc {
            inner: Arc::new(Inner { items, crs }),
            origin_x: meta.origin_x,
            origin_y: meta.origin_y,
            base_resolution: meta.pixel_width,
            crs,
        })
    }

    /// items kept after the crs filter
    pub fn item_count(&self) -> usize {
        self.inner.items.len()
    }

    /// the crs all kept items share, from the most recent item
    pub fn crs(&self) -> Crs {
        self.crs
    }
}

impl Inner {
    fn read_sync(&self, req: &WindowReq) -> Result<RasterChunk> {
        let mut merged: Option<RasterChunk> = None;
        for item in &self.items {
            if !item.bbox.intersects(&req.bbox) {
                continue;
            }
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

    fn read<'a>(&'a self, req: &'a WindowReq) -> BoxFuture<'a, Result<RasterChunk>> {
        let inner = self.inner.clone();
        let req = *req;
        Box::pin(async move { crate::engine::offload(move || inner.read_sync(&req)).await })
    }
}
