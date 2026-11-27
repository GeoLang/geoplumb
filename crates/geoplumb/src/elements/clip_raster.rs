use crate::bands::{RasterBands, masked, narrowed};
use crate::caps::{
    BandDtypes, Caps, CapsPattern, CapsSet, Constraint, Dtype, FieldMask, RasterPattern, SetField,
};
use crate::chunk::{Chunk, RasterChunk};
use crate::element::Fanin;
use crate::error::{Error, Result};
use crate::window::{GridSpec, WindowReq};
use terrano_core::SampleFormat;
use topoi_core::geojson::FeatureGeometry;
use topoi_core::{GridWindow, rasterize};

pub struct ClipRaster {
    nodata: Option<f64>,
    output_dtypes: Option<BandDtypes>,
}

impl ClipRaster {
    pub fn new(nodata: Option<f64>) -> ClipRaster {
        ClipRaster {
            nodata,
            output_dtypes: None,
        }
    }
}

fn dtype_pattern(dtype: Dtype) -> CapsPattern {
    CapsPattern::Raster(RasterPattern {
        dtypes: SetField::one(BandDtypes::from(dtype)),
        ..RasterPattern::default()
    })
}

fn sentinel_members(nodata: f64) -> impl Iterator<Item = Constraint> {
    SampleFormat::ALL
        .into_iter()
        .filter(|format| format.integer_range().is_some())
        .filter_map(move |format| {
            let with_sentinel = Dtype::from_format(format, Some(nodata)).ok()?;
            let without = Dtype::from_format(format, None).expect("a format without nodata builds");
            Some(Constraint::Derived {
                input: CapsSet::one(dtype_pattern(without)),
                passthrough: FieldMask {
                    dtypes: false,
                    ..FieldMask::ALL
                },
                output: dtype_pattern(with_sentinel),
            })
        })
}

impl Fanin for ClipRaster {
    fn identity(&self) -> String {
        format!("ClipRaster nodata={:?}", self.nodata)
    }

    fn constraint(&self) -> Constraint {
        // the solver fixates a link's first alternative, so a sentinel member goes before the pass-through one
        let mut members: Vec<Constraint> =
            self.nodata.into_iter().flat_map(sentinel_members).collect();
        members.push(Constraint::Derived {
            input: CapsSet::any_raster(),
            passthrough: FieldMask::ALL,
            output: CapsPattern::Raster(RasterPattern::default()),
        });
        // fan-in constraints apply to every input link, the polygon link included
        members.push(Constraint::Derived {
            input: CapsSet::any_vector(),
            passthrough: FieldMask::ALL,
            output: CapsPattern::Raster(RasterPattern::default()),
        });
        Constraint::AnyOf(members)
    }

    fn configure(&mut self, inputs: &[Caps], output: &Caps) -> Result<()> {
        let [Caps::Raster(raster), Caps::Vector(_)] = inputs else {
            return Err(Error::InvalidGraph(
                "a raster clip takes a raster and then a polygon layer as its two inputs".into(),
            ));
        };
        let mut dtypes = Vec::with_capacity(usize::from(raster.bands));
        for band in 0..usize::from(raster.bands) {
            let from = raster.dtypes.band(band);
            let to = match self.nodata {
                None if from.marks_absent_cells() => from,
                None => {
                    return Err(Error::InvalidGraph(format!(
                        "band {band} is {from:?}, which has no nodata for the pixels outside \
                         the cutline, so the clip needs a nodata value"
                    )));
                }
                Some(value) if from.marks_absent_cells() => {
                    return Err(Error::InvalidGraph(format!(
                        "band {band} is {from:?}, which marks absent cells already, so the \
                         clip takes no nodata {value}"
                    )));
                }
                Some(value) => Dtype::from_format(from.format(), Some(value))?,
            };
            dtypes.push(to);
        }
        let dtypes = BandDtypes::from_bands(dtypes);
        let solved = &output.raster().dtypes;
        if *solved != dtypes {
            return Err(Error::InvalidGraph(format!(
                "the clip writes {dtypes:?} and the solve gave its output {solved:?}"
            )));
        }
        self.output_dtypes = Some(dtypes);
        Ok(())
    }

    fn output_grid(&self, inputs: &[GridSpec]) -> GridSpec {
        inputs[0]
    }

    fn compute(&self, out: &WindowReq, inputs: &[Chunk]) -> Result<Chunk> {
        let raster = inputs[0].raster()?.crop_to(&out.bbox);
        let cutline = inputs[1].vector()?;
        let mut polygons = Vec::with_capacity(cutline.features.len());
        for feature in &cutline.features {
            match &feature.geometry {
                FeatureGeometry::Polygon(_) | FeatureGeometry::MultiPolygon(_) => {
                    polygons.push((feature.geometry.clone(), 1.0))
                }
                _ => {
                    return Err(Error::Source(format!(
                        "cutline feature {} is not a polygon or multipolygon",
                        feature.id
                    )));
                }
            }
        }
        let res = raster.resolution;
        let (cols, rows) = (raster.width(), raster.height());
        let window = GridWindow {
            origin_x: raster.bbox.min_x,
            origin_y: raster.bbox.min_y,
            width: cols,
            height: rows,
            cell_size: res,
        };
        let burned = rasterize(&polygons, &window);
        // topoi burns rows bottom-up, raster rows run top-down
        let keep: Vec<bool> = (0..cols * rows)
            .map(|index| !burned[(rows - 1 - index / cols) * cols + index % cols].is_nan())
            .collect();
        let output_dtypes = self.output_dtypes.as_ref().expect("configured");
        let bands = raster
            .bands
            .bands()
            .iter()
            .enumerate()
            .map(|(bi, band)| {
                let to = output_dtypes.band(bi);
                if Dtype::of(band) == to {
                    return masked(band, &keep, res);
                }
                let values = band
                    .to_raster()
                    .data()
                    .iter()
                    .zip(&keep)
                    .map(|(&value, &inside)| if inside { value } else { f64::NAN })
                    .collect();
                Ok(narrowed(to, cols, rows, res, values))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Chunk::Raster(RasterChunk {
            bands: RasterBands::new(bands)?,
            bbox: raster.bbox,
            resolution: res,
            crs: raster.crs,
        }))
    }
}
