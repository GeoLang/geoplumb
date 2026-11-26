use crate::caps::{
    BandKinds, Caps, CapsPattern, CapsSet, Constraint, Crs, RasterPattern, SetField, ValueKind,
};
use crate::chunk::{Chunk, RasterChunk};
use crate::element::Transform;
use crate::error::{Error, Result};
use crate::window::{Bbox, WindowReq};
use terrano_core::{
    BandedRaster, Raster, fill_sinks, flow_accumulation, flow_direction, stream_order, watershed,
};

fn projected_dem(crs: Crs, output: ValueKind) -> Constraint {
    Constraint::pinned_value_kind(
        CapsSet::one(CapsPattern::Raster(RasterPattern {
            bands: SetField::one(1),
            crs: SetField::one(crs),
            value_kinds: SetField::one(BandKinds::from(ValueKind::Continuous)),
            ..RasterPattern::default()
        })),
        output,
    )
}

fn check_crs_and_region(element: &str, crs: Crs, region: &Bbox) -> Result<()> {
    let definition = projicio_core::epsg::proj4_definition(crs.0).ok_or_else(|| {
        Error::InvalidGraph(format!("{element} crs {crs} has no projicio definition"))
    })?;
    // d8 distances in degrees differ between the two axes
    if definition.contains("+proj=longlat") {
        return Err(Error::InvalidGraph(format!(
            "{element} needs a projected crs, {crs} is longitude and latitude"
        )));
    }
    let edges = [region.min_x, region.min_y, region.max_x, region.max_y];
    let has_area = region.min_x < region.max_x && region.min_y < region.max_y;
    if !(edges.iter().all(|edge| edge.is_finite()) && has_area) {
        return Err(Error::InvalidGraph(format!(
            "{element} region {edges:?} is not min_x, min_y, max_x, max_y of a finite box with area"
        )));
    }
    Ok(())
}

fn filled_flow_directions(out: &WindowReq, input: &Chunk) -> Result<RasterChunk> {
    // fill_sinks seeds its flood at the raster edge
    let dem = input.raster()?.crop_to(&out.bbox);
    let band = dem
        .bands
        .band(0)
        .ok_or(Error::Source("empty input chunk".into()))?;
    let directions = flow_direction(&fill_sinks(band));
    Ok(RasterChunk {
        bands: BandedRaster::new(vec![directions]).expect("one band"),
        ..dem
    })
}

fn contributing_area(directions: &Raster, resolution: f64) -> Raster {
    let cell_area = resolution * resolution;
    // terrano's flow_accumulation counts upstream cells without the cell itself
    let upstream_cells = flow_accumulation(directions);
    let values = upstream_cells
        .data()
        .iter()
        .map(|&count| {
            if upstream_cells.is_nodata(count) {
                count
            } else {
                (count + 1.0) * cell_area
            }
        })
        .collect();
    Raster::from_vec(
        upstream_cells.width(),
        upstream_cells.height(),
        values,
        upstream_cells.cell_size,
        upstream_cells.nodata,
    )
    .expect("same dimensions as the accumulation")
}

fn with_band(grid: RasterChunk, band: Raster) -> Chunk {
    Chunk::Raster(RasterChunk {
        bands: BandedRaster::new(vec![band]).expect("one band"),
        ..grid
    })
}

pub struct FlowDirection {
    pub crs: Crs,
    pub region: Bbox,
}

impl Transform for FlowDirection {
    fn identity(&self) -> String {
        format!("FlowDirection crs={} region={:?}", self.crs, self.region)
    }

    fn constraint(&self) -> Constraint {
        projected_dem(self.crs, ValueKind::Categorical)
    }

    fn configure(&mut self, _input: &Caps, _output: &Caps) -> Result<()> {
        check_crs_and_region("flow direction", self.crs, &self.region)
    }

    fn region(&self) -> Option<Bbox> {
        Some(self.region)
    }

    fn plan(&self, out: &WindowReq) -> WindowReq {
        *out
    }

    fn compute(&self, out: &WindowReq, input: &Chunk) -> Result<Chunk> {
        Ok(Chunk::Raster(filled_flow_directions(out, input)?))
    }
}

pub struct FlowAccumulation {
    pub crs: Crs,
    pub region: Bbox,
}

impl Transform for FlowAccumulation {
    fn identity(&self) -> String {
        format!("FlowAccumulation crs={} region={:?}", self.crs, self.region)
    }

    fn constraint(&self) -> Constraint {
        projected_dem(self.crs, ValueKind::Continuous)
    }

    fn configure(&mut self, _input: &Caps, _output: &Caps) -> Result<()> {
        check_crs_and_region("flow accumulation", self.crs, &self.region)
    }

    fn region(&self) -> Option<Bbox> {
        Some(self.region)
    }

    fn plan(&self, out: &WindowReq) -> WindowReq {
        *out
    }

    fn compute(&self, out: &WindowReq, input: &Chunk) -> Result<Chunk> {
        let directions = filled_flow_directions(out, input)?;
        let codes = directions.bands.band(0).expect("one band");
        let area = contributing_area(codes, directions.resolution);
        Ok(with_band(directions, area))
    }
}

pub struct Watershed {
    pub crs: Crs,
    pub region: Bbox,
}

impl Transform for Watershed {
    fn identity(&self) -> String {
        format!("Watershed crs={} region={:?}", self.crs, self.region)
    }

    fn constraint(&self) -> Constraint {
        projected_dem(self.crs, ValueKind::Categorical)
    }

    fn configure(&mut self, _input: &Caps, _output: &Caps) -> Result<()> {
        check_crs_and_region("watershed", self.crs, &self.region)
    }

    fn region(&self) -> Option<Bbox> {
        Some(self.region)
    }

    fn plan(&self, out: &WindowReq) -> WindowReq {
        *out
    }

    fn compute(&self, out: &WindowReq, input: &Chunk) -> Result<Chunk> {
        let directions = filled_flow_directions(out, input)?;
        let basins = watershed(directions.bands.band(0).expect("one band"));
        Ok(with_band(directions, basins))
    }
}

pub struct StreamOrder {
    pub crs: Crs,
    pub region: Bbox,
    pub threshold: f64,
}

impl Transform for StreamOrder {
    fn identity(&self) -> String {
        format!(
            "StreamOrder crs={} region={:?} threshold={:?}",
            self.crs, self.region, self.threshold
        )
    }

    fn constraint(&self) -> Constraint {
        projected_dem(self.crs, ValueKind::Categorical)
    }

    fn configure(&mut self, _input: &Caps, _output: &Caps) -> Result<()> {
        if !(self.threshold.is_finite() && self.threshold > 0.0) {
            return Err(Error::InvalidGraph(format!(
                "stream order needs a positive threshold area, got {}",
                self.threshold
            )));
        }
        check_crs_and_region("stream order", self.crs, &self.region)
    }

    fn region(&self) -> Option<Bbox> {
        Some(self.region)
    }

    fn plan(&self, out: &WindowReq) -> WindowReq {
        *out
    }

    fn compute(&self, out: &WindowReq, input: &Chunk) -> Result<Chunk> {
        let directions = filled_flow_directions(out, input)?;
        let codes = directions.bands.band(0).expect("one band");
        let area = contributing_area(codes, directions.resolution);
        let orders = stream_order(&area, codes, self.threshold);
        Ok(with_band(directions, orders))
    }
}
