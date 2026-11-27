use std::path::Path;

use serde_json::Value;
use topoi_core::Point;
use topoi_core::geojson::{Feature, FeatureCollection, FeatureGeometry};

use super::vector::VecSrc;
use crate::caps::Crs;
use crate::error::{Error, Result};

impl VecSrc {
    pub fn from_csv(path: &Path, x_column: &str, y_column: &str, crs: Crs) -> Result<Self> {
        VecSrc::new(read_csv(path, x_column, y_column)?, crs)
    }
}

pub fn read_csv(path: &Path, x_column: &str, y_column: &str) -> Result<FeatureCollection> {
    let named = |message: String| Error::Source(format!("{}: {message}", path.display()));
    let mut reader = csv::Reader::from_path(path).map_err(|e| named(e.to_string()))?;
    let headers = reader.headers().map_err(|e| named(e.to_string()))?.clone();
    let column = |name: &str| {
        headers.iter().position(|h| h == name).ok_or_else(|| {
            named(format!(
                "no column is named {name}, the header names {}",
                headers.iter().collect::<Vec<_>>().join(", ")
            ))
        })
    };
    let (x, y) = (column(x_column)?, column(y_column)?);

    let mut features = Vec::new();
    for record in reader.records() {
        let record = record.map_err(|e| named(e.to_string()))?;
        let line = record
            .position()
            .expect("csv records read from a file carry their position")
            .line();
        let coordinate = |index: usize, name: &str| {
            let cell = &record[index];
            cell.trim()
                .parse::<f64>()
                .ok()
                .filter(|value| value.is_finite())
                .ok_or_else(|| named(format!("line {line}: {name} {cell:?} is not a number")))
        };
        let point = Point::new(coordinate(x, x_column)?, coordinate(y, y_column)?);
        let properties = headers
            .iter()
            .zip(record.iter())
            .enumerate()
            .filter(|(index, _)| *index != x && *index != y)
            .map(|(_, (name, cell))| (name.to_string(), cell_value(cell)))
            .collect();
        features.push(Feature {
            geometry: Some(FeatureGeometry::Point(point)),
            properties,
        });
    }
    Ok(FeatureCollection { features })
}

fn cell_value(cell: &str) -> Value {
    if let Ok(integer) = cell.parse::<i64>() {
        return Value::from(integer);
    }
    match cell.parse::<f64>() {
        Ok(number) if number.is_finite() => Value::from(number),
        _ => Value::from(cell),
    }
}
