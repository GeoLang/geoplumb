use terrano_core::{Raster, SampleFormat, SampleGrid, TypedBand};

use crate::caps::Dtype;
use crate::error::{Error, Result};

#[derive(Debug, Clone)]
pub struct RasterBands {
    bands: Vec<TypedBand>,
}

impl RasterBands {
    pub fn new(bands: Vec<TypedBand>) -> Result<RasterBands> {
        let first = bands.first().ok_or_else(|| {
            terrano_core::Error::InvalidInput("a raster chunk needs at least one band".into())
        })?;
        let (width, height) = (first.width(), first.height());
        for (index, band) in bands.iter().enumerate().skip(1) {
            if (band.width(), band.height()) != (width, height) {
                return Err(terrano_core::Error::InvalidInput(format!(
                    "band {index} is {}x{}, first band is {width}x{height}",
                    band.width(),
                    band.height()
                ))
                .into());
            }
        }
        Ok(RasterBands { bands })
    }

    pub fn from_rasters(rasters: Vec<Raster>) -> Result<RasterBands> {
        RasterBands::new(rasters.into_iter().map(TypedBand::F64).collect())
    }

    pub fn band_count(&self) -> usize {
        self.bands.len()
    }

    pub fn width(&self) -> usize {
        self.bands[0].width()
    }

    pub fn height(&self) -> usize {
        self.bands[0].height()
    }

    pub fn band(&self, index: usize) -> Option<&TypedBand> {
        self.bands.get(index)
    }

    pub fn bands(&self) -> &[TypedBand] {
        &self.bands
    }

    pub fn into_bands(self) -> Vec<TypedBand> {
        self.bands
    }

    pub fn into_rasters(self) -> Vec<Raster> {
        self.bands
            .into_iter()
            .map(|band| match band {
                TypedBand::F64(raster) => raster,
                narrow => narrow.to_raster().into_owned(),
            })
            .collect()
    }

    pub fn byte_size(&self) -> usize {
        self.bands
            .iter()
            .map(|band| band.width() * band.height() * band.format().bytes() as usize)
            .sum()
    }
}

pub(crate) trait Sample: Copy + PartialEq + Send + Sync + 'static {
    const FLOAT: bool;

    fn view(band: &TypedBand) -> Option<(&[Self], Option<Self>)>;

    fn build(
        width: usize,
        height: usize,
        cell_size: f64,
        values: Vec<Self>,
        nodata: Option<Self>,
    ) -> TypedBand;

    fn is_absent(self, nodata: Option<Self>) -> bool;

    fn fill(nodata: Option<Self>) -> Self;

    fn widen(self) -> f64;

    fn narrow(value: f64, nodata: Option<Self>) -> Self;

    fn push_le(self, out: &mut Vec<u8>);

    fn from_le(bytes: &[u8]) -> Self;
}

macro_rules! integer_sample {
    ($($sample:ty => $variant:ident),*) => {$(
        impl Sample for $sample {
            const FLOAT: bool = false;

            fn view(band: &TypedBand) -> Option<(&[Self], Option<Self>)> {
                match band {
                    TypedBand::$variant { samples, nodata } => Some((samples.values(), *nodata)),
                    _ => None,
                }
            }

            fn build(
                width: usize,
                height: usize,
                cell_size: f64,
                values: Vec<Self>,
                nodata: Option<Self>,
            ) -> TypedBand {
                let samples = SampleGrid::new(width, height, cell_size, values)
                    .expect("callers size the values to width times height");
                TypedBand::$variant { samples, nodata }
            }

            fn is_absent(self, nodata: Option<Self>) -> bool {
                Some(self) == nodata
            }

            fn fill(nodata: Option<Self>) -> Self {
                // gdal fills integer cells outside the image with nodata, or zero without one
                nodata.unwrap_or(0)
            }

            fn widen(self) -> f64 {
                f64::from(self)
            }

            fn narrow(value: f64, nodata: Option<Self>) -> Self {
                if value.is_nan() {
                    return Self::fill(nodata);
                }
                // rust float to integer casts saturate at the type's range
                let narrowed = value.round() as Self;
                // gdal's warp kernel moves a valid value off the nodata one step down, or up at the minimum
                match nodata {
                    Some(sentinel) if narrowed == sentinel && sentinel == Self::MIN => narrowed + 1,
                    Some(sentinel) if narrowed == sentinel => narrowed - 1,
                    _ => narrowed,
                }
            }

            fn push_le(self, out: &mut Vec<u8>) {
                out.extend_from_slice(&self.to_le_bytes());
            }

            fn from_le(bytes: &[u8]) -> Self {
                Self::from_le_bytes(bytes.try_into().expect("callers pass one sample's bytes"))
            }
        }
    )*};
}

integer_sample!(u8 => U8, i8 => I8, u16 => U16, i16 => I16, u32 => U32, i32 => I32);

impl Sample for f32 {
    const FLOAT: bool = true;

    fn view(band: &TypedBand) -> Option<(&[Self], Option<Self>)> {
        match band {
            TypedBand::F32(samples) => Some((samples.values(), None)),
            _ => None,
        }
    }

    fn build(
        width: usize,
        height: usize,
        cell_size: f64,
        values: Vec<Self>,
        _nodata: Option<Self>,
    ) -> TypedBand {
        TypedBand::F32(
            SampleGrid::new(width, height, cell_size, values)
                .expect("callers size the values to width times height"),
        )
    }

    fn is_absent(self, _nodata: Option<Self>) -> bool {
        self.is_nan()
    }

    fn fill(_nodata: Option<Self>) -> Self {
        f32::NAN
    }

    fn widen(self) -> f64 {
        f64::from(self)
    }

    fn narrow(value: f64, _nodata: Option<Self>) -> Self {
        value as f32
    }

    fn push_le(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_le_bytes());
    }

    fn from_le(bytes: &[u8]) -> Self {
        Self::from_le_bytes(bytes.try_into().expect("callers pass one sample's bytes"))
    }
}

impl Sample for f64 {
    const FLOAT: bool = true;

    fn view(band: &TypedBand) -> Option<(&[Self], Option<Self>)> {
        match band {
            TypedBand::F64(raster) => Some((raster.data(), Some(raster.nodata))),
            _ => None,
        }
    }

    fn build(
        width: usize,
        height: usize,
        cell_size: f64,
        values: Vec<Self>,
        nodata: Option<Self>,
    ) -> TypedBand {
        TypedBand::F64(
            Raster::from_vec(width, height, values, cell_size, nodata.unwrap_or(f64::NAN))
                .expect("callers size the values to width times height"),
        )
    }

    fn is_absent(self, nodata: Option<Self>) -> bool {
        self.is_nan() || Some(self) == nodata
    }

    fn fill(nodata: Option<Self>) -> Self {
        nodata.unwrap_or(f64::NAN)
    }

    fn widen(self) -> f64 {
        self
    }

    fn narrow(value: f64, _nodata: Option<Self>) -> Self {
        value
    }

    fn push_le(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_le_bytes());
    }

    fn from_le(bytes: &[u8]) -> Self {
        Self::from_le_bytes(bytes.try_into().expect("callers pass one sample's bytes"))
    }
}

macro_rules! by_format {
    ($format:expr, $function:ident($($argument:expr),* $(,)?)) => {
        match $format {
            SampleFormat::U8 => $function::<u8>($($argument),*),
            SampleFormat::I8 => $function::<i8>($($argument),*),
            SampleFormat::U16 => $function::<u16>($($argument),*),
            SampleFormat::I16 => $function::<i16>($($argument),*),
            SampleFormat::U32 => $function::<u32>($($argument),*),
            SampleFormat::I32 => $function::<i32>($($argument),*),
            SampleFormat::F32 => $function::<f32>($($argument),*),
            SampleFormat::F64 => $function::<f64>($($argument),*),
        }
    };
}

fn gathered_samples<T: Sample>(
    sources: &[&TypedBand],
    width: usize,
    height: usize,
    cell_size: f64,
    pick: &impl Fn(usize) -> Option<(usize, usize)>,
) -> TypedBand {
    let views: Vec<(&[T], Option<T>)> = sources
        .iter()
        .map(|band| T::view(band).expect("callers pass bands of one format"))
        .collect();
    let nodata = views[0].1;
    let fill = T::fill(nodata);
    let values = (0..width * height)
        .map(|index| match pick(index) {
            Some((source, at)) => views[source].0[at],
            None => fill,
        })
        .collect();
    T::build(width, height, cell_size, values, nodata)
}

pub(crate) fn picked(
    band: &TypedBand,
    width: usize,
    height: usize,
    cell_size: f64,
    pick: impl Fn(usize) -> Option<usize>,
) -> TypedBand {
    let sources = [band];
    let pick = |index| pick(index).map(|at| (0, at));
    by_format!(
        band.format(),
        gathered_samples(&sources, width, height, cell_size, &pick)
    )
}

pub(crate) fn gathered(
    sources: &[&TypedBand],
    width: usize,
    height: usize,
    cell_size: f64,
    pick: impl Fn(usize) -> Option<(usize, usize)>,
) -> Result<TypedBand> {
    let dtype = Dtype::of(sources[0]);
    if let Some(other) = sources.iter().find(|band| Dtype::of(band) != dtype) {
        return Err(Error::InvalidGraph(format!(
            "bands of {dtype:?} and {:?} cannot fill one band",
            Dtype::of(other)
        )));
    }
    Ok(by_format!(
        dtype.format(),
        gathered_samples(sources, width, height, cell_size, &pick)
    ))
}

fn narrowed_to<T: Sample>(
    nodata: Option<T>,
    width: usize,
    height: usize,
    cell_size: f64,
    values: Vec<f64>,
) -> TypedBand {
    let samples = values
        .into_iter()
        .map(|value| T::narrow(value, nodata))
        .collect();
    T::build(width, height, cell_size, samples, nodata)
}

pub(crate) fn narrowed(
    dtype: Dtype,
    width: usize,
    height: usize,
    cell_size: f64,
    values: Vec<f64>,
) -> TypedBand {
    match dtype {
        Dtype::U8 { nodata } => narrowed_to(nodata, width, height, cell_size, values),
        Dtype::I8 { nodata } => narrowed_to(nodata, width, height, cell_size, values),
        Dtype::U16 { nodata } => narrowed_to(nodata, width, height, cell_size, values),
        Dtype::I16 { nodata } => narrowed_to(nodata, width, height, cell_size, values),
        Dtype::U32 { nodata } => narrowed_to(nodata, width, height, cell_size, values),
        Dtype::I32 { nodata } => narrowed_to(nodata, width, height, cell_size, values),
        Dtype::F32 => narrowed_to::<f32>(None, width, height, cell_size, values),
        Dtype::F64 => narrowed_to::<f64>(None, width, height, cell_size, values),
    }
}

pub(crate) fn narrowed_raster(dtype: Dtype, raster: &Raster) -> TypedBand {
    let values = raster
        .data()
        .iter()
        .map(|&value| {
            if raster.is_nodata(value) {
                f64::NAN
            } else {
                value
            }
        })
        .collect();
    narrowed(
        dtype,
        raster.width(),
        raster.height(),
        raster.cell_size,
        values,
    )
}

fn narrowed_like_samples<T: Sample>(
    like: &TypedBand,
    width: usize,
    height: usize,
    cell_size: f64,
    values: Vec<f64>,
) -> TypedBand {
    let (_, nodata) = T::view(like).expect("dispatched on the band's own format");
    narrowed_to(nodata, width, height, cell_size, values)
}

pub(crate) fn narrowed_like(
    like: &TypedBand,
    width: usize,
    height: usize,
    cell_size: f64,
    values: Vec<f64>,
) -> TypedBand {
    by_format!(
        like.format(),
        narrowed_like_samples(like, width, height, cell_size, values)
    )
}

fn masked_samples<T: Sample>(band: &TypedBand, keep: &[bool], cell_size: f64) -> Result<TypedBand> {
    let (values, nodata) = T::view(band).expect("dispatched on the band's own format");
    let masked_nodata = if T::FLOAT { None } else { nodata };
    if !T::FLOAT && masked_nodata.is_none() && keep.contains(&false) {
        return Err(Error::InvalidGraph(format!(
            "a {} band with no nodata value has no way to mark a masked cell",
            band.format().name()
        )));
    }
    let absent = T::fill(masked_nodata);
    let masked = values
        .iter()
        .zip(keep)
        .map(|(&value, &keep)| {
            if keep && !value.is_absent(nodata) {
                value
            } else {
                absent
            }
        })
        .collect();
    Ok(T::build(
        band.width(),
        band.height(),
        cell_size,
        masked,
        masked_nodata,
    ))
}

pub(crate) fn masked(band: &TypedBand, keep: &[bool], cell_size: f64) -> Result<TypedBand> {
    by_format!(band.format(), masked_samples(band, keep, cell_size))
}

fn finite_sample<T: Sample>(band: &TypedBand, index: usize) -> Option<f64> {
    let (values, nodata) = T::view(band).expect("dispatched on the band's own format");
    let value = values[index];
    if value.is_absent(nodata) {
        return None;
    }
    Some(value.widen()).filter(|value| value.is_finite())
}

pub(crate) fn finite_value(band: &TypedBand, index: usize) -> Option<f64> {
    by_format!(band.format(), finite_sample(band, index))
}

fn encoded_samples<T: Sample>(band: &TypedBand) -> Vec<u8> {
    let (values, nodata) = T::view(band).expect("dispatched on the band's own format");
    let mut out = Vec::with_capacity(1 + size_of::<T>() * (values.len() + 1));
    out.push(u8::from(nodata.is_some()));
    nodata.unwrap_or(T::fill(None)).push_le(&mut out);
    for value in values {
        value.push_le(&mut out);
    }
    out
}

pub(crate) fn encoded_band(band: &TypedBand) -> Vec<u8> {
    by_format!(band.format(), encoded_samples(band))
}

fn decoded_samples<T: Sample>(
    nodata: Option<&[u8]>,
    samples: &[u8],
    width: usize,
    height: usize,
    cell_size: f64,
) -> TypedBand {
    let values = samples
        .chunks_exact(size_of::<T>())
        .map(T::from_le)
        .collect();
    T::build(width, height, cell_size, values, nodata.map(T::from_le))
}

pub(crate) fn decoded_band(
    format: SampleFormat,
    nodata: Option<&[u8]>,
    samples: &[u8],
    width: usize,
    height: usize,
    cell_size: f64,
) -> TypedBand {
    by_format!(
        format,
        decoded_samples(nodata, samples, width, height, cell_size)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(dtype: Dtype, values: &[f64]) -> TypedBand {
        narrowed(dtype, values.len(), 1, 1.0, values.to_vec())
    }

    fn u8_values(band: &TypedBand) -> &[u8] {
        u8::view(band).expect("a u8 band").0
    }

    #[test]
    fn narrowing_rounds_saturates_and_steps_off_the_sentinel() {
        let heights = row(
            Dtype::I16 {
                nodata: Some(i16::MIN),
            },
            &[-40_000.0, 0.5, -0.5, 2.5, 40_000.0, f64::NAN],
        );
        let (values, nodata) = i16::view(&heights).expect("an i16 band");
        assert_eq!(values, &[i16::MIN + 1, 1, -1, 3, i16::MAX, i16::MIN]);
        assert_eq!(nodata, Some(i16::MIN));

        let zero_sentinel = row(Dtype::U8 { nodata: Some(0) }, &[0.2, 300.0, f64::NAN]);
        assert_eq!(u8_values(&zero_sentinel), &[1, 255, 0]);
        let top_sentinel = row(Dtype::U8 { nodata: Some(255) }, &[300.0, 254.6]);
        assert_eq!(u8_values(&top_sentinel), &[254, 254]);
        let inner_sentinel = row(Dtype::U8 { nodata: Some(7) }, &[7.4, 6.6, 8.0]);
        assert_eq!(u8_values(&inner_sentinel), &[6, 6, 8]);
        let unmarked = row(Dtype::U8 { nodata: None }, &[f64::NAN, 3.0]);
        assert_eq!(u8_values(&unmarked), &[0, 3]);

        let floats = row(Dtype::F32, &[1.5, f64::NAN]);
        let values = f32::view(&floats).expect("an f32 band").0;
        assert_eq!(values[0], 1.5);
        assert!(values[1].is_nan());
    }

    #[test]
    fn gathering_refuses_bands_of_two_dtypes() {
        let codes = row(Dtype::U8 { nodata: Some(0) }, &[1.0, 2.0]);
        let other_sentinel = row(Dtype::U8 { nodata: Some(9) }, &[1.0, 2.0]);
        let picked = gathered(&[&codes, &codes], 2, 1, 1.0, |index| Some((index, index)));
        assert_eq!(u8_values(&picked.unwrap()), &[1, 2]);
        assert!(gathered(&[&codes, &other_sentinel], 2, 1, 1.0, |_| None).is_err());
    }

    #[test]
    fn masking_writes_the_sentinel_and_refuses_a_band_without_one() {
        let codes = row(Dtype::U8 { nodata: Some(0) }, &[3.0, 4.0, 5.0]);
        let masked_codes = masked(&codes, &[true, false, true], 1.0).unwrap();
        assert_eq!(u8_values(&masked_codes), &[3, 0, 5]);

        let unmarked = row(Dtype::U8 { nodata: None }, &[3.0, 4.0, 5.0]);
        assert!(masked(&unmarked, &[true, true, true], 1.0).is_ok());
        assert!(masked(&unmarked, &[true, false, true], 1.0).is_err());
    }
}
