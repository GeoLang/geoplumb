use std::sync::Arc;

use crate::caps::{CapsPattern, CapsSet, Constraint, Domain, FieldMask, SetField, TensorPattern};
use crate::chunk::{Chunk, TensorChunk, TensorData};
use crate::element::{Adapter, Transform};
use crate::elements::tensor::ConvolutionWindow;
use crate::error::Result;
use crate::gpu::{Convolution, GpuContext};
use crate::window::{Bbox, WindowReq};

fn tensor_in_domain(domain: Domain) -> CapsPattern {
    CapsPattern::Tensor(TensorPattern {
        domain: SetField::one(domain),
        ..TensorPattern::default()
    })
}

fn domain_transfer(from: Domain, to: Domain) -> Constraint {
    Constraint::Derived {
        input: CapsSet::one(tensor_in_domain(from)),
        passthrough: FieldMask::ALL,
        output: tensor_in_domain(to),
    }
}

fn demands_domain(target: &CapsPattern, domain: Domain) -> bool {
    matches!(target, CapsPattern::Tensor(t) if matches!(&t.domain, SetField::OneOf(d) if d.contains(&domain)))
}

pub struct Upload {
    pub context: Arc<GpuContext>,
}

impl Upload {
    pub fn adapter(context: Arc<GpuContext>) -> Adapter {
        Adapter {
            template: domain_transfer(Domain::System, Domain::Wgpu),
            build: Box::new(move |target| {
                demands_domain(target, Domain::Wgpu).then(|| {
                    Box::new(Upload {
                        context: context.clone(),
                    }) as Box<dyn Transform>
                })
            }),
        }
    }
}

impl Transform for Upload {
    fn identity(&self) -> String {
        "Upload".into()
    }

    fn constraint(&self) -> Constraint {
        domain_transfer(Domain::System, Domain::Wgpu)
    }

    fn plan(&self, out: &WindowReq) -> WindowReq {
        *out
    }

    fn compute(&self, out: &WindowReq, input: &Chunk) -> Result<Chunk> {
        let input = input.tensor()?.crop_to(&out.bbox)?;
        let buffer = self.context.upload(input.system())?;
        Ok(Chunk::Tensor(TensorChunk {
            data: TensorData::Wgpu {
                buffer,
                context: self.context.clone(),
            },
            ..input
        }))
    }
}

pub struct Download;

impl Download {
    pub fn adapter() -> Adapter {
        Adapter {
            template: domain_transfer(Domain::Wgpu, Domain::System),
            build: Box::new(|target| {
                demands_domain(target, Domain::System)
                    .then(|| Box::new(Download) as Box<dyn Transform>)
            }),
        }
    }
}

impl Transform for Download {
    fn identity(&self) -> String {
        "Download".into()
    }

    fn constraint(&self) -> Constraint {
        domain_transfer(Domain::Wgpu, Domain::System)
    }

    fn plan(&self, out: &WindowReq) -> WindowReq {
        *out
    }

    fn compute(&self, out: &WindowReq, input: &Chunk) -> Result<Chunk> {
        let input = input.tensor()?.crop_to(&out.bbox)?;
        let (buffer, context) = input.wgpu();
        let values = context.download(buffer)?;
        Ok(Chunk::Tensor(TensorChunk {
            data: TensorData::System(values),
            ..input
        }))
    }
}

pub struct WgpuConvolve {
    pub kernel: [[f32; 3]; 3],
}

impl Transform for WgpuConvolve {
    fn identity(&self) -> String {
        format!("WgpuConvolve kernel={:?}", self.kernel)
    }

    fn constraint(&self) -> Constraint {
        Constraint::Identity(CapsSet::one(tensor_in_domain(Domain::Wgpu)))
    }

    fn plan(&self, out: &WindowReq) -> WindowReq {
        out.with_window(out.bbox.expand(out.resolution), out.resolution)
    }

    fn spread(&self, dirty: &Bbox, resolution: f64) -> Bbox {
        dirty.expand(resolution)
    }

    fn compute(&self, out: &WindowReq, input: &Chunk) -> Result<Chunk> {
        let input = input.tensor()?;
        let (buffer, context) = input.wgpu();
        let convolved = context.convolve(&Convolution {
            source: buffer,
            window: ConvolutionWindow::new(out, input),
            channels: input.channels as usize,
            kernel: self.kernel,
        })?;
        Ok(Chunk::Tensor(TensorChunk {
            data: TensorData::Wgpu {
                buffer: convolved,
                context: context.clone(),
            },
            channels: input.channels,
            bbox: out.bbox,
            resolution: out.resolution,
            crs: input.crs,
        }))
    }
}
