use std::fmt;
use std::sync::mpsc;

use futures::executor::block_on;

use crate::elements::tensor::ConvolutionWindow;
use crate::error::{Error, Result};

const WORKGROUP_SIDE: usize = 8;
const TENSOR_USAGE: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE
    .union(wgpu::BufferUsages::COPY_SRC)
    .union(wgpu::BufferUsages::COPY_DST);

pub struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    budget_bytes: usize,
    copy_window_pipeline: wgpu::ComputePipeline,
    convolve_pipeline: wgpu::ComputePipeline,
}

impl fmt::Debug for GpuContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GpuContext")
            .field("budget_bytes", &self.budget_bytes)
            .finish_non_exhaustive()
    }
}

pub(crate) struct WindowCopy<'a> {
    pub source: &'a wgpu::Buffer,
    pub source_cols: usize,
    pub source_rows: usize,
    pub destination: &'a wgpu::Buffer,
    pub destination_cols: usize,
    pub destination_rows: usize,
    pub channels: usize,
    pub source_col0: usize,
    pub source_row0: usize,
    pub destination_col0: usize,
    pub destination_row0: usize,
    pub cols: usize,
    pub rows: usize,
}

pub(crate) struct Convolution<'a> {
    pub source: &'a wgpu::Buffer,
    pub window: ConvolutionWindow,
    pub channels: usize,
    pub kernel: [[f32; 3]; 3],
}

impl GpuContext {
    pub fn new(device: wgpu::Device, queue: wgpu::Queue, budget_bytes: usize) -> GpuContext {
        let copy_window_pipeline =
            compute_pipeline(&device, "copy_window", include_str!("copy_window.wgsl"));
        let convolve_pipeline =
            compute_pipeline(&device, "convolve", include_str!("convolve.wgsl"));
        GpuContext {
            device,
            queue,
            budget_bytes,
            copy_window_pipeline,
            convolve_pipeline,
        }
    }

    pub fn request(budget_bytes: usize) -> Result<GpuContext> {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = block_on(choose_adapter(&instance))?;
        let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
            .map_err(|e| Error::Gpu(format!("wgpu device: {e}")))?;
        Ok(GpuContext::new(device, queue, budget_bytes))
    }

    pub fn budget_bytes(&self) -> usize {
        self.budget_bytes
    }

    pub(crate) fn upload(&self, data: &[f32]) -> Result<wgpu::Buffer> {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.scoped(|| self.initialized_buffer(&bytes, TENSOR_USAGE))
    }

    pub(crate) fn nan_buffer(&self, len: usize) -> Result<wgpu::Buffer> {
        let bytes = f32::NAN.to_le_bytes().repeat(len);
        self.scoped(|| self.initialized_buffer(&bytes, TENSOR_USAGE))
    }

    pub(crate) fn download(&self, buffer: &wgpu::Buffer) -> Result<Vec<f32>> {
        let size = buffer.size();
        if size == 0 {
            return Ok(Vec::new());
        }
        self.scoped(|| {
            let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("download"),
                size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("download"),
                });
            encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
            self.queue.submit([encoder.finish()]);
            let (sender, receiver) = mpsc::channel();
            staging.map_async(wgpu::MapMode::Read, .., move |mapped| {
                let _ = sender.send(mapped);
            });
            self.device
                .poll(wgpu::PollType::wait_indefinitely())
                .map_err(|e| Error::Gpu(format!("poll: {e}")))?;
            receiver
                .recv()
                .map_err(|_| Error::Gpu("download map callback never ran".into()))?
                .map_err(|e| Error::Gpu(format!("download map: {e}")))?;
            let values = {
                let view = staging
                    .get_mapped_range(..)
                    .map_err(|e| Error::Gpu(format!("download view: {e}")))?;
                let (words, _) = view.as_chunks::<4>();
                words.iter().map(|word| f32::from_le_bytes(*word)).collect()
            };
            staging.unmap();
            Ok(values)
        })
    }

    pub(crate) fn copy_window(&self, copy: &WindowCopy<'_>) -> Result<()> {
        if copy.cols == 0 || copy.rows == 0 || copy.channels == 0 {
            return Ok(());
        }
        let mut params = Vec::with_capacity(48);
        for value in [
            copy.source_cols,
            copy.source_rows,
            copy.destination_cols,
            copy.destination_rows,
            copy.channels,
            copy.source_col0,
            copy.source_row0,
            copy.destination_col0,
            copy.destination_row0,
            copy.cols,
            copy.rows,
            0,
        ] {
            params.extend(shader_word(value)?.to_le_bytes());
        }
        self.scoped(|| {
            self.dispatch(
                &self.copy_window_pipeline,
                &params,
                copy.source,
                copy.destination,
                [copy.cols, copy.rows, copy.channels],
            )
        })
    }

    pub(crate) fn convolve(&self, convolution: &Convolution<'_>) -> Result<wgpu::Buffer> {
        let window = &convolution.window;
        let len = convolution.channels * window.output_cols * window.output_rows;
        let mut params = Vec::with_capacity(80);
        for value in [window.input_cols, window.input_rows, convolution.channels] {
            params.extend(shader_word(value)?.to_le_bytes());
        }
        for value in [window.col0, window.row0] {
            params.extend(signed_shader_word(value)?.to_le_bytes());
        }
        for value in [window.output_cols, window.output_rows, 0] {
            params.extend(shader_word(value)?.to_le_bytes());
        }
        // a uniform array element is sixteen bytes, so each kernel row is a padded vec4
        for row in convolution.kernel {
            for value in [row[0], row[1], row[2], 0.0] {
                params.extend(value.to_le_bytes());
            }
        }
        self.scoped(|| {
            let output = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("convolve"),
                size: (len * size_of::<f32>()) as wgpu::BufferAddress,
                usage: TENSOR_USAGE,
                mapped_at_creation: false,
            });
            if len > 0 {
                self.dispatch(
                    &self.convolve_pipeline,
                    &params,
                    convolution.source,
                    &output,
                    [window.output_cols, window.output_rows, convolution.channels],
                )?;
            }
            Ok(output)
        })
    }

    fn dispatch(
        &self,
        pipeline: &wgpu::ComputePipeline,
        params: &[u8],
        source: &wgpu::Buffer,
        destination: &wgpu::Buffer,
        invocations: [usize; 3],
    ) -> Result<()> {
        let workgroups_x = shader_word(invocations[0].div_ceil(WORKGROUP_SIDE))?;
        let workgroups_y = shader_word(invocations[1].div_ceil(WORKGROUP_SIDE))?;
        let workgroups_z = shader_word(invocations[2])?;
        let uniform = self.initialized_buffer(params, wgpu::BufferUsages::UNIFORM)?;
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: source.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: destination.as_entire_binding(),
                },
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
        }
        self.queue.submit([encoder.finish()]);
        Ok(())
    }

    fn initialized_buffer(&self, bytes: &[u8], usage: wgpu::BufferUsages) -> Result<wgpu::Buffer> {
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: bytes.len() as wgpu::BufferAddress,
            usage,
            mapped_at_creation: !bytes.is_empty(),
        });
        if bytes.is_empty() {
            return Ok(buffer);
        }
        buffer
            .get_mapped_range_mut(..)
            .map_err(|e| Error::Gpu(format!("buffer init: {e}")))?
            .copy_from_slice(bytes);
        buffer.unmap();
        Ok(buffer)
    }

    // wgpu's default handler panics on an uncaptured error, out of device memory included
    fn scoped<T>(&self, work: impl FnOnce() -> Result<T>) -> Result<T> {
        let out_of_memory = self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let validation = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let result = work();
        let validation_error = block_on(validation.pop());
        let out_of_memory_error = block_on(out_of_memory.pop());
        match validation_error.or(out_of_memory_error) {
            Some(error) => Err(Error::Gpu(error.to_string())),
            None => result,
        }
    }
}

// wgpu's own WGPU_ADAPTER_NAME filter panics when the name matches no adapter
async fn choose_adapter(instance: &wgpu::Instance) -> Result<wgpu::Adapter> {
    let Ok(wanted) = std::env::var("WGPU_ADAPTER_NAME") else {
        return instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::from_env().unwrap_or_default(),
                ..Default::default()
            })
            .await
            .map_err(|e| Error::Gpu(format!("no wgpu adapter: {e}")));
    };
    let wanted = wanted.to_lowercase();
    instance
        .enumerate_adapters(wgpu::Backends::all())
        .await
        .into_iter()
        .find(|adapter| adapter.get_info().name.to_lowercase().contains(&wanted))
        .ok_or_else(|| Error::Gpu(format!("no wgpu adapter named {wanted:?}")))
}

fn compute_pipeline(device: &wgpu::Device, label: &str, source: &str) -> wgpu::ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

fn shader_word(value: usize) -> Result<u32> {
    u32::try_from(value).map_err(|_| Error::Gpu(format!("{value} does not fit a shader u32")))
}

fn signed_shader_word(value: isize) -> Result<i32> {
    i32::try_from(value).map_err(|_| Error::Gpu(format!("{value} does not fit a shader i32")))
}
