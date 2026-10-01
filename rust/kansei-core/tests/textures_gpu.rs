//! Texture arrays and device limits on a real GPU: layers uploaded by `Texture::from_rgba_layers`
//! read back through a `Binding::texture_2d_array`, and a device requested with
//! `RequiredLimits::Adapter` binding more sampled textures than WebGPU's default 16. Skipped
//! (passes) when no adapter is available.

use kansei_core::buffers::{Bindable, Texture};
use kansei_core::materials::{BindGroupBuilder, Binding, BindingResource};
use kansei_core::renderers::RequiredLimits;

fn adapter() -> Option<wgpu::Adapter> {
    let instance = wgpu::Instance::default();
    pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
}

fn request_device(adapter: &wgpu::Adapter, limits: &RequiredLimits) -> (wgpu::Device, wgpu::Queue) {
    pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor { required_limits: limits.resolve(&adapter.limits()), ..Default::default() },
        None,
    ))
    .unwrap()
}

#[test]
fn rgba_layers_upload_into_their_own_array_layers() {
    let Some(adapter) = adapter() else { return eprintln!("no GPU adapter: skipping") };
    let (device, queue) = request_device(&adapter, &RequiredLimits::Default);
    let (w, h) = (4u32, 4u32);
    let colours = [[255u8, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]];
    let layers: Vec<Vec<u8>> = colours.iter().map(|c| c.repeat((w * h) as usize)).collect();
    let refs: Vec<&[u8]> = layers.iter().map(|l| l.as_slice()).collect();
    let mut texture = Texture::from_rgba_layers("Layers", w, h, &refs);
    texture.ensure_ready(&device, &queue);

    // read texel (1, 2) of every layer through a texture_2d_array binding
    let out = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 48,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 48,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let bindings = [Binding::texture_2d_array(0, wgpu::ShaderStages::COMPUTE), Binding::storage(1, wgpu::ShaderStages::COMPUTE, false)];
    let layout = BindGroupBuilder::create_layout(&device, "Layers", &bindings);
    let group = BindGroupBuilder::create_bind_group(
        &device,
        "Layers",
        &layout,
        &[(0, texture.binding_resource().unwrap()), (1, BindingResource::Buffer { buffer: &out, offset: 0, size: None })],
    );
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(
            r#"
            @group(0) @binding(0) var layers : texture_2d_array<f32>;
            @group(0) @binding(1) var<storage, read_write> out : array<vec4f, 3>;
            @compute @workgroup_size(1) fn main() {
                for (var i = 0; i < 3; i++) { out[i] = textureLoad(layers, vec2i(1, 2), i, 0); }
            }
            "#
            .into(),
        ),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&layout], push_constant_ranges: &[] });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&out, 0, &readback, 0, 48);
    queue.submit(std::iter::once(encoder.finish()));
    readback.slice(..).map_async(wgpu::MapMode::Read, |r| r.unwrap());
    device.poll(wgpu::Maintain::Wait);
    let values: Vec<f32> = bytemuck::cast_slice(&readback.slice(..).get_mapped_range()).to_vec();
    for (layer, colour) in colours.iter().enumerate() {
        let got = &values[layer * 4..layer * 4 + 4];
        let want: Vec<f32> = colour.iter().map(|&c| c as f32 / 255.0).collect();
        assert!(got.iter().zip(&want).all(|(a, b)| (a - b).abs() < 1e-3), "layer {layer}: {got:?} vs {want:?}");
    }
}

#[test]
fn adapter_limits_bind_more_textures_than_the_default() {
    let Some(adapter) = adapter() else { return eprintln!("no GPU adapter: skipping") };
    let supported = adapter.limits().max_sampled_textures_per_shader_stage;
    let defaults = wgpu::Limits::default().max_sampled_textures_per_shader_stage;
    if supported <= defaults {
        return eprintln!("adapter supports only {supported} sampled textures: skipping");
    }
    let (device, _queue) = request_device(&adapter, &RequiredLimits::Adapter);
    assert_eq!(device.limits().max_sampled_textures_per_shader_stage, supported);
    // one more texture than WebGPU's default allows, in a single fragment stage
    let bindings: Vec<Binding> = (0..=defaults).map(|i| Binding::texture_2d(i, wgpu::ShaderStages::FRAGMENT)).collect();
    device.push_error_scope(wgpu::ErrorFilter::Validation);
    let _layout = BindGroupBuilder::create_layout(&device, "ManyTextures", &bindings);
    let error = pollster::block_on(device.pop_error_scope());
    assert!(error.is_none(), "{error:?}");
    // and the default limits refuse it
    let (default_device, _queue) = request_device(&adapter, &RequiredLimits::Default);
    default_device.push_error_scope(wgpu::ErrorFilter::Validation);
    let _layout = BindGroupBuilder::create_layout(&default_device, "ManyTextures", &bindings);
    assert!(pollster::block_on(default_device.pop_error_scope()).is_some());
    eprintln!("{} sampled textures bound (adapter supports {supported})", defaults + 1);
}
