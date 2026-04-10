use crate::cameras::Camera;
use crate::postprocessing::PostProcessingEffect;
use crate::renderers::GBuffer;

pub struct DepthOfFieldOptions {
    pub focus_distance: f32,
    pub focus_range: f32,
    pub max_blur: f32,
}

impl Default for DepthOfFieldOptions {
    fn default() -> Self {
        Self { focus_distance: 50.0, focus_range: 20.0, max_blur: 14.0 }
    }
}

// ── DoF Params uniform: 32 bytes ──
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DoFParamsGpu {
    focus_distance: f32,
    focus_range: f32,
    max_blur: f32,
    near: f32,
    far: f32,
    screen_width: f32,
    screen_height: f32,
    _pad: f32,
}

// ── Shader sources ──

const COMMON: &str = r#"
struct DoFParams {
    focusDistance : f32,
    focusRange   : f32,
    maxBlur      : f32,
    near         : f32,
    far          : f32,
    screenWidth  : f32,
    screenHeight : f32,
    _pad         : f32,
}
fn linearDepth(d: f32, n: f32, f: f32) -> f32 {
    // glam::perspective_rh maps depth to [0,1] (WebGPU convention)
    return n * f / (f - d * (f - n));
}
fn computeCoC(d: f32, p: DoFParams) -> f32 {
    let ld = linearDepth(d, p.near, p.far);
    return clamp((ld - p.focusDistance) / p.focusRange, -1.0, 1.0) * p.maxBlur;
}
"#;

const COC_SHADER: &str = r#"
@group(0) @binding(0) var depthTex : texture_depth_2d;
@group(0) @binding(1) var cocOut   : texture_storage_2d<r32float, write>;
@group(0) @binding(2) var<uniform> params : DoFParams;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    let coord = gid.xy;
    let w = u32(params.screenWidth); let h = u32(params.screenHeight);
    if (coord.x >= w || coord.y >= h) { return; }
    let depth = textureLoad(depthTex, coord, 0);
    let coc = computeCoC(depth, params);
    textureStore(cocOut, coord, vec4f(coc, 0.0, 0.0, 0.0));
}
"#;

const DILATE_H_SHADER: &str = r#"
@group(0) @binding(0) var cocIn  : texture_2d<f32>;
@group(0) @binding(1) var cocOut : texture_storage_2d<r32float, write>;
@group(0) @binding(2) var<uniform> params : DoFParams;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    let coord = gid.xy;
    let w = u32(params.screenWidth); let h = u32(params.screenHeight);
    if (coord.x >= w || coord.y >= h) { return; }
    let ownCoc = textureLoad(cocIn, coord, 0).r;
    var nearR = max(0.0, -ownCoc);
    let radius = i32(params.maxBlur); let iCoord = vec2i(coord);
    for (var dx = -radius; dx <= radius; dx++) {
        let sx = clamp(iCoord.x + dx, 0, i32(w) - 1);
        let sCoc = textureLoad(cocIn, vec2u(u32(sx), coord.y), 0).r;
        if (sCoc < 0.0) {
            let sR = -sCoc; let dist = f32(abs(dx));
            if (sR >= dist) { let t = dist / sR; nearR = max(nearR, sR * (1.0 - t) * (1.0 - t)); }
        }
    }
    textureStore(cocOut, coord, vec4f(max(nearR, abs(ownCoc)), 0.0, 0.0, 0.0));
}
"#;

const DILATE_V_SHADER: &str = r#"
@group(0) @binding(0) var cocIn  : texture_2d<f32>;
@group(0) @binding(1) var cocOut : texture_storage_2d<r32float, write>;
@group(0) @binding(2) var<uniform> params : DoFParams;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    let coord = gid.xy;
    let w = u32(params.screenWidth); let h = u32(params.screenHeight);
    if (coord.x >= w || coord.y >= h) { return; }
    let ownCoc = textureLoad(cocIn, coord, 0).r;
    var nearR = max(0.0, -ownCoc);
    let radius = i32(params.maxBlur); let iCoord = vec2i(coord);
    for (var dy = -radius; dy <= radius; dy++) {
        let sy = clamp(iCoord.y + dy, 0, i32(h) - 1);
        let sCoc = textureLoad(cocIn, vec2u(coord.x, u32(sy)), 0).r;
        if (sCoc < 0.0) {
            let sR = -sCoc; let dist = f32(abs(dy));
            if (sR >= dist) { let t = dist / sR; nearR = max(nearR, sR * (1.0 - t) * (1.0 - t)); }
        }
    }
    textureStore(cocOut, coord, vec4f(max(nearR, abs(ownCoc)), 0.0, 0.0, 0.0));
}
"#;

const DOWNSAMPLE_SHADER: &str = r#"
@group(0) @binding(0) var colorTex : texture_2d<f32>;
@group(0) @binding(1) var depthTex : texture_depth_2d;
@group(0) @binding(2) var nearOut  : texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var farOut   : texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var<uniform> params : DoFParams;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    let halfCoord = gid.xy;
    let halfW = u32(ceil(params.screenWidth * 0.5));
    let halfH = u32(ceil(params.screenHeight * 0.5));
    if (halfCoord.x >= halfW || halfCoord.y >= halfH) { return; }
    let fullW = u32(params.screenWidth); let fullH = u32(params.screenHeight);
    let base = halfCoord * 2u;
    var nearAccum = vec4f(0.0); var farAccum = vec4f(0.0); var farWeight = 0.0;
    for (var dy = 0u; dy < 2u; dy++) { for (var dx = 0u; dx < 2u; dx++) {
        let fc = vec2u(min(base.x + dx, fullW - 1u), min(base.y + dy, fullH - 1u));
        let color = textureLoad(colorTex, fc, 0);
        let depth = textureLoad(depthTex, fc, 0);
        let origCoc = computeCoC(depth, params);
        if (origCoc < 0.0) {
            let coverage = saturate(abs(origCoc) / (params.maxBlur * 0.5));
            nearAccum += vec4f(color.rgb * coverage, coverage);
        } else {
            let normCoc = abs(origCoc) / max(params.maxBlur, 0.001);
            let w = smoothstep(0.0, 0.15, normCoc);
            farAccum += vec4f(color.rgb * w, normCoc * w); farWeight += w;
        }
    }}
    nearAccum *= 0.25;
    if (farWeight > 0.001) { farAccum /= farWeight; } else { farAccum = vec4f(0.0); }
    textureStore(nearOut, halfCoord, nearAccum);
    textureStore(farOut, halfCoord, farAccum);
}
"#;

const BLUR_SHADER: &str = r#"
@group(0) @binding(0) var nearIn     : texture_2d<f32>;
@group(0) @binding(1) var farIn      : texture_2d<f32>;
@group(0) @binding(2) var cocDilated  : texture_2d<f32>;
@group(0) @binding(3) var nearOut    : texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var farOut     : texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var<uniform> params : DoFParams;
const GOLDEN_ANGLE : f32 = 2.39996323;
const NUM_SAMPLES : u32 = 64u;
fn vogelDisk(i: u32, n: u32) -> vec2f {
    let r = sqrt((f32(i) + 0.5) / f32(n));
    let theta = f32(i) * GOLDEN_ANGLE;
    return vec2f(cos(theta), sin(theta)) * r;
}
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    let halfCoord = gid.xy;
    let halfW = u32(ceil(params.screenWidth * 0.5));
    let halfH = u32(ceil(params.screenHeight * 0.5));
    if (halfCoord.x >= halfW || halfCoord.y >= halfH) { return; }
    let base = halfCoord * 2u;
    let fullW = u32(params.screenWidth); let fullH = u32(params.screenHeight);
    var maxCoc = 0.0;
    for (var dy = 0u; dy < 2u; dy++) { for (var dx = 0u; dx < 2u; dx++) {
        let fc = vec2u(min(base.x + dx, fullW - 1u), min(base.y + dy, fullH - 1u));
        maxCoc = max(maxCoc, abs(textureLoad(cocDilated, fc, 0).r));
    }}
    var halfR = maxCoc * 0.5;
    if (halfR < 0.5) {
        let farProbe = textureLoad(farIn, halfCoord, 0);
        if (farProbe.a < 0.01) {
            let searchMax = params.maxBlur * 0.5;
            let sLim = vec2i(i32(halfW) - 1, i32(halfH) - 1);
            for (var fi = 0u; fi < 16u; fi++) {
                let foff = vogelDisk(fi, 16u) * searchMax;
                let fsc = clamp(vec2i(halfCoord) + vec2i(i32(foff.x), i32(foff.y)), vec2i(0), sLim);
                let fs = textureLoad(farIn, vec2u(fsc), 0);
                if (fs.a > 0.01) { halfR = max(halfR, fs.a * params.maxBlur * 0.5); }
            }
        }
    }
    if (halfR < 0.5) {
        textureStore(nearOut, halfCoord, textureLoad(nearIn, halfCoord, 0));
        textureStore(farOut, halfCoord, textureLoad(farIn, halfCoord, 0));
        return;
    }
    let centerFarCocHalf = textureLoad(farIn, halfCoord, 0).a * params.maxBlur * 0.5;
    var nearAccum = vec4f(0.0); var nearCount = 0.0;
    var farAccum = vec4f(0.0); var farWeight = 0.0;
    let limHalf = vec2i(i32(halfW) - 1, i32(halfH) - 1);
    for (var i = 0u; i < NUM_SAMPLES; i++) {
        let off = vogelDisk(i, NUM_SAMPLES) * halfR;
        let sc = clamp(vec2i(halfCoord) + vec2i(i32(off.x), i32(off.y)), vec2i(0), limHalf);
        let scu = vec2u(sc);
        nearAccum += textureLoad(nearIn, scu, 0); nearCount += 1.0;
        let farSample = textureLoad(farIn, scu, 0);
        let sampleFarCocHalf = farSample.a * params.maxBlur * 0.5;
        let dist = length(off);
        let sampleCovers = smoothstep(dist - 1.0, dist + 1.0, sampleFarCocHalf);
        let centerCovers = smoothstep(dist - 1.0, dist + 1.0, centerFarCocHalf);
        let cocWt = smoothstep(0.0, 0.15, farSample.a);
        let coverage = max(sampleCovers, centerCovers) * cocWt;
        farAccum += vec4f(farSample.rgb * coverage, farSample.a * coverage);
        farWeight += coverage;
    }
    if (nearCount > 0.0) { nearAccum /= nearCount; }
    if (farWeight > 0.001) { farAccum = vec4f(farAccum.rgb / farWeight, farAccum.a / farWeight); }
    textureStore(nearOut, halfCoord, nearAccum);
    textureStore(farOut, halfCoord, farAccum);
}
"#;

const COMPOSITE_SHADER: &str = r#"
@group(0) @binding(0) var colorTex   : texture_2d<f32>;
@group(0) @binding(1) var depthTex   : texture_depth_2d;
@group(0) @binding(2) var nearBlurTex : texture_2d<f32>;
@group(0) @binding(3) var farBlurTex  : texture_2d<f32>;
@group(0) @binding(4) var outputTex  : texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var<uniform> params : DoFParams;
struct HalfResCoords { ix0: u32, iy0: u32, ix1: u32, iy1: u32, frx: f32, fry: f32, }
fn halfResCoords(fullCoord: vec2u, halfW: u32, halfH: u32) -> HalfResCoords {
    let hc = (vec2f(f32(fullCoord.x), f32(fullCoord.y)) + 0.5) * 0.5 - 0.5;
    let fl = floor(hc); let fr = hc - fl;
    return HalfResCoords(u32(max(i32(fl.x),0)), u32(max(i32(fl.y),0)),
        min(u32(max(i32(fl.x),0))+1u, halfW-1u), min(u32(max(i32(fl.y),0))+1u, halfH-1u), fr.x, fr.y);
}
fn sampleBilinear(tex: texture_2d<f32>, fullCoord: vec2u, halfW: u32, halfH: u32) -> vec4f {
    let c = halfResCoords(fullCoord, halfW, halfH);
    let s00 = textureLoad(tex, vec2u(c.ix0,c.iy0), 0); let s10 = textureLoad(tex, vec2u(c.ix1,c.iy0), 0);
    let s01 = textureLoad(tex, vec2u(c.ix0,c.iy1), 0); let s11 = textureLoad(tex, vec2u(c.ix1,c.iy1), 0);
    return mix(mix(s00, s10, c.frx), mix(s01, s11, c.frx), c.fry);
}
fn sampleBilinearFar(tex: texture_2d<f32>, fullCoord: vec2u, halfW: u32, halfH: u32) -> vec4f {
    let c = halfResCoords(fullCoord, halfW, halfH);
    let s00 = textureLoad(tex, vec2u(c.ix0,c.iy0), 0); let s10 = textureLoad(tex, vec2u(c.ix1,c.iy0), 0);
    let s01 = textureLoad(tex, vec2u(c.ix0,c.iy1), 0); let s11 = textureLoad(tex, vec2u(c.ix1,c.iy1), 0);
    let bw00=(1.0-c.frx)*(1.0-c.fry); let bw10=c.frx*(1.0-c.fry);
    let bw01=(1.0-c.frx)*c.fry; let bw11=c.frx*c.fry;
    let w00=bw00*s00.a; let w10=bw10*s10.a; let w01=bw01*s01.a; let w11=bw11*s11.a;
    let total=w00+w10+w01+w11;
    if (total < 0.001) { return vec4f(0.0); }
    return (s00*w00 + s10*w10 + s01*w01 + s11*w11) / total;
}
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    let coord = gid.xy;
    let w = u32(params.screenWidth); let h = u32(params.screenHeight);
    if (coord.x >= w || coord.y >= h) { return; }
    let sharp = textureLoad(colorTex, coord, 0);
    let depth = textureLoad(depthTex, coord, 0);
    let coc = computeCoC(depth, params); let absCoc = abs(coc);
    let halfW = u32(ceil(params.screenWidth * 0.5));
    let halfH = u32(ceil(params.screenHeight * 0.5));
    let nearBlur = sampleBilinear(nearBlurTex, coord, halfW, halfH);
    let nearAlpha = saturate(nearBlur.a * 2.0);
    if (absCoc < 1.0 && nearAlpha < 0.001) { textureStore(outputTex, coord, sharp); return; }
    let farBlur = sampleBilinearFar(farBlurTex, coord, halfW, halfH);
    var result = sharp;
    if (coc >= 0.0) {
        let farMix = smoothstep(1.0, max(params.maxBlur * 0.3, 4.0), absCoc);
        result = mix(sharp, vec4f(farBlur.rgb, sharp.a), farMix);
    }
    if (nearAlpha > 0.001) {
        let nearRgb = nearBlur.rgb / max(nearBlur.a, 0.001);
        result = vec4f(mix(result.rgb, nearRgb, nearAlpha), result.a);
    }
    textureStore(outputTex, coord, result);
}
"#;

// ── Helper: create a compute pipeline from WGSL ──
fn make_pipeline(
    device: &wgpu::Device,
    label: &str,
    source: &str,
    bgl: &wgpu::BindGroupLayout,
) -> wgpu::ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(format!("{COMMON}\n{source}").into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[bgl],
        push_constant_ranges: &[],
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

fn mk_tex(device: &wgpu::Device, label: &str, w: u32, h: u32, format: wgpu::TextureFormat) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1, sample_count: 1,
        dimension: wgpu::TextureDimension::D2, format,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = tex.create_view(&Default::default());
    (tex, view)
}

// ── Effect struct ──

pub struct DepthOfFieldEffect {
    pub options: DepthOfFieldOptions,
    // Pipelines
    coc_pipeline: Option<wgpu::ComputePipeline>,
    dilate_h_pipeline: Option<wgpu::ComputePipeline>,
    dilate_v_pipeline: Option<wgpu::ComputePipeline>,
    downsample_pipeline: Option<wgpu::ComputePipeline>,
    blur_pipeline: Option<wgpu::ComputePipeline>,
    composite_pipeline: Option<wgpu::ComputePipeline>,
    // BGLs
    coc_bgl: Option<wgpu::BindGroupLayout>,
    dilate_bgl: Option<wgpu::BindGroupLayout>,
    downsample_bgl: Option<wgpu::BindGroupLayout>,
    blur_bgl: Option<wgpu::BindGroupLayout>,
    composite_bgl: Option<wgpu::BindGroupLayout>,
    // Internal textures
    coc_tex: Option<wgpu::Texture>, coc_view: Option<wgpu::TextureView>,
    coc_dil_temp_tex: Option<wgpu::Texture>, coc_dil_temp_view: Option<wgpu::TextureView>,
    near_half_tex: Option<wgpu::Texture>, near_half_view: Option<wgpu::TextureView>,
    far_half_tex: Option<wgpu::Texture>, far_half_view: Option<wgpu::TextureView>,
    near_blur_tex: Option<wgpu::Texture>, near_blur_view: Option<wgpu::TextureView>,
    far_blur_tex: Option<wgpu::Texture>, far_blur_view: Option<wgpu::TextureView>,
    // Params buffer
    params_buf: Option<wgpu::Buffer>,
    width: u32, height: u32,
    initialized: bool,
    // Track input/depth/output for bind group invalidation
    current_input: usize,
    current_depth: usize,
    current_output: usize,
    // Bind groups (recreated when textures change)
    coc_bg: Option<wgpu::BindGroup>,
    dilate_h_bg: Option<wgpu::BindGroup>,
    dilate_v_bg: Option<wgpu::BindGroup>,
    downsample_bg: Option<wgpu::BindGroup>,
    blur_bg: Option<wgpu::BindGroup>,
    composite_bg: Option<wgpu::BindGroup>,
}

impl DepthOfFieldEffect {
    pub fn new(options: DepthOfFieldOptions) -> Self {
        Self {
            options,
            coc_pipeline: None, dilate_h_pipeline: None, dilate_v_pipeline: None,
            downsample_pipeline: None, blur_pipeline: None, composite_pipeline: None,
            coc_bgl: None, dilate_bgl: None, downsample_bgl: None, blur_bgl: None, composite_bgl: None,
            coc_tex: None, coc_view: None, coc_dil_temp_tex: None, coc_dil_temp_view: None,
            near_half_tex: None, near_half_view: None, far_half_tex: None, far_half_view: None,
            near_blur_tex: None, near_blur_view: None, far_blur_tex: None, far_blur_view: None,
            params_buf: None, width: 0, height: 0, initialized: false,
            current_input: 0, current_depth: 0, current_output: 0,
            coc_bg: None, dilate_h_bg: None, dilate_v_bg: None,
            downsample_bg: None, blur_bg: None, composite_bg: None,
        }
    }

    fn create_textures(&mut self, device: &wgpu::Device, w: u32, h: u32) {
        let hw = (w + 1) / 2;
        let hh = (h + 1) / 2;
        let (t, v) = mk_tex(device, "DoF/CoC", w, h, wgpu::TextureFormat::R32Float);
        self.coc_tex = Some(t); self.coc_view = Some(v);
        let (t, v) = mk_tex(device, "DoF/CoCDilTemp", w, h, wgpu::TextureFormat::R32Float);
        self.coc_dil_temp_tex = Some(t); self.coc_dil_temp_view = Some(v);
        let (t, v) = mk_tex(device, "DoF/NearHalf", hw, hh, wgpu::TextureFormat::Rgba16Float);
        self.near_half_tex = Some(t); self.near_half_view = Some(v);
        let (t, v) = mk_tex(device, "DoF/FarHalf", hw, hh, wgpu::TextureFormat::Rgba16Float);
        self.far_half_tex = Some(t); self.far_half_view = Some(v);
        let (t, v) = mk_tex(device, "DoF/NearBlur", hw, hh, wgpu::TextureFormat::Rgba16Float);
        self.near_blur_tex = Some(t); self.near_blur_view = Some(v);
        let (t, v) = mk_tex(device, "DoF/FarBlur", hw, hh, wgpu::TextureFormat::Rgba16Float);
        self.far_blur_tex = Some(t); self.far_blur_view = Some(v);
        self.width = w;
        self.height = h;
        // Invalidate bind groups
        self.coc_bg = None; self.dilate_h_bg = None; self.dilate_v_bg = None;
        self.downsample_bg = None; self.blur_bg = None; self.composite_bg = None;
        self.current_input = 0; self.current_depth = 0; self.current_output = 0;
    }

    fn create_bind_group_layouts(device: &wgpu::Device) -> (
        wgpu::BindGroupLayout, wgpu::BindGroupLayout, wgpu::BindGroupLayout,
        wgpu::BindGroupLayout, wgpu::BindGroupLayout,
    ) {
        use wgpu::*;
        let tex2d = |binding, sample_type: TextureSampleType| BindGroupLayoutEntry {
            binding, visibility: ShaderStages::COMPUTE,
            ty: BindingType::Texture { sample_type, view_dimension: TextureViewDimension::D2, multisampled: false },
            count: None,
        };
        let depth2d = |binding| BindGroupLayoutEntry {
            binding, visibility: ShaderStages::COMPUTE,
            ty: BindingType::Texture { sample_type: TextureSampleType::Depth, view_dimension: TextureViewDimension::D2, multisampled: false },
            count: None,
        };
        let storage_w = |binding, format: TextureFormat| BindGroupLayoutEntry {
            binding, visibility: ShaderStages::COMPUTE,
            ty: BindingType::StorageTexture { access: StorageTextureAccess::WriteOnly, format, view_dimension: TextureViewDimension::D2 },
            count: None,
        };
        let uniform = |binding| BindGroupLayoutEntry {
            binding, visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer { ty: BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let flt = TextureSampleType::Float { filterable: false };

        let coc_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("DoF/CoCBGL"),
            entries: &[depth2d(0), storage_w(1, TextureFormat::R32Float), uniform(2)],
        });
        let dilate_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("DoF/DilateBGL"),
            entries: &[tex2d(0, flt), storage_w(1, TextureFormat::R32Float), uniform(2)],
        });
        let downsample_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("DoF/DownsampleBGL"),
            entries: &[tex2d(0, flt), depth2d(1), storage_w(2, TextureFormat::Rgba16Float), storage_w(3, TextureFormat::Rgba16Float), uniform(4)],
        });
        let blur_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("DoF/BlurBGL"),
            entries: &[tex2d(0, flt), tex2d(1, flt), tex2d(2, flt), storage_w(3, TextureFormat::Rgba16Float), storage_w(4, TextureFormat::Rgba16Float), uniform(5)],
        });
        let composite_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("DoF/CompositeBGL"),
            entries: &[tex2d(0, flt), depth2d(1), tex2d(2, flt), tex2d(3, flt), storage_w(4, TextureFormat::Rgba16Float), uniform(5)],
        });
        (coc_bgl, dilate_bgl, downsample_bgl, blur_bgl, composite_bgl)
    }
}

impl PostProcessingEffect for DepthOfFieldEffect {
    fn initialize(&mut self, device: &wgpu::Device, gbuffer: &GBuffer, _camera: &Camera) {
        if self.initialized { return; }

        let (coc_bgl, dilate_bgl, downsample_bgl, blur_bgl, composite_bgl) =
            Self::create_bind_group_layouts(device);

        self.coc_pipeline = Some(make_pipeline(device, "DoF/CoC", COC_SHADER, &coc_bgl));
        self.dilate_h_pipeline = Some(make_pipeline(device, "DoF/DilateH", DILATE_H_SHADER, &dilate_bgl));
        self.dilate_v_pipeline = Some(make_pipeline(device, "DoF/DilateV", DILATE_V_SHADER, &dilate_bgl));
        self.downsample_pipeline = Some(make_pipeline(device, "DoF/Downsample", DOWNSAMPLE_SHADER, &downsample_bgl));
        self.blur_pipeline = Some(make_pipeline(device, "DoF/Blur", BLUR_SHADER, &blur_bgl));
        self.composite_pipeline = Some(make_pipeline(device, "DoF/Composite", COMPOSITE_SHADER, &composite_bgl));

        self.coc_bgl = Some(coc_bgl);
        self.dilate_bgl = Some(dilate_bgl);
        self.downsample_bgl = Some(downsample_bgl);
        self.blur_bgl = Some(blur_bgl);
        self.composite_bgl = Some(composite_bgl);

        self.params_buf = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DoF/Params"), size: std::mem::size_of::<DoFParamsGpu>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        self.create_textures(device, gbuffer.width, gbuffer.height);
        self.initialized = true;
    }

    fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        _gbuffer: &GBuffer,
        input: &wgpu::TextureView,
        depth: &wgpu::TextureView,
        output: &wgpu::TextureView,
        camera: &Camera,
        width: u32,
        height: u32,
    ) {
        if !self.initialized { return; }

        // Upload params
        let params = DoFParamsGpu {
            focus_distance: self.options.focus_distance,
            focus_range: self.options.focus_range,
            max_blur: self.options.max_blur,
            near: camera.near,
            far: camera.far,
            screen_width: width as f32,
            screen_height: height as f32,
            _pad: 0.0,
        };
        queue.write_buffer(self.params_buf.as_ref().unwrap(), 0, bytemuck::bytes_of(&params));

        // Recreate bind groups if external textures changed
        let input_ptr = input as *const _ as usize;
        let depth_ptr = depth as *const _ as usize;
        let output_ptr = output as *const _ as usize;
        let needs_rebind = self.coc_bg.is_none()
            || input_ptr != self.current_input
            || depth_ptr != self.current_depth
            || output_ptr != self.current_output;

        if needs_rebind {
            self.current_input = input_ptr;
            self.current_depth = depth_ptr;
            self.current_output = output_ptr;

            let pb = self.params_buf.as_ref().unwrap();

            self.coc_bg = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("DoF/CoCBG"), layout: self.coc_bgl.as_ref().unwrap(),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(depth) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(self.coc_view.as_ref().unwrap()) },
                    wgpu::BindGroupEntry { binding: 2, resource: pb.as_entire_binding() },
                ],
            }));
            self.dilate_h_bg = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("DoF/DilateHBG"), layout: self.dilate_bgl.as_ref().unwrap(),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(self.coc_view.as_ref().unwrap()) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(self.coc_dil_temp_view.as_ref().unwrap()) },
                    wgpu::BindGroupEntry { binding: 2, resource: pb.as_entire_binding() },
                ],
            }));
            self.dilate_v_bg = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("DoF/DilateVBG"), layout: self.dilate_bgl.as_ref().unwrap(),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(self.coc_dil_temp_view.as_ref().unwrap()) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(self.coc_view.as_ref().unwrap()) },
                    wgpu::BindGroupEntry { binding: 2, resource: pb.as_entire_binding() },
                ],
            }));
            self.downsample_bg = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("DoF/DownsampleBG"), layout: self.downsample_bgl.as_ref().unwrap(),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(input) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(depth) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(self.near_half_view.as_ref().unwrap()) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(self.far_half_view.as_ref().unwrap()) },
                    wgpu::BindGroupEntry { binding: 4, resource: pb.as_entire_binding() },
                ],
            }));
            self.blur_bg = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("DoF/BlurBG"), layout: self.blur_bgl.as_ref().unwrap(),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(self.near_half_view.as_ref().unwrap()) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(self.far_half_view.as_ref().unwrap()) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(self.coc_view.as_ref().unwrap()) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(self.near_blur_view.as_ref().unwrap()) },
                    wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(self.far_blur_view.as_ref().unwrap()) },
                    wgpu::BindGroupEntry { binding: 5, resource: pb.as_entire_binding() },
                ],
            }));
            self.composite_bg = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("DoF/CompositeBG"), layout: self.composite_bgl.as_ref().unwrap(),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(input) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(depth) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(self.near_blur_view.as_ref().unwrap()) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(self.far_blur_view.as_ref().unwrap()) },
                    wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(output) },
                    wgpu::BindGroupEntry { binding: 5, resource: pb.as_entire_binding() },
                ],
            }));
        }

        let wg_full = ((width + 7) / 8, (height + 7) / 8);
        let hw = (width + 1) / 2;
        let hh = (height + 1) / 2;
        let wg_half = ((hw + 7) / 8, (hh + 7) / 8);

        // Pass 1: CoC
        { let mut p = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("DoF/CoC"), timestamp_writes: None });
          p.set_pipeline(self.coc_pipeline.as_ref().unwrap());
          p.set_bind_group(0, self.coc_bg.as_ref().unwrap(), &[]);
          p.dispatch_workgroups(wg_full.0, wg_full.1, 1); }
        // Pass 2a: Dilate H
        { let mut p = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("DoF/DilateH"), timestamp_writes: None });
          p.set_pipeline(self.dilate_h_pipeline.as_ref().unwrap());
          p.set_bind_group(0, self.dilate_h_bg.as_ref().unwrap(), &[]);
          p.dispatch_workgroups(wg_full.0, wg_full.1, 1); }
        // Pass 2b: Dilate V
        { let mut p = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("DoF/DilateV"), timestamp_writes: None });
          p.set_pipeline(self.dilate_v_pipeline.as_ref().unwrap());
          p.set_bind_group(0, self.dilate_v_bg.as_ref().unwrap(), &[]);
          p.dispatch_workgroups(wg_full.0, wg_full.1, 1); }
        // Pass 3: Downsample + near/far separation
        { let mut p = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("DoF/Downsample"), timestamp_writes: None });
          p.set_pipeline(self.downsample_pipeline.as_ref().unwrap());
          p.set_bind_group(0, self.downsample_bg.as_ref().unwrap(), &[]);
          p.dispatch_workgroups(wg_half.0, wg_half.1, 1); }
        // Pass 4: Vogel disk blur
        { let mut p = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("DoF/Blur"), timestamp_writes: None });
          p.set_pipeline(self.blur_pipeline.as_ref().unwrap());
          p.set_bind_group(0, self.blur_bg.as_ref().unwrap(), &[]);
          p.dispatch_workgroups(wg_half.0, wg_half.1, 1); }
        // Pass 5: Composite
        { let mut p = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("DoF/Composite"), timestamp_writes: None });
          p.set_pipeline(self.composite_pipeline.as_ref().unwrap());
          p.set_bind_group(0, self.composite_bg.as_ref().unwrap(), &[]);
          p.dispatch_workgroups(wg_full.0, wg_full.1, 1); }
    }

    fn resize(&mut self, width: u32, height: u32, _gbuffer: &GBuffer) {
        if width != self.width || height != self.height {
            // Textures will be lazily recreated. For now, just mark as needing rebuild.
            // We can't create textures here because we don't have the device.
            // The volume calls initialize() again after resize if needed.
            self.width = 0;
            self.height = 0;
        }
    }

    fn destroy(&mut self) {
        self.initialized = false;
        self.coc_tex = None; self.coc_view = None;
        self.coc_dil_temp_tex = None; self.coc_dil_temp_view = None;
        self.near_half_tex = None; self.near_half_view = None;
        self.far_half_tex = None; self.far_half_view = None;
        self.near_blur_tex = None; self.near_blur_view = None;
        self.far_blur_tex = None; self.far_blur_view = None;
    }
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
}
