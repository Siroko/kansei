//! `cached_bind_group`: bind groups kept by what they bind, for the passes that make one from the
//! frame's textures every frame (ping-pong targets, histories swapped each frame), where the same
//! few come round again: the cached one is returned instead of a new one.

use std::cell::RefCell;
use std::collections::HashMap;

/// One resource of a bind group, as the cache compares it (by identity).
#[derive(Clone, PartialEq, Eq, Hash)]
enum Resource {
    Buffer(wgpu::Buffer, u64, Option<u64>),
    Buffers(Vec<(wgpu::Buffer, u64, Option<u64>)>),
    Sampler(wgpu::Sampler),
    Samplers(Vec<wgpu::Sampler>),
    View(wgpu::TextureView),
    Views(Vec<wgpu::TextureView>),
    Tlas(wgpu::Tlas),
}

/// What a bind group binds: its layout, and each entry's binding and resource.
#[derive(Clone, PartialEq, Eq, Hash)]
struct Key {
    layout: wgpu::BindGroupLayout,
    entries: Vec<(u32, Resource)>,
}

impl Key {
    /// None when an entry binds something the cache doesn't compare.
    fn of(descriptor: &wgpu::BindGroupDescriptor) -> Option<Self> {
        let buffer = |b: &wgpu::BufferBinding| (b.buffer.clone(), b.offset, b.size.map(u64::from));
        let entries = descriptor
            .entries
            .iter()
            .map(|e| {
                let r = match &e.resource {
                    wgpu::BindingResource::Buffer(b) => {
                        let (buffer, offset, size) = buffer(b);
                        Resource::Buffer(buffer, offset, size)
                    }
                    wgpu::BindingResource::BufferArray(bs) => Resource::Buffers(bs.iter().map(buffer).collect()),
                    wgpu::BindingResource::Sampler(s) => Resource::Sampler((*s).clone()),
                    wgpu::BindingResource::SamplerArray(ss) => Resource::Samplers(ss.iter().map(|s| (*s).clone()).collect()),
                    wgpu::BindingResource::TextureView(v) => Resource::View((*v).clone()),
                    wgpu::BindingResource::TextureViewArray(vs) => Resource::Views(vs.iter().map(|v| (*v).clone()).collect()),
                    wgpu::BindingResource::AccelerationStructure(t) => Resource::Tlas((*t).clone()),
                    _ => return None,
                };
                Some((e.binding, r))
            })
            .collect::<Option<Vec<_>>>()?;
        Some(Self { layout: descriptor.layout.clone(), entries })
    }
}

/// Lookups between sweeps, and how many lookups a bind group may go unused before a sweep drops
/// it (with its resources: a resized target's old views aren't kept alive for long).
const SWEEP_EVERY: u64 = 1024;
const UNUSED_LOOKUPS: u64 = 2048;

#[derive(Default)]
struct Cache {
    groups: HashMap<Key, (wgpu::BindGroup, u64)>,
    clock: u64,
    made: u64,
}

thread_local! {
    static CACHE: RefCell<Cache> = RefCell::new(Cache::default());
}

/// `device.create_bind_group(descriptor)`, or the bind group made before for the same layout and
/// the same buffers (and ranges), views, samplers and acceleration structures. For bind groups
/// made from the frame's resources each frame: the same few then cost a hash lookup instead of a
/// bind group a frame (a WebGPU call, and garbage on the web). A view of a texture made anew is
/// another view, so a stale bind group is never returned; unused ones are dropped after a while.
pub fn cached_bind_group(device: &wgpu::Device, descriptor: &wgpu::BindGroupDescriptor) -> wgpu::BindGroup {
    let Some(key) = Key::of(descriptor) else { return device.create_bind_group(descriptor) };
    CACHE.with(|cache| {
        let mut c = cache.borrow_mut();
        c.clock += 1;
        let now = c.clock;
        if now % SWEEP_EVERY == 0 {
            c.groups.retain(|_, (_, used)| now - *used <= UNUSED_LOOKUPS);
        }
        if let Some((group, used)) = c.groups.get_mut(&key) {
            *used = now;
            return group.clone();
        }
        let group = device.create_bind_group(descriptor);
        c.made += 1;
        c.groups.insert(key, (group.clone(), now));
        group
    })
}

/// Bind groups `cached_bind_group` has made on this thread, and holds now (for tests and stats).
pub fn cached_bind_group_stats() -> (u64, usize) {
    CACHE.with(|c| {
        let c = c.borrow();
        (c.made, c.groups.len())
    })
}
