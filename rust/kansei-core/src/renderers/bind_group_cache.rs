//! `BindGroupCache`: bind groups kept by what they bind, for the passes that make one from the
//! frame's textures every frame (ping-pong targets, histories swapped each frame), where the same
//! few come round again: the cached one is returned instead of a new one. An effect owns one
//! (wgpu compares its objects by their ids, which another `wgpu::Instance` hands out again: a
//! cache shared by every device in a thread could hand one device another's bind group).

use std::cell::{Cell, RefCell};
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
const SWEEP_EVERY: u64 = 256;
const UNUSED_LOOKUPS: u64 = 512;

/// An owner's bind groups by what they bind (`get`): the owner's passes make the same few from
/// the frame's resources each frame, which then cost a hash lookup instead of a bind group a
/// frame (a WebGPU call, and garbage on the web). A view of a texture made anew is another view,
/// so a stale bind group is never returned; unused ones are dropped after a while. One device's:
/// the owner's (effects are made for one).
#[derive(Default)]
pub struct BindGroupCache {
    groups: RefCell<HashMap<Key, (wgpu::BindGroup, u64)>>,
    clock: Cell<u64>,
    made: Cell<u64>,
}

impl BindGroupCache {
    /// `device.create_bind_group(descriptor)`, or the bind group made before for the same layout
    /// and the same buffers (and ranges), views, samplers and acceleration structures.
    pub fn get(&self, device: &wgpu::Device, descriptor: &wgpu::BindGroupDescriptor) -> wgpu::BindGroup {
        let Some(key) = Key::of(descriptor) else { return device.create_bind_group(descriptor) };
        let now = self.clock.get() + 1;
        self.clock.set(now);
        let mut groups = self.groups.borrow_mut();
        if now % SWEEP_EVERY == 0 {
            groups.retain(|_, (_, used)| now - *used <= UNUSED_LOOKUPS);
        }
        if let Some((group, used)) = groups.get_mut(&key) {
            *used = now;
            return group.clone();
        }
        let group = device.create_bind_group(descriptor);
        self.made.set(self.made.get() + 1);
        groups.insert(key, (group.clone(), now));
        group
    }

    /// Bind groups it has made, and holds now (for tests and stats).
    pub fn stats(&self) -> (u64, usize) {
        (self.made.get(), self.groups.borrow().len())
    }

    /// Drop them all (a device lost, the owner's resources made anew).
    pub fn clear(&self) {
        self.groups.borrow_mut().clear();
    }
}
