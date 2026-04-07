use bytemuck::{Pod, Zeroable};

/// Path tracer material properties. 64 bytes (16 floats) matching TS GPU layout.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct PathTracerMaterial {
    pub albedo: [f32; 3],
    pub roughness: f32,
    pub metallic: f32,
    pub ior: f32,
    pub max_bounces: f32,       // negative = invisible to probes
    pub transmission: f32,      // 0=opaque, 1=glass
    pub absorption_color: [f32; 3],
    pub absorption_density: f32,
    pub emissive: [f32; 3],
    pub emissive_intensity: f32,
}

impl Default for PathTracerMaterial {
    fn default() -> Self {
        Self {
            albedo: [0.8, 0.8, 0.8],
            roughness: 0.5,
            metallic: 0.0,
            ior: 1.5,
            max_bounces: 4.0,
            transmission: 0.0,
            absorption_color: [1.0, 1.0, 1.0],
            absorption_density: 0.0,
            emissive: [0.0, 0.0, 0.0],
            emissive_intensity: 0.0,
        }
    }
}

impl PathTracerMaterial {
    pub const GPU_STRIDE: usize = 64;

    pub fn emissive(color: [f32; 3], intensity: f32) -> Self {
        Self {
            emissive: color,
            emissive_intensity: intensity,
            ..Default::default()
        }
    }

    pub fn glass(ior: f32) -> Self {
        Self {
            transmission: 1.0,
            ior,
            roughness: 0.0,
            ..Default::default()
        }
    }

    pub fn metal(albedo: [f32; 3], roughness: f32) -> Self {
        Self {
            albedo,
            metallic: 1.0,
            roughness,
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem;

    #[test]
    fn size_is_64_bytes() {
        assert_eq!(mem::size_of::<PathTracerMaterial>(), PathTracerMaterial::GPU_STRIDE);
    }
}
