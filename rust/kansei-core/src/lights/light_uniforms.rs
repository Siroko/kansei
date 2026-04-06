use super::{Light, DirectionalLight, PointLight};

pub const MAX_DIRECTIONAL_LIGHTS: usize = 4;
pub const MAX_POINT_LIGHTS: usize = 8;

const DIR_LIGHT_FLOATS: usize = 8;
const POINT_LIGHT_FLOATS: usize = 8;
const HEADER_FLOATS: usize = 4;

pub const LIGHT_UNIFORM_FLOATS: usize = HEADER_FLOATS
    + MAX_DIRECTIONAL_LIGHTS * DIR_LIGHT_FLOATS
    + MAX_POINT_LIGHTS * POINT_LIGHT_FLOATS;
pub const LIGHT_UNIFORM_BYTES: usize = LIGHT_UNIFORM_FLOATS * 4;

/// Packs scene lights into a GPU-ready uniform buffer.
pub struct LightUniforms {
    pub data: Vec<f32>,
}

impl LightUniforms {
    pub fn new() -> Self {
        Self { data: vec![0.0; LIGHT_UNIFORM_FLOATS] }
    }

    /// Pack lights from the scene into the uniform buffer.
    pub fn pack(&mut self, lights: &[Light]) {
        self.data.fill(0.0);

        let mut num_dir: u32 = 0;
        let mut num_point: u32 = 0;

        let dir_offset = HEADER_FLOATS;
        let point_offset = HEADER_FLOATS + MAX_DIRECTIONAL_LIGHTS * DIR_LIGHT_FLOATS;

        for light in lights {
            match light {
                Light::Directional(dl) => {
                    if (num_dir as usize) < MAX_DIRECTIONAL_LIGHTS {
                        let i = num_dir as usize;
                        let o = dir_offset + i * DIR_LIGHT_FLOATS;
                        let ec = dl.effective_color();
                        self.data[o] = dl.direction.x;
                        self.data[o + 1] = dl.direction.y;
                        self.data[o + 2] = dl.direction.z;
                        self.data[o + 4] = ec.x;
                        self.data[o + 5] = ec.y;
                        self.data[o + 6] = ec.z;
                        self.data[o + 7] = dl.intensity;
                        num_dir += 1;
                    }
                }
                Light::Point(pl) => {
                    if (num_point as usize) < MAX_POINT_LIGHTS {
                        let i = num_point as usize;
                        let o = point_offset + i * POINT_LIGHT_FLOATS;
                        let ec = pl.effective_color();
                        self.data[o] = pl.position.x;
                        self.data[o + 1] = pl.position.y;
                        self.data[o + 2] = pl.position.z;
                        self.data[o + 3] = pl.radius;
                        self.data[o + 4] = ec.x;
                        self.data[o + 5] = ec.y;
                        self.data[o + 6] = ec.z;
                        self.data[o + 7] = pl.intensity;
                        num_point += 1;
                    }
                }
                Light::Area(_) => {}
            }
        }

        self.data[0] = f32::from_bits(num_dir);
        self.data[1] = f32::from_bits(num_point);
    }

    /// Pack lights from references into the uniform buffer.
    pub fn pack_refs(&mut self, lights: &[&Light]) {
        self.data.fill(0.0);

        let mut num_dir: u32 = 0;
        let mut num_point: u32 = 0;

        let dir_offset = HEADER_FLOATS;
        let point_offset = HEADER_FLOATS + MAX_DIRECTIONAL_LIGHTS * DIR_LIGHT_FLOATS;

        for light in lights {
            match light {
                Light::Directional(dl) => {
                    if (num_dir as usize) < MAX_DIRECTIONAL_LIGHTS {
                        let i = num_dir as usize;
                        let o = dir_offset + i * DIR_LIGHT_FLOATS;
                        let ec = dl.effective_color();
                        self.data[o] = dl.direction.x;
                        self.data[o + 1] = dl.direction.y;
                        self.data[o + 2] = dl.direction.z;
                        self.data[o + 4] = ec.x;
                        self.data[o + 5] = ec.y;
                        self.data[o + 6] = ec.z;
                        self.data[o + 7] = dl.intensity;
                        num_dir += 1;
                    }
                }
                Light::Point(pl) => {
                    if (num_point as usize) < MAX_POINT_LIGHTS {
                        let i = num_point as usize;
                        let o = point_offset + i * POINT_LIGHT_FLOATS;
                        let ec = pl.effective_color();
                        self.data[o] = pl.position.x;
                        self.data[o + 1] = pl.position.y;
                        self.data[o + 2] = pl.position.z;
                        self.data[o + 3] = pl.radius;
                        self.data[o + 4] = ec.x;
                        self.data[o + 5] = ec.y;
                        self.data[o + 6] = ec.z;
                        self.data[o + 7] = pl.intensity;
                        num_point += 1;
                    }
                }
                Light::Area(_) => {}
            }
        }

        self.data[0] = f32::from_bits(num_dir);
        self.data[1] = f32::from_bits(num_point);
    }

    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.data)
    }
}

impl Default for LightUniforms {
    fn default() -> Self { Self::new() }
}
