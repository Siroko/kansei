//! Instancing: a 10 x 10 x 10 grid of cubes drawn in one call. Each cube's model matrix is an
//! instance attribute (`ComputeBuffer::with_vertex_mat4`) of one `InstancedGeometry`, drawn with
//! the stock `Material::basic_instanced`; the CPU rewrites all the matrices each frame to turn
//! the cubes. Drag to orbit, wheel or pinch to zoom.
//!
//! URL parameters: `n=<cubes per side>` (default 10, at most 40).

use wasm_bindgen::prelude::*;

use kansei_core::buffers::{BufferType, BufferUsage, ComputeBuffer};
use kansei_core::cameras::Camera;
use kansei_core::controls::CameraControls;
use kansei_core::geometries::{BoxGeometry, InstancedGeometry};
use kansei_core::materials::Material;
use kansei_core::math::{Mat4, Vec3, Vec4};
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::renderers::RendererConfig;
use kansei_wasm::{param_or, Canvas};

const SPACING: f32 = 3.0;

/// The cubes' model matrices (column-major, 16 floats each) at `time` seconds.
fn instance_matrices(n: usize, time: f32) -> Vec<f32> {
    let offset = (n as f32 - 1.0) * SPACING * 0.5;
    let mut data = Vec::with_capacity(n * n * n * 16);
    for x in 0..n {
        for y in 0..n {
            for z in 0..n {
                let position = Vec3::new(x as f32 * SPACING - offset, y as f32 * SPACING - offset, z as f32 * SPACING - offset);
                let angle = time * 0.5 + (x + y + z) as f32 * 0.3;
                data.extend_from_slice(&(Mat4::from_translation(position) * Mat4::from_rotation_y(angle)).to_cols_array());
            }
        }
    }
    data
}

#[wasm_bindgen]
pub async fn start(canvas_id: &str) -> Result<(), JsValue> {
    let n = param_or("n", 10usize).clamp(1, 40);
    let count = n * n * n;

    let canvas = Canvas::find(canvas_id)?;
    let mut renderer = canvas.renderer(RendererConfig { clear_color: Vec4::new(0.05, 0.05, 0.08, 1.0), ..Default::default() }).await;

    // one vertex buffer of matrices, read per instance at locations 3-6
    let matrices = ComputeBuffer::from_slice("Instances", BufferType::Storage, BufferUsage::VERTEX, &instance_matrices(n, 0.0)).with_vertex_mat4(3);
    let geometry = InstancedGeometry::new(BoxGeometry::new(1.0, 1.0, 1.0), count as u32, vec![matrices]);
    let mut scene = Scene::new();
    let cubes = scene.add(SceneNode::Renderable(Renderable::new(geometry, Material::basic_instanced("Cubes", [0.4, 0.7, 0.9, 1.0]))));

    let mut camera = Camera::new(45.0, 0.1, 500.0, canvas.aspect());
    let mut controls = CameraControls::from_canvas(canvas.element(), Vec3::ZERO, n as f32 * SPACING * 1.6);
    controls.set_elevation(0.35);
    log::info!("Kansei — Instancing (WASM): {count} cubes");

    kansei_wasm::run(&canvas, move |frame| {
        frame.resize(&mut renderer, &mut camera);
        let matrices = instance_matrices(n, frame.time as f32);
        if let Some(buffer) = scene.get_renderable_mut(cubes).and_then(|r| r.geometry.instance_buffers.first_mut()) {
            buffer.write(&matrices);
        }
        controls.update(&mut camera, frame.dt);
        renderer.render(&mut scene, &mut camera);
    });
    Ok(())
}
