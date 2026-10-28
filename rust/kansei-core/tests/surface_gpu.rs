//! The renderer's frames on a real GPU when there is no surface texture to draw into: a
//! minimised window's zero size, and no surface at all. Each frame is skipped, never a panic
//! (on macOS a frame runs inside winit's `drawRect:`, which cannot unwind).
//! Skipped (passes) when no adapter is available.

use kansei_core::cameras::Camera;
use kansei_core::geometries::BoxGeometry;
use kansei_core::materials::Material;
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::postprocessing::PostProcessingVolume;
use kansei_core::renderers::{Renderer, RendererConfig};

fn renderer() -> Option<Renderer> {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).ok()?;
    let mut renderer = Renderer::new(RendererConfig { width: 64, height: 64, sample_count: 1, ..Default::default() });
    pollster::block_on(renderer.initialize_headless(&adapter));
    Some(renderer)
}

#[test]
fn frames_without_a_surface_texture_are_skipped() {
    let Some(mut renderer) = renderer() else { return };
    let mut scene = Scene::new();
    scene.add(SceneNode::Renderable(Renderable::new(BoxGeometry::new(1.0, 1.0, 1.0), Material::basic_lit("Box", [1.0, 0.3, 0.2, 1.0], [0.5, 0.5, 0.5, 0.4]))));
    let mut camera = Camera::new(45.0, 0.1, 100.0, 1.0);
    camera.set_position(0.0, 1.0, 4.0);
    let mut volume = PostProcessingVolume::new(&renderer, vec![]);

    // no surface: each frame is drawn up to the surface and skipped there, the camera's too
    assert!(renderer.acquire_surface_texture().is_none());
    renderer.render(&mut scene, &mut camera);
    renderer.render_with_postprocessing(&mut scene, &mut camera, &mut volume);
    assert_eq!(camera.frame(), 0);

    // minimised (a zero size) and back
    renderer.resize(0, 0);
    renderer.render(&mut scene, &mut camera);
    renderer.render_with_postprocessing(&mut scene, &mut camera, &mut volume);
    renderer.resize(128, 72);
    renderer.render_with_postprocessing(&mut scene, &mut camera, &mut volume);
    assert_eq!(camera.frame(), 0);
    let _ = renderer.device().poll(wgpu::PollType::wait_indefinitely());
}
