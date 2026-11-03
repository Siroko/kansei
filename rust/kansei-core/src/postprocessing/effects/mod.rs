mod atmosphere;
mod bloom;
mod clouds;
mod cinematic_dof;
mod color_grading;
mod contact_ao;
mod distance_fog;
mod dof;
mod environment_background;
mod fluid_surface;
mod height_fog;
mod motion_blur;
mod screen_space_glass;
mod smaa;
mod ssao;
mod ssgi;
mod taa;
mod tonemap;
mod volumetric_fog;
pub use atmosphere::AtmosphereEffect;
pub use bloom::{BloomEffect, BloomOptions};
pub use clouds::{CloudLayer, CloudQuality, VolumetricCloudsEffect, VolumetricCloudsOptions};
pub use cinematic_dof::{CameraLens, CinematicDepthOfFieldEffect, CinematicDepthOfFieldOptions, DofDebugView, HighlightOptions};
pub use color_grading::{ColorGradingEffect, ColorGradingOptions};
pub use contact_ao::{ContactAoEffect, ContactAoOptions};
pub use distance_fog::{DistanceFogColor, DistanceFogEffect};
pub use dof::{DepthOfFieldEffect, DepthOfFieldOptions};
pub use environment_background::EnvironmentBackgroundEffect;
pub use fluid_surface::{FluidGlassSurface, FluidMask, FluidSurfaceEffect, FluidSurfaceOptions};
pub use height_fog::{HeightFogEffect, HeightFogLayer};
pub use motion_blur::{MotionBlurEffect, MotionBlurOptions};
pub use screen_space_glass::{ScreenSpaceGlassEffect, ScreenSpaceGlassOptions};
pub use smaa::{SmaaEffect, SmaaOptions};
pub use ssao::{SSAOAmbient, SSAOEffect, SSAOOptions};
pub use ssgi::{GiQuality, ScreenSpaceGIEffect, ScreenSpaceGIOptions};
pub use taa::{TemporalAAEffect, TemporalAAOptions};
pub use tonemap::{
    ev100_from_camera, exposure_from_ev100, exposure_from_ev100_lens, unreal_white_balance_matrix, white_balance_matrix, ColorGrade, FilmGrain,
    ToneMapEffect, LocalExposure, ToneMapOptions, ToneMapper, UnrealFilm, LENS_ATTENUATION_UE4, LENS_ATTENUATION_UE5,
};
pub use volumetric_fog::{LocalFogShape, LocalFogVolume, SpotScattering, VolumetricFogEffect, VolumetricFogOptions};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::postprocessing::PostProcessingEffect;

    /// The display effects and the fogs never bind the GBuffer's velocity, so a chain of them
    /// skips the renderer's velocity pass; the temporal effects read it.
    #[test]
    fn the_display_effects_and_fogs_do_not_read_velocity() {
        let skip: Vec<Box<dyn PostProcessingEffect>> = vec![
            Box::new(SmaaEffect::new(Default::default())),
            Box::new(ToneMapEffect::new(Default::default())),
            Box::new(ColorGradingEffect::new(Default::default())),
            Box::new(BloomEffect::new(Default::default())),
            Box::new(HeightFogEffect::new(HeightFogLayer::default())),
            Box::new(DistanceFogEffect::new(10.0, 30.0, DistanceFogColor::Constant([1.0; 3]))),
            Box::new(VolumetricFogEffect::new(Default::default())),
        ];
        for effect in &skip {
            assert!(!effect.reads_velocity(), "{} reads velocity", effect.name());
        }
        let read: Vec<Box<dyn PostProcessingEffect>> = vec![Box::new(TemporalAAEffect::new(Default::default())), Box::new(MotionBlurEffect::new(Default::default()))];
        for effect in &read {
            assert!(effect.reads_velocity(), "{} doesn't read velocity", effect.name());
        }
    }
}
