use bevy::{
    asset::embedded_asset,
    pbr::{ExtendedMaterial, MaterialExtension, MaterialExtensionKey, MaterialExtensionPipeline},
    prelude::*,
    render::{
        mesh::MeshVertexBufferLayoutRef,
        render_asset::RenderAssets,
        render_resource::{
            AsBindGroup, AsBindGroupShaderType, RenderPipelineDescriptor, ShaderRef, ShaderType,
            SpecializedMeshPipelineError,
        },
        texture::GpuImage,
    },
};

pub const REFERENCE_GRID_SHADER_PATH: &str = "embedded://bevy_millions_ball/reference_grid.wgsl";

pub type ReferenceGridMaterial = ExtendedMaterial<StandardMaterial, ReferenceGridExtension>;

pub struct MaterialReferenceGridPlugin;

impl Plugin for MaterialReferenceGridPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(
            app,
            "custom_materials/reference_grid",
            "reference_grid.wgsl"
        );

        app.register_asset_reflect::<ReferenceGridMaterial>()
            .add_plugins(MaterialPlugin::<ReferenceGridMaterial>::default());
    }
}

#[derive(Reflect, Debug, Clone, Copy, PartialEq)]
#[reflect(Default, Debug)]
pub enum Thickness {
    WorldSpace(Vec2),
    ScreenSpace(Vec2),
}

impl Thickness {
    pub fn world_space(x: f32, y: f32) -> Self {
        Thickness::WorldSpace(Vec2::new(x, y))
    }

    #[allow(dead_code)]
    pub fn screen_space(x: f32, y: f32) -> Self {
        Thickness::ScreenSpace(Vec2::new(x, y))
    }

    #[allow(dead_code)]
    pub fn is_world_space(&self) -> bool {
        matches!(self, Thickness::WorldSpace(_))
    }

    pub fn is_screen_space(&self) -> bool {
        matches!(self, Thickness::ScreenSpace(_))
    }
}

impl Default for Thickness {
    fn default() -> Self {
        Thickness::WorldSpace(Vec2::new(1.0, 1.0))
    }
}

impl Into<Vec2> for Thickness {
    fn into(self) -> Vec2 {
        match self {
            Thickness::WorldSpace(vec) => vec,
            Thickness::ScreenSpace(vec) => vec,
        }
    }
}

#[derive(Asset, AsBindGroup, Reflect, Debug, Clone)]
#[bind_group_data(ReferenceGridKey)]
#[uniform(13, ReferenceGridUniform)]
pub struct ReferenceGridExtension {
    /// The side length of the reference grid in world space.
    pub grid_size: Vec2,
    /// The width of the reference grid lines in world space.
    pub grid_thickness: Thickness,
    /// The color of the reference grid.
    pub grid_color: Color,
    /// Controls the strength of anti-aliasing for the ReferenceGrid, a procedural texture.
    /// Measured in pixels, `smoothness` adjusts the intensity of a low-pass filter applied to the grid.
    ///
    /// Anti-aliasing is necessary to prevent visual artifacts such as flickering or jagged edges,
    /// which occur when the texture is sampled at a frequency lower than its inherent detail (e.g.,
    /// when viewed from a distance or at a grazing angle). The low-pass filter smooths out these
    /// artifacts by suppressing high-frequency signals, and `smoothness` determines how aggressively
    /// this filtering is applied. Higher values result in smoother edges, while lower values preserve
    /// sharper details at the risk of increased aliasing.
    pub smoothness: f32,
    /// Whether to enable the checkerboard pattern.
    pub enable_checkerboard: bool,
    /// The side length of the checkerboard grid in world space.
    pub checkerboard_grid_size: Vec2,
    /// The display strength of checkerboard grid.
    pub checkerboard_strength: Vec2,
    /// Whether to enable UV distortion effects.
    /// When enabled, the UV coordinates of the grid will be distorted, creating a warped or wavy effect.
    pub enable_uv_distortion: bool,
    /// The scale of the UV distortion effect.
    /// This controls the frequency of the distortion, with larger values creating more frequent distortions.
    pub uv_distortion_scale: Vec2,
    /// The strength of the UV distortion effect.
    /// This controls the intensity of the distortion, with higher values creating more pronounced warping.
    pub uv_distortion_strength: f32,
}

impl Default for ReferenceGridExtension {
    fn default() -> Self {
        ReferenceGridExtension {
            grid_size: Vec2::new(8.0, 8.0),
            grid_thickness: Thickness::world_space(0.1, 0.1),
            grid_color: Color::BLACK,
            smoothness: 1.0,
            enable_checkerboard: true,
            checkerboard_grid_size: Vec2::new(1.0, 1.0),
            checkerboard_strength: Vec2::new(0.0, 0.1),
            enable_uv_distortion: true,
            uv_distortion_scale: Vec2::splat(1.0),
            uv_distortion_strength: 0.04,
        }
    }
}

impl MaterialExtension for ReferenceGridExtension {
    fn fragment_shader() -> ShaderRef {
        REFERENCE_GRID_SHADER_PATH.into()
    }

    fn deferred_fragment_shader() -> ShaderRef {
        REFERENCE_GRID_SHADER_PATH.into()
    }

    fn specialize(
        _pipeline: &MaterialExtensionPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        key: MaterialExtensionKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        if let Some(fragment) = descriptor.fragment.as_mut() {
            let shader_defs = &mut fragment.shader_defs;

            if key.bind_group_data.enable_checkerboard {
                shader_defs.push("ENABLE_CHECKERBOARD".into());
            }

            if key.bind_group_data.enable_screen_space_thickness {
                shader_defs.push("ENABLE_SCREEN_SPACE_THICKNESS".into());
            }

            if key.bind_group_data.enable_uv_distortion {
                shader_defs.push("ENABLE_UV_DISTORTION".into());
            }
        }

        Ok(())
    }
}

#[derive(Clone, ShaderType)]
struct ReferenceGridUniform {
    grid_size: Vec2,
    grid_thickness: Vec2,
    grid_color: Vec4,
    smoothness: f32,
    checkerboard_grid_size: Vec2,
    checkerboard_grid_strength: Vec2,
    uv_distortion_scale: Vec2,
    uv_distortion_strength: f32,
}

impl AsBindGroupShaderType<ReferenceGridUniform> for ReferenceGridExtension {
    fn as_bind_group_shader_type(&self, _images: &RenderAssets<GpuImage>) -> ReferenceGridUniform {
        ReferenceGridUniform {
            grid_size: self.grid_size,
            grid_thickness: self.grid_thickness.into(),
            grid_color: LinearRgba::from(self.grid_color).to_vec4(),
            smoothness: self.smoothness,
            checkerboard_grid_size: self.checkerboard_grid_size,
            checkerboard_grid_strength: self.checkerboard_strength,
            uv_distortion_scale: self.uv_distortion_scale,
            uv_distortion_strength: self.uv_distortion_strength,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ReferenceGridKey {
    pub enable_checkerboard: bool,
    pub enable_screen_space_thickness: bool,
    pub enable_uv_distortion: bool,
}

impl From<&ReferenceGridExtension> for ReferenceGridKey {
    fn from(extension: &ReferenceGridExtension) -> Self {
        ReferenceGridKey {
            enable_checkerboard: extension.enable_checkerboard,
            enable_screen_space_thickness: extension.grid_thickness.is_screen_space(),
            enable_uv_distortion: extension.enable_uv_distortion,
        }
    }
}
