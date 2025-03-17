pub mod reference_grid;
pub use reference_grid::*;

use bevy::{asset::load_internal_asset, prelude::*};
use bevy_wgsl_utils::WgslUtilsPlugin;

pub const SDF_SHADER_HANDLE: Handle<Shader> =
    Handle::weak_from_u128(340282366920938463463374607431768211455);
pub const MATH_SHADER_HANDLE: Handle<Shader> =
    Handle::weak_from_u128(28431294871239487123948712394871239487);
pub const PROC_TEXTURE_SHADER_HANDLE: Handle<Shader> =
    Handle::weak_from_u128(123456789012345678901234567890123456789);

pub struct CustomMaterialPlugin;

impl Plugin for CustomMaterialPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(app, SDF_SHADER_HANDLE, "sdf.wgsl", Shader::from_wgsl);
        load_internal_asset!(app, MATH_SHADER_HANDLE, "math.wgsl", Shader::from_wgsl);
        load_internal_asset!(
            app,
            PROC_TEXTURE_SHADER_HANDLE,
            "proc_texture.wgsl",
            Shader::from_wgsl
        );

        app.add_plugins(MaterialReferenceGridPlugin);

        if !app.is_plugin_added::<WgslUtilsPlugin>() {
            app.add_plugins(WgslUtilsPlugin);
        }
    }
}
