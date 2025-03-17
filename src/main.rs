#![feature(f16)]

mod custom_materials;
mod physics;

use std::f32::consts::PI;

use bevy::{
    core_pipeline::prepass::{DeferredPrepass, DepthPrepass},
    pbr::OpaqueRendererMethod,
    prelude::*,
    render::RenderPlugin,
    window::WindowResolution,
};
use bevy_egui::{EguiContexts, EguiPlugin, egui};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use custom_materials::{
    CustomMaterialPlugin, ReferenceGridExtension, ReferenceGridMaterial, Thickness,
};
use physics::{PhysicsPlugin, PhysicsSettings};

fn main() {
    App::new()
        .add_plugins(
            DefaultPlugins
                .set(RenderPlugin {
                    synchronous_pipeline_compilation: true,
                    ..default()
                })
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        name: Some("Bevy Millions Ball".into()),
                        resolution: WindowResolution::new(1920., 1080.),
                        ..default()
                    }),
                    ..default()
                }),
        )
        .add_plugins(EguiPlugin)
        .add_plugins(PanOrbitCameraPlugin)
        .add_plugins(CustomMaterialPlugin)
        .add_plugins(PhysicsPlugin {
            max_number_of_agents: 256 * 256,
            number_of_grids_one_dimension: 64,
            half_map_height: 8,
            e_agent: 0.8,
            e_envir: 0.8,
            ..default()
        })
        .add_systems(Startup, setup)
        .add_systems(Update, physics_ui)
        .add_systems(Update, freeze_orbit_yzero)
        .run();
}

#[derive(Component, Clone, Copy, Debug)]
pub struct BackgroundMarker;

fn setup(
    mut commands: Commands,
    mut rfg_mats: ResMut<Assets<ReferenceGridMaterial>>,
    asset_server: Res<AssetServer>,
    physics_settings: Res<PhysicsSettings>,
) {
    // 1. spawn a camera
    commands.spawn((
        Camera3d::default(),
        Camera {
            clear_color: ClearColorConfig::Custom(Color::srgb(
                248.0 / 255.0,
                151.0 / 255.0,
                184.0 / 255.0,
            )),
            ..default()
        },
        DepthPrepass,
        DeferredPrepass,
        Transform::from_xyz(0., 64., 64.).looking_at(Vec3::ZERO, Vec3::Y),
        PanOrbitCamera::default(),
    ));

    // 2. light
    commands.spawn((
        DirectionalLight {
            shadows_enabled: true,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::ZYX, 0., PI * -0.15, PI * -0.15)),
    ));

    let bg_mesh_handle = asset_server.load("models/inverse_cube.gltf#Mesh0/Primitive0");
    let bg_matr_handle = rfg_mats.add(ReferenceGridMaterial {
        base: StandardMaterial {
            base_color: Color::srgb(0.8, 0.7, 0.6),
            opaque_render_method: OpaqueRendererMethod::Deferred,
            ..default()
        },
        extension: ReferenceGridExtension {
            grid_size: Vec2::splat(physics_settings.grid_size as f32),
            grid_thickness: Thickness::screen_space(2.0, 2.0),
            grid_color: Color::srgb(75.0 / 255.0, 55.0 / 255.0, 55.0 / 255.0),
            ..default()
        },
    });

    commands.spawn((
        Name::new("Background"),
        Mesh3d(bg_mesh_handle),
        MeshMaterial3d(bg_matr_handle),
        Transform::from_scale(Vec3::new(
            physics_settings.map_size(),
            physics_settings.map_height(),
            physics_settings.map_size(),
        )),
        BackgroundMarker,
    ));
}

fn physics_ui(
    mut bgt: Query<
        (&mut Transform, &MeshMaterial3d<ReferenceGridMaterial>),
        With<BackgroundMarker>,
    >,
    mut light: Query<&mut DirectionalLight>,
    mut rfg_mats: ResMut<Assets<ReferenceGridMaterial>>,
    mut ctx: EguiContexts,
    mut physics_settings: ResMut<PhysicsSettings>,
) {
    let (mut bg_transform, bg_matr_handle) = bgt.single_mut();

    bg_transform.scale = Vec3::new(
        physics_settings.map_size(),
        physics_settings.map_height(),
        physics_settings.map_size(),
    );

    if let Some(bg_matr) = rfg_mats.get_mut(bg_matr_handle) {
        bg_matr.extension.grid_size = Vec2::splat(physics_settings.grid_size as f32);
    }

    // ui
    let max_grid_nums = physics_settings.max_number_of_grids_on_dimension();
    egui::Window::new("Physics Settings").show(ctx.ctx_mut(), |ui| {
        // Display information about controls and current simulation
        ui.heading("Controls & Info");
        ui.label("• Left mouse button: Rotate camera");
        ui.label("• Right mouse button: Pan camera");
        ui.label("• Scroll wheel: Zoom in/out");
        ui.label(format!(
            "• Current simulation: {} spheres",
            physics_settings.max_number_of_agents()
        ));
        ui.separator();

        ui.vertical(|ui| {
            ui.add(egui::Checkbox::new(
                &mut physics_settings.enable_simulation,
                "enable_simulation",
            ));
            ui.add(egui::Checkbox::new(
                &mut physics_settings.enable_spring,
                "enable_spring",
            ));
            ui.add(
                egui::Slider::new(&mut physics_settings.time_delta, (1. / 120.)..=(1. / 15.))
                    .text("time_delta"),
            );

            ui.label("Gravity");
            ui.horizontal(|ui| {
                ui.label("x");
                ui.add(egui::DragValue::new(&mut physics_settings.gravity.x).speed(0.1));
                ui.label("y");
                ui.add(egui::DragValue::new(&mut physics_settings.gravity.y).speed(0.1));
                ui.label("z");
                ui.add(egui::DragValue::new(&mut physics_settings.gravity.z).speed(0.1));
            });
            ui.separator();

            ui.label("Restitution");
            ui.add(egui::Slider::new(&mut physics_settings.e_envir, 0.0..=1.5).text("e_envir"));
            ui.add(egui::Slider::new(&mut physics_settings.e_agent, 0.0..=1.5).text("e_agent"));
            ui.separator();

            ui.label("Spring");
            ui.add(egui::Slider::new(&mut physics_settings.k_agent, 0.0..=400.).text("k_agent"));
            ui.add(egui::Slider::new(&mut physics_settings.c_agent, 0.0..=4.).text("c_agent"));
            ui.add(egui::Slider::new(&mut physics_settings.r_agent, 0.0..=1.).text("r_agent"));
            ui.separator();

            ui.label("Environment");
            ui.add(
                egui::Slider::new(&mut physics_settings.half_map_height, 1..=128)
                    .text("half_map_height"),
            );
            ui.add(egui::Slider::new(&mut physics_settings.grid_size, 1..=32).text("grid_size"));
            ui.add(
                egui::Slider::new(
                    &mut physics_settings.number_of_grids_one_dimension,
                    2..=max_grid_nums,
                )
                .text("grid_nums_one_dimension"),
            );
            ui.separator();

            ui.add(egui::Checkbox::new(
                &mut light.single_mut().shadows_enabled,
                "shadows_enabled (disable shadows to improve performance)",
            ));
        });
    });
}

fn freeze_orbit_yzero(mut orbit: Query<&mut PanOrbitCamera>) {
    let mut orbit = orbit.single_mut();

    orbit.target_focus.y = 0.;
    orbit.force_update = true;
}
