#![feature(f16)]

mod physics;

use bevy::{prelude::*, render::RenderPlugin};
use physics::PhysicsPlugin;

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
                        ..default()
                    }),
                    ..default()
                }),
        )
        .add_plugins(PhysicsPlugin::default())
        .run();
}
