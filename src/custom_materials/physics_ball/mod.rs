use bevy::{
    asset::embedded_asset,
    core_pipeline::{
        deferred::{AlphaMask3dDeferred, Opaque3dDeferred},
        prepass::{
            AlphaMask3dPrepass, DeferredPrepass, DepthPrepass, MotionVectorPrepass, NormalPrepass,
            Opaque3dPrepass, OpaqueNoLightmap3dBinKey,
        },
    },
    ecs::system::{SystemParamItem, lifetimeless::SRes},
    pbr::{
        ExtractedDirectionalLight, ExtractedPointLight, LightEntity, MaterialPipeline,
        MaterialPipelineKey, MeshPipelineKey, OpaqueRendererMethod, PreparedMaterial,
        PrepassPipeline, PrepassPipelinePlugin, RenderCascadesVisibleEntities,
        RenderCubemapVisibleEntities, RenderMaterialInstances, RenderMeshInstanceFlags,
        RenderMeshInstances, RenderVisibleMeshEntities, SetMeshBindGroup, SetPrepassViewBindGroup,
        Shadow, ShadowBinKey, ViewLightEntities, alpha_mode_pipeline_key, extract_mesh_materials,
        queue_material_meshes,
    },
    prelude::*,
    render::{
        Render, RenderApp, RenderSet,
        mesh::{
            MeshVertexBufferLayoutRef, RenderMesh, RenderMeshBufferInfo, allocator::MeshAllocator,
        },
        render_asset::{RenderAssetPlugin, RenderAssets, prepare_assets},
        render_phase::{
            AddRenderCommand, BinnedRenderPhaseType, DrawFunctions, PhaseItem, RenderCommand,
            RenderCommandResult, SetItemPipeline, TrackedRenderPass, ViewBinnedRenderPhases,
        },
        render_resource::{
            AsBindGroup, AsBindGroupError, BindGroup, BindGroupEntries, BindGroupLayout,
            BindGroupLayoutEntries, BindGroupLayoutEntry, PipelineCache, PreparedBindGroup,
            RenderPipelineDescriptor, ShaderRef, ShaderStages, SpecializedMeshPipelineError,
            SpecializedMeshPipelines, UnpreparedBindGroup, binding_types::storage_buffer_read_only,
        },
        renderer::RenderDevice,
        storage::GpuShaderStorageBuffer,
        view::{ExtractedView, NoFrustumCulling, RenderVisibleEntities},
    },
};

use crate::physics::{
    EVE_POSITION_WITH_SCALE_STORAGE_BUFFER_HANDLE, ODD_POSITION_WITH_SCALE_STORAGE_BUFFER_HANDLE,
    PhysicsBindGroup, PhysicsSettings,
};
use bevy_radix_sort::{
    EVE_GLOBAL_KEYS_STORAGE_BUFFER_HANDLE, EVE_GLOBAL_VALS_STORAGE_BUFFER_HANDLE,
    ODD_GLOBAL_KEYS_STORAGE_BUFFER_HANDLE, ODD_GLOBAL_VALS_STORAGE_BUFFER_HANDLE,
};

#[derive(Asset, Reflect, Debug, Clone)]
#[reflect(Default, Debug)]
#[derive(Default)]
pub struct PhysicsBallMaterial {}

impl AsBindGroup for PhysicsBallMaterial {
    type Data = ();
    type Param = ();

    fn label() -> Option<&'static str> {
        Some("PhysicsBallMaterial")
    }

    fn as_bind_group(
        &self,
        layout: &BindGroupLayout,
        render_device: &RenderDevice,
        empty: &mut SystemParamItem<'_, '_, Self::Param>,
    ) -> Result<PreparedBindGroup<Self::Data>, AsBindGroupError> {
        let UnpreparedBindGroup { bindings, data } =
            Self::unprepared_bind_group(self, layout, render_device, empty)?;

        // NOTE: This is a workaround.
        //
        // The bind group used in rendering is already defined in [`PhysicsBallBindGroup`], so we can set it to empty here.
        let bind_group_layout =
            render_device.create_bind_group_layout("empty bind_group_layout", &[]);
        let bind_group =
            render_device.create_bind_group("empty bind_group", &bind_group_layout, &[]);

        Ok(PreparedBindGroup {
            bindings,
            bind_group,
            data,
        })
    }

    fn unprepared_bind_group(
        &self,
        _layout: &BindGroupLayout,
        _render_device: &RenderDevice,
        _empty: &mut SystemParamItem<'_, '_, Self::Param>,
    ) -> Result<UnpreparedBindGroup<Self::Data>, AsBindGroupError> {
        Ok(UnpreparedBindGroup {
            bindings: vec![],
            data: (),
        })
    }

    fn bind_group_layout_entries(_render_device: &RenderDevice) -> Vec<BindGroupLayoutEntry> {
        BindGroupLayoutEntries::sequential(
            ShaderStages::VERTEX_FRAGMENT,
            (
                storage_buffer_read_only::<u32>(false),
                storage_buffer_read_only::<u32>(false),
                storage_buffer_read_only::<u32>(false),
                storage_buffer_read_only::<Vec4>(false),
            ),
        )
        .to_vec()
    }
}

impl Material for PhysicsBallMaterial {
    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Opaque
    }

    fn opaque_render_method(&self) -> OpaqueRendererMethod {
        OpaqueRendererMethod::Deferred
    }

    fn depth_bias(&self) -> f32 {
        0.0
    }

    fn reads_view_transmission_texture(&self) -> bool {
        false
    }

    fn prepass_vertex_shader() -> ShaderRef {
        "embedded://bevy_millions_ball/prepass_physics_ball.wgsl".into()
    }

    fn prepass_fragment_shader() -> ShaderRef {
        "embedded://bevy_millions_ball/prepass_physics_ball.wgsl".into()
    }

    fn deferred_vertex_shader() -> ShaderRef {
        "embedded://bevy_millions_ball/deferred_physics_ball.wgsl".into()
    }

    fn deferred_fragment_shader() -> ShaderRef {
        "embedded://bevy_millions_ball/deferred_physics_ball.wgsl".into()
    }

    fn specialize(
        _pipeline: &MaterialPipeline<Self>,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        _key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        if let Some(label) = &mut descriptor.label {
            *label = format!("physics_ball_{}", *label).into();
        }

        Ok(())
    }
}

pub struct MaterialPhysicsBallPlugin {
    shadow_enable: bool,
    prepass_enable: bool,
}

impl Plugin for MaterialPhysicsBallPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(
            app,
            "custom_materials/physics_ball",
            "prepass_physics_ball.wgsl"
        );
        embedded_asset!(
            app,
            "custom_materials/physics_ball",
            "deferred_physics_ball.wgsl"
        );

        app.init_asset::<PhysicsBallMaterial>()
            .register_type::<MeshMaterial3d<PhysicsBallMaterial>>()
            .register_asset_reflect::<PhysicsBallMaterial>()
            .add_plugins(RenderAssetPlugin::<PreparedMaterial<PhysicsBallMaterial>>::default());

        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .init_resource::<DrawFunctions<Shadow>>()
                .init_resource::<RenderMaterialInstances<PhysicsBallMaterial>>()
                .add_render_command::<Shadow, DrawPrepassPhysicsBall>()
                .add_systems(
                    ExtractSchedule,
                    extract_mesh_materials::<PhysicsBallMaterial>,
                )
                .add_systems(
                    Render,
                    PhysicsBallBindGroup::initialize
                        .in_set(RenderSet::PrepareBindGroups)
                        .run_if(not(resource_exists::<PhysicsBallBindGroup>)),
                )
                .add_systems(
                    Render,
                    PhysicsBallBindGroup::prepare_assets
                        .in_set(RenderSet::PrepareAssets)
                        .run_if(resource_exists::<PhysicsBallBindGroup>),
                );

            if self.shadow_enable {
                render_app.add_systems(
                    Render,
                    queue_shadows
                        .in_set(RenderSet::QueueMeshes)
                        .after(prepare_assets::<PreparedMaterial<PhysicsBallMaterial>>),
                );
            }
        }

        if self.shadow_enable || self.prepass_enable {
            app.add_plugins(PrepassPipelinePlugin::<PhysicsBallMaterial>::default());
        }

        if self.prepass_enable {
            app.add_plugins(PrepassPhysicsBallPlugin);
        }
    }

    fn finish(&self, app: &mut App) {
        spawn_physics_ball_mesh(app.world_mut());

        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app.init_resource::<MaterialPipeline<PhysicsBallMaterial>>();
        }
    }
}

fn queue_shadows(
    shadow_draw_functions: Res<DrawFunctions<Shadow>>,
    prepass_pipeline: Res<PrepassPipeline<PhysicsBallMaterial>>,
    render_meshes: Res<RenderAssets<RenderMesh>>,
    render_mesh_instances: Res<RenderMeshInstances>,
    render_materials: Res<RenderAssets<PreparedMaterial<PhysicsBallMaterial>>>,
    render_material_instances: Res<RenderMaterialInstances<PhysicsBallMaterial>>,
    mut shadow_render_phases: ResMut<ViewBinnedRenderPhases<Shadow>>,
    mut pipelines: ResMut<SpecializedMeshPipelines<PrepassPipeline<PhysicsBallMaterial>>>,
    pipeline_cache: Res<PipelineCache>,
    // render_lightmaps: Res<RenderLightmaps>,
    view_lights: Query<(Entity, &ViewLightEntities)>,
    view_light_entities: Query<&LightEntity>,
    point_light_entities: Query<&RenderCubemapVisibleEntities, With<ExtractedPointLight>>,
    directional_light_entities: Query<
        &RenderCascadesVisibleEntities,
        With<ExtractedDirectionalLight>,
    >,
    spot_light_entities: Query<&RenderVisibleMeshEntities, With<ExtractedPointLight>>,
) {
    for (entity, view_lights) in &view_lights {
        let draw_shadow_mesh = shadow_draw_functions.read().id::<DrawPrepassPhysicsBall>();
        for view_light_entity in view_lights.lights.iter().copied() {
            let Ok(light_entity) = view_light_entities.get(view_light_entity) else {
                continue;
            };
            let Some(shadow_phase) = shadow_render_phases.get_mut(&view_light_entity) else {
                continue;
            };

            let is_directional_light = matches!(light_entity, LightEntity::Directional { .. });
            let visible_entities = match light_entity {
                LightEntity::Directional {
                    light_entity,
                    cascade_index,
                } => directional_light_entities
                    .get(*light_entity)
                    .expect("Failed to get directional light visible entities")
                    .entities
                    .get(&entity)
                    .expect("Failed to get directional light visible entities for view")
                    .get(*cascade_index)
                    .expect("Failed to get directional light visible entities for cascade"),
                LightEntity::Point {
                    light_entity,
                    face_index,
                } => point_light_entities
                    .get(*light_entity)
                    .expect("Failed to get point light visible entities")
                    .get(*face_index),
                LightEntity::Spot { light_entity } => spot_light_entities
                    .get(*light_entity)
                    .expect("Failed to get spot light visible entities"),
            };
            let mut light_key = MeshPipelineKey::DEPTH_PREPASS;
            light_key.set(MeshPipelineKey::DEPTH_CLAMP_ORTHO, is_directional_light);

            // NOTE: Lights with shadow mapping disabled will have no visible entities
            // so no meshes will be queued

            for (entity, main_entity) in visible_entities.iter().copied() {
                let Some(mesh_instance) = render_mesh_instances.render_mesh_queue_data(main_entity)
                else {
                    continue;
                };
                if !mesh_instance
                    .flags
                    .contains(RenderMeshInstanceFlags::SHADOW_CASTER)
                {
                    continue;
                }
                let Some(material_asset_id) = render_material_instances.get(&main_entity) else {
                    continue;
                };
                let Some(material) = render_materials.get(*material_asset_id) else {
                    continue;
                };
                let Some(mesh) = render_meshes.get(mesh_instance.mesh_asset_id) else {
                    continue;
                };

                let mut mesh_key =
                    light_key | MeshPipelineKey::from_bits_retain(mesh.key_bits.bits());

                // // Even though we don't use the lightmap in the shadow map, the
                // // `SetMeshBindGroup` render command will bind the data for it. So
                // // we need to include the appropriate flag in the mesh pipeline key
                // // to ensure that the necessary bind group layout entries are
                // // present.
                // if render_lightmaps.render_lightmaps.contains_key(&main_entity) {
                //     mesh_key |= MeshPipelineKey::LIGHTMAPPED;
                // }

                mesh_key |= match material.properties.alpha_mode {
                    AlphaMode::Mask(_)
                    | AlphaMode::Blend
                    | AlphaMode::Premultiplied
                    | AlphaMode::Add
                    | AlphaMode::AlphaToCoverage => MeshPipelineKey::MAY_DISCARD,
                    _ => MeshPipelineKey::NONE,
                };
                let pipeline_id = pipelines.specialize(
                    &pipeline_cache,
                    &prepass_pipeline,
                    MaterialPipelineKey {
                        mesh_key,
                        bind_group_data: material.key.clone(),
                    },
                    &mesh.layout,
                );

                let pipeline_id = match pipeline_id {
                    Ok(id) => id,
                    Err(err) => {
                        error!("{}", err);
                        continue;
                    }
                };

                mesh_instance
                    .material_bind_group_id
                    .set(material.get_bind_group_id());

                shadow_phase.add(
                    ShadowBinKey {
                        draw_function: draw_shadow_mesh,
                        pipeline: pipeline_id,
                        asset_id: mesh_instance.mesh_asset_id.into(),
                    },
                    (entity, main_entity),
                    BinnedRenderPhaseType::mesh(mesh_instance.should_batch()),
                );
            }
        }
    }
}

fn spawn_physics_ball_mesh(world: &mut World) {
    let mut mesh = tetrahedron_from_sphere(1.0).mesh().build();
    // mesh.remove_attribute(Mesh::ATTRIBUTE_NORMAL);
    mesh.remove_attribute(Mesh::ATTRIBUTE_UV_0);

    let name = Name::new("PhysicsBallRenderer");
    let mesh = Mesh3d(world.resource_mut::<Assets<Mesh>>().add(mesh));
    let material = MeshMaterial3d(
        world
            .resource_mut::<Assets<PhysicsBallMaterial>>()
            .add(PhysicsBallMaterial::default()),
    );

    world.spawn((name, mesh, material, NoFrustumCulling));
}

impl Default for MaterialPhysicsBallPlugin {
    fn default() -> Self {
        Self {
            shadow_enable: true,
            prepass_enable: true,
        }
    }
}

struct PrepassPhysicsBallPlugin;

impl Plugin for PrepassPhysicsBallPlugin {
    fn build(&self, app: &mut App) {
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .add_render_command::<Opaque3dPrepass, DrawPrepassPhysicsBall>()
                .add_render_command::<AlphaMask3dPrepass, DrawPrepassPhysicsBall>()
                .add_render_command::<Opaque3dDeferred, DrawPrepassPhysicsBall>()
                .add_render_command::<AlphaMask3dDeferred, DrawPrepassPhysicsBall>()
                .add_systems(
                    Render,
                    queue_prepass_physics_ball_material_meshes
                        .in_set(RenderSet::QueueMeshes)
                        .after(prepare_assets::<PreparedMaterial<PhysicsBallMaterial>>)
                        // queue_material_meshes only writes to `material_bind_group_id`, which `queue_prepass_material_meshes` doesn't read
                        .ambiguous_with(queue_material_meshes::<StandardMaterial>),
                );
        }
    }
}

#[allow(clippy::type_complexity)]
#[allow(clippy::too_many_arguments)]
pub fn queue_prepass_physics_ball_material_meshes(
    (
        opaque_draw_functions,
        alpha_mask_draw_functions,
        opaque_deferred_draw_functions,
        alpha_mask_deferred_draw_functions,
    ): (
        Res<DrawFunctions<Opaque3dPrepass>>,
        Res<DrawFunctions<AlphaMask3dPrepass>>,
        Res<DrawFunctions<Opaque3dDeferred>>,
        Res<DrawFunctions<AlphaMask3dDeferred>>,
    ),
    prepass_pipeline: Res<PrepassPipeline<PhysicsBallMaterial>>,
    mut pipelines: ResMut<SpecializedMeshPipelines<PrepassPipeline<PhysicsBallMaterial>>>,
    pipeline_cache: Res<PipelineCache>,
    render_meshes: Res<RenderAssets<RenderMesh>>,
    render_mesh_instances: Res<RenderMeshInstances>,
    render_materials: Res<RenderAssets<PreparedMaterial<PhysicsBallMaterial>>>,
    render_material_instances: Res<RenderMaterialInstances<PhysicsBallMaterial>>,
    // render_lightmaps: Res<RenderLightmaps>,
    mut opaque_prepass_render_phases: ResMut<ViewBinnedRenderPhases<Opaque3dPrepass>>,
    mut alpha_mask_prepass_render_phases: ResMut<ViewBinnedRenderPhases<AlphaMask3dPrepass>>,
    mut opaque_deferred_render_phases: ResMut<ViewBinnedRenderPhases<Opaque3dDeferred>>,
    mut alpha_mask_deferred_render_phases: ResMut<ViewBinnedRenderPhases<AlphaMask3dDeferred>>,
    views: Query<
        (
            Entity,
            &RenderVisibleEntities,
            &Msaa,
            Option<&DepthPrepass>,
            Option<&NormalPrepass>,
            Option<&MotionVectorPrepass>,
            Option<&DeferredPrepass>,
        ),
        With<ExtractedView>,
    >,
) {
    let opaque_draw_prepass = opaque_draw_functions
        .read()
        .get_id::<DrawPrepassPhysicsBall>()
        .unwrap();
    let alpha_mask_draw_prepass = alpha_mask_draw_functions
        .read()
        .get_id::<DrawPrepassPhysicsBall>()
        .unwrap();
    let opaque_draw_deferred = opaque_deferred_draw_functions
        .read()
        .get_id::<DrawPrepassPhysicsBall>()
        .unwrap();
    let alpha_mask_draw_deferred = alpha_mask_deferred_draw_functions
        .read()
        .get_id::<DrawPrepassPhysicsBall>()
        .unwrap();
    for (
        view,
        visible_entities,
        msaa,
        depth_prepass,
        normal_prepass,
        motion_vector_prepass,
        deferred_prepass,
    ) in &views
    {
        let (
            mut opaque_phase,
            mut alpha_mask_phase,
            mut opaque_deferred_phase,
            mut alpha_mask_deferred_phase,
        ) = (
            opaque_prepass_render_phases.get_mut(&view),
            alpha_mask_prepass_render_phases.get_mut(&view),
            opaque_deferred_render_phases.get_mut(&view),
            alpha_mask_deferred_render_phases.get_mut(&view),
        );

        // Skip if there's no place to put the mesh.
        if opaque_phase.is_none()
            && alpha_mask_phase.is_none()
            && opaque_deferred_phase.is_none()
            && alpha_mask_deferred_phase.is_none()
        {
            continue;
        }

        let mut view_key = MeshPipelineKey::from_msaa_samples(msaa.samples());
        if depth_prepass.is_some() {
            view_key |= MeshPipelineKey::DEPTH_PREPASS;
        }
        if normal_prepass.is_some() {
            view_key |= MeshPipelineKey::NORMAL_PREPASS;
        }
        if motion_vector_prepass.is_some() {
            view_key |= MeshPipelineKey::MOTION_VECTOR_PREPASS;
        }

        for (render_entity, visible_entity) in visible_entities.iter::<With<Mesh3d>>() {
            let Some(material_asset_id) = render_material_instances.get(visible_entity) else {
                continue;
            };
            let Some(mesh_instance) = render_mesh_instances.render_mesh_queue_data(*visible_entity)
            else {
                continue;
            };
            let Some(material) = render_materials.get(*material_asset_id) else {
                continue;
            };
            let Some(mesh) = render_meshes.get(mesh_instance.mesh_asset_id) else {
                continue;
            };

            let mut mesh_key = view_key | MeshPipelineKey::from_bits_retain(mesh.key_bits.bits());

            let alpha_mode = material.properties.alpha_mode;
            match alpha_mode {
                AlphaMode::Opaque | AlphaMode::AlphaToCoverage | AlphaMode::Mask(_) => {
                    mesh_key |= alpha_mode_pipeline_key(alpha_mode, msaa);
                }
                AlphaMode::Blend
                | AlphaMode::Premultiplied
                | AlphaMode::Add
                | AlphaMode::Multiply => continue,
            }

            if material.properties.reads_view_transmission_texture {
                // No-op: Materials reading from `ViewTransmissionTexture` are not rendered in the `Opaque3d`
                // phase, and are therefore also excluded from the prepass much like alpha-blended materials.
                continue;
            }

            let forward = match material.properties.render_method {
                OpaqueRendererMethod::Forward => true,
                OpaqueRendererMethod::Deferred => false,
                OpaqueRendererMethod::Auto => unreachable!(),
            };

            let deferred = deferred_prepass.is_some() && !forward;

            if deferred {
                mesh_key |= MeshPipelineKey::DEFERRED_PREPASS;
            }

            // Even though we don't use the lightmap in the prepass, the
            // `SetMeshBindGroup` render command will bind the data for it. So
            // we need to include the appropriate flag in the mesh pipeline key
            // to ensure that the necessary bind group layout entries are
            // present.
            // if render_lightmaps
            //     .render_lightmaps
            //     .contains_key(visible_entity)
            // {
            //     mesh_key |= MeshPipelineKey::LIGHTMAPPED;
            // }

            // If the previous frame has skins or morph targets, note that.
            if motion_vector_prepass.is_some() {
                if mesh_instance
                    .flags
                    .contains(RenderMeshInstanceFlags::HAS_PREVIOUS_SKIN)
                {
                    mesh_key |= MeshPipelineKey::HAS_PREVIOUS_SKIN;
                }
                if mesh_instance
                    .flags
                    .contains(RenderMeshInstanceFlags::HAS_PREVIOUS_MORPH)
                {
                    mesh_key |= MeshPipelineKey::HAS_PREVIOUS_MORPH;
                }
            }

            let pipeline_id = pipelines.specialize(
                &pipeline_cache,
                &prepass_pipeline,
                MaterialPipelineKey {
                    mesh_key,
                    bind_group_data: material.key,
                },
                &mesh.layout,
            );
            let pipeline_id = match pipeline_id {
                Ok(id) => id,
                Err(err) => {
                    error!("{}", err);
                    continue;
                }
            };

            mesh_instance
                .material_bind_group_id
                .set(material.get_bind_group_id());
            match mesh_key
                .intersection(MeshPipelineKey::BLEND_RESERVED_BITS | MeshPipelineKey::MAY_DISCARD)
            {
                MeshPipelineKey::BLEND_OPAQUE | MeshPipelineKey::BLEND_ALPHA_TO_COVERAGE => {
                    if deferred {
                        opaque_deferred_phase.as_mut().unwrap().add(
                            OpaqueNoLightmap3dBinKey {
                                draw_function: opaque_draw_deferred,
                                pipeline: pipeline_id,
                                asset_id: mesh_instance.mesh_asset_id.into(),
                                material_bind_group_id: material.get_bind_group_id().0,
                            },
                            (*render_entity, *visible_entity),
                            BinnedRenderPhaseType::mesh(mesh_instance.should_batch()),
                        );
                    } else if let Some(opaque_phase) = opaque_phase.as_mut() {
                        opaque_phase.add(
                            OpaqueNoLightmap3dBinKey {
                                draw_function: opaque_draw_prepass,
                                pipeline: pipeline_id,
                                asset_id: mesh_instance.mesh_asset_id.into(),
                                material_bind_group_id: material.get_bind_group_id().0,
                            },
                            (*render_entity, *visible_entity),
                            BinnedRenderPhaseType::mesh(mesh_instance.should_batch()),
                        );
                    }
                }
                // Alpha mask
                MeshPipelineKey::MAY_DISCARD => {
                    if deferred {
                        let bin_key = OpaqueNoLightmap3dBinKey {
                            pipeline: pipeline_id,
                            draw_function: alpha_mask_draw_deferred,
                            asset_id: mesh_instance.mesh_asset_id.into(),
                            material_bind_group_id: material.get_bind_group_id().0,
                        };
                        alpha_mask_deferred_phase.as_mut().unwrap().add(
                            bin_key,
                            (*render_entity, *visible_entity),
                            BinnedRenderPhaseType::mesh(mesh_instance.should_batch()),
                        );
                    } else if let Some(alpha_mask_phase) = alpha_mask_phase.as_mut() {
                        let bin_key = OpaqueNoLightmap3dBinKey {
                            pipeline: pipeline_id,
                            draw_function: alpha_mask_draw_prepass,
                            asset_id: mesh_instance.mesh_asset_id.into(),
                            material_bind_group_id: material.get_bind_group_id().0,
                        };
                        alpha_mask_phase.add(
                            bin_key,
                            (*render_entity, *visible_entity),
                            BinnedRenderPhaseType::mesh(mesh_instance.should_batch()),
                        );
                    }
                }
                _ => {}
            }
        }
    }
}

#[derive(Resource, Debug, Clone)]
struct PhysicsBallBindGroup {
    eve_bind_group: BindGroup,
    odd_bind_group: BindGroup,
    time: i32,
}

impl PhysicsBallBindGroup {
    fn initialize(
        mut commands: Commands,
        slime_pipeline: Res<PrepassPipeline<PhysicsBallMaterial>>,
        render_device: Res<RenderDevice>,
        sbufs: Res<RenderAssets<GpuShaderStorageBuffer>>,
    ) {
        let bind_group_layout = &slime_pipeline.material_layout;

        let eve_global_keys_buf = sbufs
            .get(EVE_GLOBAL_KEYS_STORAGE_BUFFER_HANDLE.id())
            .unwrap();
        let odd_global_keys_buf = sbufs
            .get(ODD_GLOBAL_KEYS_STORAGE_BUFFER_HANDLE.id())
            .unwrap();

        let eve_global_vals_buf = sbufs
            .get(EVE_GLOBAL_VALS_STORAGE_BUFFER_HANDLE.id())
            .unwrap();
        let odd_global_vals_buf = sbufs
            .get(ODD_GLOBAL_VALS_STORAGE_BUFFER_HANDLE.id())
            .unwrap();

        let eve_position_with_scale_buf = sbufs
            .get(EVE_POSITION_WITH_SCALE_STORAGE_BUFFER_HANDLE.id())
            .unwrap();
        let odd_position_with_scale_buf = sbufs
            .get(ODD_POSITION_WITH_SCALE_STORAGE_BUFFER_HANDLE.id())
            .unwrap();

        let eve_bind_group = render_device.create_bind_group(
            "slime renderer bindgroup eve",
            bind_group_layout,
            &BindGroupEntries::sequential((
                odd_global_keys_buf.buffer.as_entire_binding(),
                eve_global_keys_buf.buffer.as_entire_binding(),
                eve_global_vals_buf.buffer.as_entire_binding(),
                // Why use ODD_POSITION_WITH_SCALE_STORAGE_BUFFER_HANDLE?
                //
                // Under the is_even condition, `ODD_POSITION_WITH_SCALE_STORAGE_BUFFER` stores the position and scale information
                // after sorting and physical calculations of the unordered `EVE_POSITION_WITH_SCALE_STORAGE_BUFFER`.
                // This is actually a performance optimization trick.
                // The most accurate way is to input `EVE_POSITION_WITH_SCALE_STORAGE_BUFFER` + `ODD_GLOBAL_VALS_STORAGE_BUFFER`,
                // and then sort `EVE_POSITION_WITH_SCALE_STORAGE_BUFFER` in the shader based on `ODD_GLOBAL_VALS_STORAGE_BUFFER`
                // (similar to the approach in physics.wgsl).
                //
                // However, such precision is unnecessary in the rendering step, and since the physics system iterates continuously
                // with very short steps, using `ODD_POSITION_WITH_SCALE_STORAGE_BUFFER` in the rendering step is sufficient.
                odd_position_with_scale_buf.buffer.as_entire_binding(),
            )),
        );

        let odd_bind_group = render_device.create_bind_group(
            "slime renderer bindgroup odd",
            bind_group_layout,
            &BindGroupEntries::sequential((
                eve_global_keys_buf.buffer.as_entire_binding(),
                odd_global_keys_buf.buffer.as_entire_binding(),
                odd_global_vals_buf.buffer.as_entire_binding(),
                eve_position_with_scale_buf.buffer.as_entire_binding(),
            )),
        );

        let slime_bind_group = Self {
            eve_bind_group,
            odd_bind_group,
            time: -1,
        };

        commands.insert_resource(slime_bind_group);
    }

    fn prepare_assets(
        mut slime_bind_group: ResMut<Self>,
        physics_bind_group: Res<PhysicsBindGroup>,
    ) {
        slime_bind_group.time = physics_bind_group.time();
    }

    fn is_even(&self) -> bool {
        self.time % 2 == 0
    }

    fn bind_group(&self) -> &BindGroup {
        if self.is_even() {
            &self.eve_bind_group
        } else {
            &self.odd_bind_group
        }
    }
}

struct SetPhysicsBallBindGroup<const I: usize>;

impl<P: PhaseItem, const I: usize> RenderCommand<P> for SetPhysicsBallBindGroup<I> {
    type Param = SRes<PhysicsBallBindGroup>;
    type ViewQuery = ();
    type ItemQuery = ();

    #[inline]
    fn render<'w>(
        _item: &P,
        _view: (),
        _item_query: Option<()>,
        slime_bind_group: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        pass.set_bind_group(I, slime_bind_group.into_inner().bind_group(), &[]);

        RenderCommandResult::Success
    }
}

struct DrawMeshInstanced;

impl<P: PhaseItem> RenderCommand<P> for DrawMeshInstanced {
    type Param = (
        SRes<RenderAssets<RenderMesh>>,
        SRes<RenderMeshInstances>,
        SRes<MeshAllocator>,
        SRes<PhysicsSettings>,
    );
    type ViewQuery = ();
    type ItemQuery = ();

    fn render<'w>(
        item: &P,
        _view: (),
        _entity: Option<()>,
        (meshes, render_mesh_instances, mesh_allocator, physics_settings): SystemParamItem<
            'w,
            '_,
            Self::Param,
        >,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let mesh_allocator = mesh_allocator.into_inner();
        let meshes = meshes.into_inner();

        let Some(mesh_instance) = render_mesh_instances.render_mesh_queue_data(item.main_entity())
        else {
            return RenderCommandResult::Skip;
        };
        let Some(gpu_mesh) = meshes.get(mesh_instance.mesh_asset_id) else {
            return RenderCommandResult::Skip;
        };
        let Some(vertex_buffer_slice) =
            mesh_allocator.mesh_vertex_slice(&mesh_instance.mesh_asset_id)
        else {
            return RenderCommandResult::Skip;
        };

        pass.set_vertex_buffer(0, vertex_buffer_slice.buffer.slice(..));

        let instance_range = 0..physics_settings.max_number_of_agents();
        match &gpu_mesh.buffer_info {
            RenderMeshBufferInfo::Indexed {
                count,
                index_format,
            } => {
                let Some(index_buffer_slice) =
                    mesh_allocator.mesh_index_slice(&mesh_instance.mesh_asset_id)
                else {
                    return RenderCommandResult::Skip;
                };

                pass.set_index_buffer(index_buffer_slice.buffer.slice(..), 0, *index_format);
                pass.draw_indexed(
                    index_buffer_slice.range.start..(index_buffer_slice.range.start + count),
                    vertex_buffer_slice.range.start as i32,
                    instance_range,
                );
            }
            RenderMeshBufferInfo::NonIndexed => {
                pass.draw(vertex_buffer_slice.range, instance_range)
            }
        }
        RenderCommandResult::Success
    }
}

type DrawPrepassPhysicsBall = (
    SetItemPipeline,
    SetPrepassViewBindGroup<0>,
    SetMeshBindGroup<1>,
    SetPhysicsBallBindGroup<2>,
    DrawMeshInstanced,
);

fn tetrahedron_from_sphere(radius: f32) -> Tetrahedron {
    let ratio = radius * 6f32.sqrt();

    Tetrahedron {
        vertices: [
            Vec3::new(0., -1. / 6f32.sqrt(), 2. / 3f32.sqrt()) * ratio,
            Vec3::new(1., -1. / 6f32.sqrt(), -1. / 3f32.sqrt()) * ratio,
            Vec3::new(-1., -1. / 6f32.sqrt(), -1. / 3f32.sqrt()) * ratio,
            Vec3::new(0., 3f32.sqrt() / 2f32.sqrt(), 0.) * ratio,
        ],
    }
}
