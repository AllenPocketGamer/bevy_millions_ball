use bevy::{
    asset::load_internal_asset,
    prelude::*,
    render::{
        Render, RenderApp, RenderSet,
        render_asset::RenderAssets,
        render_resource::{
            BindGroup, BindGroupEntries, BindGroupLayout, BindGroupLayoutEntries,
            CachedComputePipelineId, CachedPipelineState, CommandEncoder, ComputePassDescriptor,
            ComputePipelineDescriptor, PipelineCache, PushConstantRange, ShaderDefVal,
            ShaderStages,
            binding_types::{storage_buffer, storage_buffer_read_only},
        },
        renderer::RenderDevice,
        storage::GpuShaderStorageBuffer,
    },
};

use super::{LoadState, dispatch_workgroup_ext};

use bevy_radix_sort::{
    EVE_GLOBAL_KEYS_STORAGE_BUFFER_HANDLE, EVE_GLOBAL_VALS_STORAGE_BUFFER_HANDLE,
    ODD_GLOBAL_KEYS_STORAGE_BUFFER_HANDLE, ODD_GLOBAL_VALS_STORAGE_BUFFER_HANDLE,
};

pub const SPATIAL_HASHING_SHADER_HANDLE: Handle<Shader> =
    Handle::weak_from_u128(125521088680071949847982117826760245819);

pub const NUMBER_OF_THREADS_PER_WORKGROUP: u32 = 256;

pub struct SpatialHashingPlugin;

impl Plugin for SpatialHashingPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            SPATIAL_HASHING_SHADER_HANDLE,
            "spatial_hashing.wgsl",
            Shader::from_wgsl
        );

        app.sub_app_mut(RenderApp).add_systems(
            Render,
            SpatialHashingBindGroup::initialize
                .in_set(RenderSet::PrepareBindGroups)
                .run_if(not(resource_exists::<SpatialHashingBindGroup>)),
        );
    }

    fn finish(&self, app: &mut App) {
        app.sub_app_mut(RenderApp)
            .init_resource::<SpatialHashingPipeline>();
    }
}

#[derive(Resource, Debug, Clone)]
pub struct SpatialHashingPipeline {
    zeroing_pipeline: CachedComputePipelineId,
    hashing_pipeline: CachedComputePipelineId,
    bind_group_layout: BindGroupLayout,
}

impl SpatialHashingPipeline {
    pub fn new(render_device: &RenderDevice, pipeline_cache: &PipelineCache) -> Self {
        let bind_group_layout = render_device.create_bind_group_layout(
            "spatial hashing bind group layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    // A buffer stores the `collision grid` index
                    storage_buffer_read_only::<u32>(false),
                    // A buffer writes the start index of each `collision grid`
                    storage_buffer::<u32>(false),
                    // A buffer writes the end index of each `collision grid`
                    storage_buffer::<u32>(false),
                ),
            ),
        );

        let cdefs = vec![ShaderDefVal::UInt(
            "NUMBER_OF_THREADS_PER_WORKGROUP".into(),
            NUMBER_OF_THREADS_PER_WORKGROUP,
        )];

        let zeroing_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("spatial hashing: zeroing pipeline".into()),
            layout: vec![bind_group_layout.clone()],
            push_constant_ranges: vec![PUSH_CONSTANT_RANGES],
            shader: SPATIAL_HASHING_SHADER_HANDLE,
            shader_defs: [cdefs.as_slice(), &["ZEROING_PIPELINE".into()]].concat(),
            entry_point: "main".into(),
            zero_initialize_workgroup_memory: false,
        });

        let hashing_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("spatial hashing: hashing pipeline".into()),
            layout: vec![bind_group_layout.clone()],
            push_constant_ranges: vec![PUSH_CONSTANT_RANGES],
            shader: SPATIAL_HASHING_SHADER_HANDLE,
            shader_defs: [cdefs.as_slice(), &["HASHING_PIPELINE".into()]].concat(),
            entry_point: "main".into(),
            zero_initialize_workgroup_memory: false,
        });

        Self {
            zeroing_pipeline,
            hashing_pipeline,
            bind_group_layout,
        }
    }

    pub fn zeroing_pipeline(&self) -> CachedComputePipelineId {
        self.zeroing_pipeline
    }

    pub fn hashing_pipeline(&self) -> CachedComputePipelineId {
        self.hashing_pipeline
    }

    pub fn bind_group_layout(&self) -> &BindGroupLayout {
        &self.bind_group_layout
    }
}

impl FromWorld for SpatialHashingPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let pipeline_cache = world.resource::<PipelineCache>();

        Self::new(render_device, pipeline_cache)
    }
}

#[derive(Resource, Debug, Clone)]
pub struct SpatialHashingBindGroup {
    eve_bind_group: BindGroup,
    odd_bind_group: BindGroup,
}

impl SpatialHashingBindGroup {
    pub fn initialize(
        mut commands: Commands,
        spatial_hashing_pipeline: Res<SpatialHashingPipeline>,
        render_device: Res<RenderDevice>,
        sbufs: Res<RenderAssets<GpuShaderStorageBuffer>>,
    ) {
        let bind_group_layout = spatial_hashing_pipeline.bind_group_layout();

        let eve_global_keys_buf = sbufs
            .get(EVE_GLOBAL_KEYS_STORAGE_BUFFER_HANDLE.id())
            .unwrap();
        let eve_global_vals_buf = sbufs
            .get(EVE_GLOBAL_VALS_STORAGE_BUFFER_HANDLE.id())
            .unwrap();
        let odd_global_keys_buf = sbufs
            .get(ODD_GLOBAL_KEYS_STORAGE_BUFFER_HANDLE.id())
            .unwrap();
        let odd_global_vals_buf = sbufs
            .get(ODD_GLOBAL_VALS_STORAGE_BUFFER_HANDLE.id())
            .unwrap();

        let eve_bind_group = render_device.create_bind_group(
            "spatial hashing bindgroup eve",
            bind_group_layout,
            &BindGroupEntries::sequential((
                eve_global_keys_buf.buffer.as_entire_binding(),
                odd_global_keys_buf.buffer.as_entire_binding(),
                odd_global_vals_buf.buffer.as_entire_binding(),
            )),
        );

        let odd_bind_group = render_device.create_bind_group(
            "spatial hashing bindgroup odd",
            bind_group_layout,
            &BindGroupEntries::sequential((
                odd_global_keys_buf.buffer.as_entire_binding(),
                eve_global_keys_buf.buffer.as_entire_binding(),
                eve_global_vals_buf.buffer.as_entire_binding(),
            )),
        );

        let spatial_hashing_bind_group = Self {
            eve_bind_group,
            odd_bind_group,
        };

        commands.insert_resource(spatial_hashing_bind_group);
    }
}

pub fn run(
    encoder: &mut CommandEncoder,
    pipeline_cache: &PipelineCache,
    spatial_hashing_pipeline: &SpatialHashingPipeline,
    spatial_hashing_bind_group: &SpatialHashingBindGroup,
    max_compute_workgroups_per_dimension: u32,
    number_of_keys: u32,
    // Read from `eve_global_keys_buf`?
    read_from_even: bool,
) {
    let zeroing_pipeline = pipeline_cache
        .get_compute_pipeline(spatial_hashing_pipeline.zeroing_pipeline())
        .unwrap();

    let hashing_pipeline = pipeline_cache
        .get_compute_pipeline(spatial_hashing_pipeline.hashing_pipeline())
        .unwrap();

    let number_of_workgroups = number_of_keys.div_ceil(NUMBER_OF_THREADS_PER_WORKGROUP);

    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("spatial_hashing compute pass"),
            ..default()
        });

        pass.set_pipeline(zeroing_pipeline);

        if read_from_even {
            pass.set_bind_group(0, &spatial_hashing_bind_group.eve_bind_group, &[]);
        } else {
            pass.set_bind_group(0, &spatial_hashing_bind_group.odd_bind_group, &[]);
        }

        pass.set_push_constants(NUMBER_OF_KEYS_OFFSET, bytemuck::bytes_of(&number_of_keys));

        dispatch_workgroup_ext(
            &mut pass,
            number_of_workgroups,
            max_compute_workgroups_per_dimension,
            WORKGROUP_OFFSET_OFFSET,
        );

        pass.set_pipeline(hashing_pipeline);

        dispatch_workgroup_ext(
            &mut pass,
            number_of_workgroups,
            max_compute_workgroups_per_dimension,
            WORKGROUP_OFFSET_OFFSET,
        );
    }
}

const WORKGROUP_OFFSET_OFFSET: u32 = 0;
const NUMBER_OF_KEYS_OFFSET: u32 = 4;

const PUSH_CONSTANT_RANGES: PushConstantRange = PushConstantRange {
    stages: ShaderStages::COMPUTE,
    range: 0..8,
};

pub(super) fn check_load_state(world: &World) -> LoadState {
    let pipeline_cache = world.resource::<PipelineCache>();
    let spatial_hashing_pipeline = world.resource::<SpatialHashingPipeline>();

    let (zeroing_pipeline_state, hashing_pipeline_state) = (
        pipeline_cache.get_compute_pipeline_state(spatial_hashing_pipeline.zeroing_pipeline()),
        pipeline_cache.get_compute_pipeline_state(spatial_hashing_pipeline.hashing_pipeline()),
    );

    if let CachedPipelineState::Err(err) = zeroing_pipeline_state {
        return LoadState::Failed(format!("Failed to load spatial_zeroing_pipeline: {}", err));
    }

    if let CachedPipelineState::Err(err) = hashing_pipeline_state {
        return LoadState::Failed(format!("Failed to load spatial_hashing_pipeline: {}", err));
    }

    if matches!(zeroing_pipeline_state, CachedPipelineState::Ok(_))
        && matches!(hashing_pipeline_state, CachedPipelineState::Ok(_))
    {
        return LoadState::Loaded;
    }

    LoadState::OnLoad
}

#[cfg(test)]
mod tests {
    use bevy::{
        render::{
            Render, RenderPlugin, RenderSet,
            render_resource::{
                Buffer, BufferAddress, BufferDescriptor, BufferInitDescriptor, BufferUsages,
                CommandEncoderDescriptor, Maintain, MapMode,
            },
            renderer::RenderQueue,
        },
        scene::ScenePlugin,
    };

    use bevy_radix_sort::{
        GetSubgroupSizePlugin, RadixSortBindGroup, RadixSortPipeline, RadixSortPlugin,
        RadixSortSettings,
    };

    use super::*;

    const MAX_NUMBER_OF_KEYS: u32 = 16 * 1024 * 1024;
    const NUMBER_OF_BYTES_PER_KEY: u32 = 4;

    #[derive(Resource, Debug, Clone)]
    pub struct UnitTestHelper {
        /// The staging buffer used to store the input keys.
        ///
        /// cpu-buffer -> gpu-staging-buffer -> gpu-destination-buffer
        pub ikeys_staging_buf: Buffer,
        /// The staging buffer used to store the input vals.
        ///
        /// cpu-buffer -> gpu-staging-buffer -> gpu-destination-buffer
        pub ivals_staging_buf: Buffer,
        /// The staging buffer used to store the output collision_grid_start_indexs.
        ///
        /// gpu-source-buffer -> gpu-staging-buffer -> cpu-buffer
        pub o_collision_grid_start_indexs_staging_buf: Buffer,
        /// The staging buffer used to store the output collision_grid_close_indexs.
        ///
        /// gpu-source-buffer -> gpu-staging-buffer -> cpu-buffer
        pub o_collision_grid_close_indexs_staging_buf: Buffer,
    }

    impl UnitTestHelper {
        pub fn new(number_of_keys: u32, render_device: &RenderDevice) -> Self {
            let keys: Vec<u32> = (0..number_of_keys).map(|v| v % 256).collect();
            let vals: Vec<u32> = (0..number_of_keys).collect();

            let ikeys_staging_buf = render_device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("radix_sort: input keys staging buffer"),
                usage: BufferUsages::COPY_SRC | BufferUsages::MAP_WRITE,
                contents: bytemuck::cast_slice(keys.as_slice()),
            });

            let ivals_staging_buf = render_device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("radix_sort: input vals staging buffer"),
                usage: BufferUsages::COPY_SRC | BufferUsages::MAP_WRITE,
                contents: bytemuck::cast_slice(vals.as_slice()),
            });

            let okeys_staging_buf = render_device.create_buffer(&BufferDescriptor {
                label: Some("radix_sort: output keys staging buffer"),
                size: (number_of_keys * NUMBER_OF_BYTES_PER_KEY) as BufferAddress,
                usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

            let ovals_staging_buf = render_device.create_buffer(&BufferDescriptor {
                label: Some("radix_sort: output vals staging buffer"),
                size: (number_of_keys * NUMBER_OF_BYTES_PER_KEY) as BufferAddress,
                usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

            Self {
                ikeys_staging_buf,
                ivals_staging_buf,
                o_collision_grid_start_indexs_staging_buf: okeys_staging_buf,
                o_collision_grid_close_indexs_staging_buf: ovals_staging_buf,
            }
        }
    }

    fn prepare_unit_test_helper(
        mut commands: Commands,
        render_device: Res<RenderDevice>,
        radix_sort_settings: Res<RadixSortSettings>,
    ) {
        let unit_test_helper =
            UnitTestHelper::new(radix_sort_settings.max_number_of_keys(), &render_device);
        commands.insert_resource(unit_test_helper);
    }

    fn run_once(app: &mut App) {
        app.finish();
        app.cleanup();

        app.update();
    }

    fn create_unit_test_app() -> App {
        let mut app = App::new();

        app.add_plugins(MinimalPlugins)
            .add_plugins(WindowPlugin::default())
            .add_plugins(AssetPlugin::default())
            .add_plugins(ScenePlugin)
            .add_plugins(RenderPlugin {
                synchronous_pipeline_compilation: true,
                ..default()
            })
            .add_plugins(ImagePlugin::default())
            .add_plugins(GetSubgroupSizePlugin)
            .add_plugins(RadixSortPlugin {
                settings: MAX_NUMBER_OF_KEYS.into(),
            })
            .add_plugins(SpatialHashingPlugin);

        app.sub_app_mut(RenderApp).add_systems(
            Render,
            prepare_unit_test_helper
                .in_set(RenderSet::PrepareResources)
                .run_if(not(resource_exists::<UnitTestHelper>)),
        );

        app
    }

    fn run_spatial_hashing_test(number_of_keys: u32, pass_count: u32, read_from_even: bool) {
        let mut app = create_unit_test_app();

        let unit_test_system =
            move |render_device: Res<RenderDevice>,
                  render_queue: Res<RenderQueue>,
                  pipeline_cache: Res<PipelineCache>,
                  radix_sort_pipeline: Res<RadixSortPipeline>,
                  radix_bind_group: Res<RadixSortBindGroup>,
                  spatial_hashing_pipeline: Res<SpatialHashingPipeline>,
                  spatial_hashing_bind_group: Res<SpatialHashingBindGroup>,
                  unit_test_helper: Res<UnitTestHelper>,
                  sbufs: Res<RenderAssets<GpuShaderStorageBuffer>>| {
                let eve_global_keys_buf = sbufs
                    .get(EVE_GLOBAL_KEYS_STORAGE_BUFFER_HANDLE.id())
                    .unwrap();
                let eve_global_vals_buf = sbufs
                    .get(EVE_GLOBAL_VALS_STORAGE_BUFFER_HANDLE.id())
                    .unwrap();
                let odd_global_keys_buf = sbufs
                    .get(ODD_GLOBAL_KEYS_STORAGE_BUFFER_HANDLE.id())
                    .unwrap();
                let odd_global_vals_buf = sbufs
                    .get(ODD_GLOBAL_VALS_STORAGE_BUFFER_HANDLE.id())
                    .unwrap();

                let max_compute_workgroups_per_dimension =
                    render_device.limits().max_compute_workgroups_per_dimension;

                let copy_size = (number_of_keys * NUMBER_OF_BYTES_PER_KEY) as BufferAddress;

                let mut encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
                    label: Some("unit_test: radix_sort command encoder"),
                });

                let global_keys_buf = if read_from_even {
                    &eve_global_keys_buf.buffer
                } else {
                    &odd_global_keys_buf.buffer
                };
                encoder.copy_buffer_to_buffer(
                    &unit_test_helper.ikeys_staging_buf,
                    0,
                    global_keys_buf,
                    0,
                    copy_size,
                );

                let global_vals_buf = if read_from_even {
                    &eve_global_vals_buf.buffer
                } else {
                    &odd_global_vals_buf.buffer
                };
                encoder.copy_buffer_to_buffer(
                    &unit_test_helper.ivals_staging_buf,
                    0,
                    global_vals_buf,
                    0,
                    copy_size,
                );

                bevy_radix_sort::run(
                    &mut encoder,
                    &pipeline_cache,
                    &radix_sort_pipeline,
                    &radix_bind_group,
                    max_compute_workgroups_per_dimension,
                    number_of_keys,
                    0..pass_count,
                    false,
                    read_from_even,
                );

                let result_in_even = (pass_count + !read_from_even as u32) % 2 == 0;

                run(
                    &mut encoder,
                    &pipeline_cache,
                    &spatial_hashing_pipeline,
                    &spatial_hashing_bind_group,
                    max_compute_workgroups_per_dimension,
                    number_of_keys,
                    result_in_even,
                );

                let collision_grid_count = number_of_keys.min(256);
                let result_size = (collision_grid_count * NUMBER_OF_BYTES_PER_KEY) as BufferAddress;

                let collision_grid_start_indexs_buf = if result_in_even {
                    &odd_global_keys_buf.buffer
                } else {
                    &eve_global_keys_buf.buffer
                };
                encoder.copy_buffer_to_buffer(
                    collision_grid_start_indexs_buf,
                    0,
                    &unit_test_helper.o_collision_grid_start_indexs_staging_buf,
                    0,
                    result_size,
                );

                let collision_grid_close_indexs_buf = if result_in_even {
                    &odd_global_vals_buf.buffer
                } else {
                    &eve_global_vals_buf.buffer
                };
                encoder.copy_buffer_to_buffer(
                    collision_grid_close_indexs_buf,
                    0,
                    &unit_test_helper.o_collision_grid_close_indexs_staging_buf,
                    0,
                    result_size,
                );

                render_queue.submit([encoder.finish()]);

                let keys_slice = unit_test_helper
                    .o_collision_grid_start_indexs_staging_buf
                    .slice(0..result_size);
                let vals_slice = unit_test_helper
                    .o_collision_grid_close_indexs_staging_buf
                    .slice(0..result_size);

                keys_slice.map_async(MapMode::Read, |_| ());
                vals_slice.map_async(MapMode::Read, |_| ());

                render_device.poll(Maintain::Wait).panic_on_timeout();

                // assert! collision_grid_start_indexs
                {
                    let view = keys_slice.get_mapped_range();
                    let data: &[u32] = bytemuck::cast_slice(&view);

                    // println!("");
                    // let nums = number_of_keys.div_ceil(NUMBER_OF_THREADS_PER_WORKGROUP);
                    // for i in 0..nums as usize {
                    //     let open = i * NUMBER_OF_THREADS_PER_WORKGROUP as usize;
                    //     let stop = (open + NUMBER_OF_THREADS_PER_WORKGROUP as usize)
                    //         .min(number_of_keys as usize);
                    //     let block = &data[open..stop];
                    //     println!("block {}: {:?}", i, block);
                    // }

                    let answer: Vec<u32> = (0..collision_grid_count)
                        // convert to collision_grid_agent_count
                        .map(|collision_grid_index| {
                            number_of_keys / 256
                                + (number_of_keys % 256 > collision_grid_index) as u32
                        })
                        // prefix sum
                        .scan(0u32, |acc, collision_grid_agent_count| {
                            let exclusive = *acc;
                            *acc += collision_grid_agent_count;
                            Some(exclusive)
                        })
                        .collect();
                    assert_eq!(data, &answer);
                }

                // assert! collision_grid_close_indexs
                {
                    let view = vals_slice.get_mapped_range();
                    let data: &[u32] = bytemuck::cast_slice(&view);

                    let answer: Vec<u32> = (0..collision_grid_count)
                        // convert to collision_grid_agent_count
                        .map(|collision_grid_index| {
                            number_of_keys / 256
                                + (number_of_keys % 256 > collision_grid_index) as u32
                        })
                        // prefix sum
                        .scan(0u32, |acc, collision_grid_agent_count| {
                            let exclusive = *acc;
                            *acc += collision_grid_agent_count;
                            Some(exclusive + collision_grid_agent_count - 1)
                        })
                        .collect();
                    assert_eq!(data, &answer);
                }

                unit_test_helper
                    .o_collision_grid_start_indexs_staging_buf
                    .unmap();
                unit_test_helper
                    .o_collision_grid_close_indexs_staging_buf
                    .unmap();
            };

        app.sub_app_mut(RenderApp)
            .add_systems(Render, unit_test_system.in_set(RenderSet::Cleanup));

        run_once(&mut app);
    }

    #[test]
    fn test_sh_1() {
        run_spatial_hashing_test(1, 1, true);
    }

    #[test]
    fn test_sh_100() {
        run_spatial_hashing_test(100, 4, true);
        run_spatial_hashing_test(100, 4, false);

        run_spatial_hashing_test(100, 3, true);
        run_spatial_hashing_test(100, 3, false);
    }

    #[test]
    fn test_sh_256() {
        run_spatial_hashing_test(256, 4, true);
        run_spatial_hashing_test(256, 4, false);

        run_spatial_hashing_test(256, 3, true);
        run_spatial_hashing_test(256, 3, false);
    }

    #[test]
    fn test_sh_16_777_216() {
        run_spatial_hashing_test(16_777_216, 4, true);
        run_spatial_hashing_test(16_777_216, 4, false);

        run_spatial_hashing_test(16_777_216, 3, true);
        run_spatial_hashing_test(16_777_216, 3, false);
    }
}
