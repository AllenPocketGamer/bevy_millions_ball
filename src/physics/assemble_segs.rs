use bevy::{
    asset::load_internal_asset,
    prelude::*,
    render::{
        render_resource::{
            binding_types::{storage_buffer, storage_buffer_read_only},
            BindGroup, BindGroupEntries, BindGroupLayout, BindGroupLayoutEntries, Buffer,
            BufferDescriptor, BufferUsages, CachedComputePipelineId, CachedPipelineState,
            CommandEncoder, ComputePassDescriptor, ComputePipelineDescriptor, PipelineCache,
            PushConstantRange, ShaderDefVal, ShaderStages,
        },
        renderer::RenderDevice,
        RenderApp,
    },
};

use super::{
    dispatch_workgroup_ext, get_subgroup_size::SubgroupSize, radix_sort::RadixSortBindGroup,
    LoadState, WORKGROUP_OFFSET_OFFSET,
};

pub const ASSEMBLE_SEGS_SHADER_HANDLE: Handle<Shader> =
    Handle::weak_from_u128(125521088680071949847982117826760245819);

/// The row size of the `keys` processed by each workgroup.
pub const NUMBER_OF_ROWS_PER_WORKGROUP: u32 = 16;
/// The number of threads per workgroup.
pub const NUMBER_OF_THREADS_PER_WORKGROUP: u32 = 256;

pub struct AssembleSegsPlugin;

impl Plugin for AssembleSegsPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            ASSEMBLE_SEGS_SHADER_HANDLE,
            "assemble_segs.wgsl",
            Shader::from_wgsl
        );
    }

    fn finish(&self, app: &mut App) {
        app.sub_app_mut(RenderApp)
            .init_resource::<AssembleSegsPipeline>()
            .init_resource::<AssembleSegsBindGroup>();
    }
}

/// It will segment a sequential unsigned integer (u32) array (`global_keys`),
/// storing consecutive segments (`segment`) with the same key in order into the `global_segs` buffer, while storing
/// the mapping from the key to the `global_segs` in the `key_seg_map` buffer.
///
/// ## For example
///
/// ```
///                 0   1   2   3   4   5   6   7                                                                                                                                                                 
///               ┌───┬───┬───┬───┬───┬───┬───┬───┐                                                                                                                                                               
///  global_keys: │ 1 │ 1 │ 1 │ 1 │ 4 │ 4 │ 8 │ 8 │                                                                                                                                                               
///               └─▲─┴───┴───┴───┴─▲─┴───┴─▲─┴───┘                                                                                                                                                               
///                 │   ┌───────────┘       │                                                                                                                                                                     
///                 │   │   ┌───────────────┘                                                                                                                                                                     
///               ┌─┼─┬─┼─┬─┴─┬───┬───┬───┬───┬───┐                                                                                                                                                               
///  number_segs: │ 0 │ 4 │ 6 │ x │ x │ x │ x │ x │                                                                                                                                                               
///               └─┬─┴─┬─┴─┬─┴───┴───┴───┴───┴───┘                                                                                                                                                               
///                 │   └─┐ └─────────┐                                                                                                                                                                           
///                 └───┐ └─────────┐ └─────────────┐                                                                                                                                                             
///               ┌───┬─▼─┬───┬───┬─▼─┬───┬───┬───┬─▼─┬───┬───┬───┐                                                                                                                                               
///  key_seg_map: │ x │ 0 │ x │ x │ 1 │ x │ x │ x │ 2 │ x │ x │ x │                                                                                                                                               
///               └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘                                                                                                                                               
///                 0  *1*  2   3  *4*  5   6   7  *8*  9  10   11          
/// ```                                                                                                                                      
#[derive(Resource, Debug, Clone)]
pub struct AssembleSegsPipeline {
    count_segs_pipeline: CachedComputePipelineId,
    // scan(1/3): calculate the sum of each workgroup in hireachy.
    scan_sums_pipeline: CachedComputePipelineId,
    // scan(2/3): calculate the prefix sum of the last workgroup.
    scan_last_pipeline: CachedComputePipelineId,
    // scan(3/3): calculate the prefix sum of each workgroup in hireachy.
    scan_prfx_pipeline: CachedComputePipelineId,
    assemble_pipeline: CachedComputePipelineId,
    bind_group_layout: BindGroupLayout,
}

impl AssembleSegsPipeline {
    pub fn new(
        render_device: &RenderDevice,
        pipeline_cache: &PipelineCache,
        subgroup_size: u32,
    ) -> Self {
        let bind_group_layout = render_device.create_bind_group_layout(
            "assemble segments bind group layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    // `global_keys`: The original keys to be stored in the buffer.
                    storage_buffer_read_only::<u32>(false),
                    // `number_segs`: A tempoary buffer stores the number of segments to be scanned in each workgroup.
                    storage_buffer::<u32>(false),
                    // `global_segs`: The segments of key is stored in the buffer.
                    storage_buffer::<u32>(false),
                    // `key_seg_map`: A buffer stores the mapping from the key to the segment index in `global_segs`.
                    storage_buffer::<u32>(false),
                    // `tnumber_seg`: A value store the total number of segments.
                    storage_buffer::<u32>(false),
                ),
            ),
        );

        let cdefs = vec![
            ShaderDefVal::UInt(
                "NUMBER_OF_THREADS_PER_WORKGROUP".into(),
                NUMBER_OF_THREADS_PER_WORKGROUP,
            ),
            ShaderDefVal::UInt("NUMBER_OF_THREADS_PER_SUBGROUP".into(), subgroup_size),
            ShaderDefVal::UInt(
                "NUMBER_OF_ROWS_PER_WORKGROUP".into(),
                NUMBER_OF_ROWS_PER_WORKGROUP,
            ),
        ];

        let count_segs_pipeline =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("assemble_segs: count_segs pipeline".into()),
                layout: vec![bind_group_layout.clone()],
                push_constant_ranges: vec![PUSH_CONSTANT_RANGES],
                shader: ASSEMBLE_SEGS_SHADER_HANDLE,
                shader_defs: [cdefs.as_slice(), &["COUNT_SEGS_PIPELINE".into()]].concat(),
                entry_point: "main".into(),
            });

        let scan_sums_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("assemble_segs: scan_sums pipeline".into()),
            layout: vec![bind_group_layout.clone()],
            push_constant_ranges: vec![PUSH_CONSTANT_RANGES],
            shader: ASSEMBLE_SEGS_SHADER_HANDLE,
            shader_defs: [cdefs.as_slice(), &["SCAN_SUMS_PIPELINE".into()]].concat(),
            entry_point: "main".into(),
        });

        let scan_last_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("assemble_segs: scan_last pipeline".into()),
            layout: vec![bind_group_layout.clone()],
            push_constant_ranges: vec![PUSH_CONSTANT_RANGES],
            shader: ASSEMBLE_SEGS_SHADER_HANDLE,
            shader_defs: [cdefs.as_slice(), &["SCAN_LAST_PIPELINE".into()]].concat(),
            entry_point: "main".into(),
        });

        let scan_prfx_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("assemble_segs: scan_prfx pipeline".into()),
            layout: vec![bind_group_layout.clone()],
            push_constant_ranges: vec![PUSH_CONSTANT_RANGES],
            shader: ASSEMBLE_SEGS_SHADER_HANDLE,
            shader_defs: [cdefs.as_slice(), &["SCAN_PRFX_PIPELINE".into()]].concat(),
            entry_point: "main".into(),
        });

        let assemble_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("assemble_segs: assemble pipeline".into()),
            layout: vec![bind_group_layout.clone()],
            push_constant_ranges: vec![PUSH_CONSTANT_RANGES],
            shader: ASSEMBLE_SEGS_SHADER_HANDLE,
            shader_defs: [cdefs.as_slice(), &["ASSEMBLE_PIPELINE".into()]].concat(),
            entry_point: "main".into(),
        });

        Self {
            count_segs_pipeline,
            scan_sums_pipeline,
            scan_last_pipeline,
            scan_prfx_pipeline,
            assemble_pipeline,
            bind_group_layout,
        }
    }

    pub fn count_segs_pipeline(&self) -> CachedComputePipelineId {
        self.count_segs_pipeline
    }

    pub fn scan_sums_pipeline(&self) -> CachedComputePipelineId {
        self.scan_sums_pipeline
    }

    pub fn scan_last_pipeline(&self) -> CachedComputePipelineId {
        self.scan_last_pipeline
    }

    pub fn scan_prfx_pipeline(&self) -> CachedComputePipelineId {
        self.scan_prfx_pipeline
    }

    pub fn assemble_pipeline(&self) -> CachedComputePipelineId {
        self.assemble_pipeline
    }

    pub fn bind_group_layout(&self) -> &BindGroupLayout {
        &self.bind_group_layout
    }

    pub fn create_bind_group(
        &self,
        radix_sort_bind_group: &RadixSortBindGroup,
        render_device: &RenderDevice,
    ) -> AssembleSegsBindGroup {
        AssembleSegsBindGroup::new(
            self.bind_group_layout(),
            radix_sort_bind_group,
            render_device,
        )
    }
}

impl FromWorld for AssembleSegsPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let subgroup_size = world.resource::<SubgroupSize>();

        Self::new(render_device, pipeline_cache, subgroup_size.into())
    }
}

#[derive(Resource, Debug, Clone)]
pub struct AssembleSegsBindGroup {
    tnumber_seg_buf: Buffer,

    eve_bind_group: BindGroup,
    odd_bind_group: BindGroup,
}

impl AssembleSegsBindGroup {
    pub fn new(
        bind_group_layout: &BindGroupLayout,
        radix_sort_bind_group: &RadixSortBindGroup,
        render_device: &RenderDevice,
    ) -> Self {
        let tnumber_seg_buf = render_device.create_buffer(&BufferDescriptor {
            label: Some("assemble_segs: a buffer stores the total number of segments"),
            size: 4,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let eve_bind_group = render_device.create_bind_group(
            "assemble segments bind group eve",
            bind_group_layout,
            &BindGroupEntries::sequential((
                radix_sort_bind_group
                    .eve_global_keys_buf()
                    .as_entire_binding(),
                radix_sort_bind_group
                    .global_blocks_buf()
                    .as_entire_binding(),
                radix_sort_bind_group
                    .odd_global_keys_buf()
                    .as_entire_binding(),
                radix_sort_bind_group
                    .odd_global_vals_buf()
                    .as_entire_binding(),
                tnumber_seg_buf.as_entire_binding(),
            )),
        );

        let odd_bind_group = render_device.create_bind_group(
            "assemble segments bindgroup odd",
            bind_group_layout,
            &BindGroupEntries::sequential((
                radix_sort_bind_group
                    .odd_global_keys_buf()
                    .as_entire_binding(),
                radix_sort_bind_group
                    .global_blocks_buf()
                    .as_entire_binding(),
                radix_sort_bind_group
                    .eve_global_keys_buf()
                    .as_entire_binding(),
                radix_sort_bind_group
                    .eve_global_vals_buf()
                    .as_entire_binding(),
                tnumber_seg_buf.as_entire_binding(),
            )),
        );

        Self {
            tnumber_seg_buf,

            eve_bind_group,
            odd_bind_group,
        }
    }

    /// Return the buffer stores the total number of segments.
    pub fn tnumber_seg_buf(&self) -> &Buffer {
        &self.tnumber_seg_buf
    }
}

impl FromWorld for AssembleSegsBindGroup {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let radix_sort_bind_group = world.resource::<RadixSortBindGroup>();
        let assemble_segs_pipeline = world.resource::<AssembleSegsPipeline>();

        assemble_segs_pipeline.create_bind_group(radix_sort_bind_group, render_device)
    }
}

/// ## Corner case
///
/// When `number_of_keys == 0`, `tnumber_seg_buf` will be filled with 0, and `global_segs_buf` and `key_seg_map_buf` will not be filled,
/// accessing their values will be undefined.
pub fn run(
    encoder: &mut CommandEncoder,
    pipeline_cache: &PipelineCache,
    assemble_segs_pipeline: &AssembleSegsPipeline,
    assemble_segs_bind_group: &AssembleSegsBindGroup,
    max_compute_workgroups_per_dimension: u32,
    number_of_keys: u32,
    // Read from `eve_global_keys_buf`?
    read_from_even: bool,
) {
    let count_segs_pipeline = pipeline_cache
        .get_compute_pipeline(assemble_segs_pipeline.count_segs_pipeline())
        .unwrap();

    let scan_sums_pipeline = pipeline_cache
        .get_compute_pipeline(assemble_segs_pipeline.scan_sums_pipeline())
        .unwrap();

    let scan_last_pipeline = pipeline_cache
        .get_compute_pipeline(assemble_segs_pipeline.scan_last_pipeline())
        .unwrap();

    let scan_prfx_pipeline = pipeline_cache
        .get_compute_pipeline(assemble_segs_pipeline.scan_prfx_pipeline())
        .unwrap();

    let assemble_pipeline = pipeline_cache
        .get_compute_pipeline(assemble_segs_pipeline.assemble_pipeline())
        .unwrap();

    let number_of_keys_per_block = NUMBER_OF_THREADS_PER_WORKGROUP * NUMBER_OF_ROWS_PER_WORKGROUP;
    let number_of_blocks = number_of_keys.div_ceil(number_of_keys_per_block);

    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("assemble_segments compute pass"),
            ..default()
        });

        // 1. Count the number of segments in each workgroup.
        pass.set_pipeline(count_segs_pipeline);

        if read_from_even {
            pass.set_bind_group(0, &assemble_segs_bind_group.eve_bind_group, &[]);
        } else {
            pass.set_bind_group(0, &assemble_segs_bind_group.odd_bind_group, &[]);
        }

        pass.set_push_constants(NUMBER_OF_KEYS_OFFSET, bytemuck::bytes_of(&number_of_keys));

        dispatch_workgroup_ext(
            &mut pass,
            number_of_blocks,
            max_compute_workgroups_per_dimension,
            WORKGROUP_OFFSET_OFFSET,
        );

        // 2-0. Scan: prepare the scan argument for each dispatch.
        let mut load_base = 0u32;
        let mut save_base = number_of_blocks;
        let mut rounds = vec![];
        while save_base - load_base > NUMBER_OF_THREADS_PER_WORKGROUP {
            let number_of_workgroups =
                (save_base - load_base).div_ceil(NUMBER_OF_THREADS_PER_WORKGROUP);

            rounds.push((load_base, save_base, number_of_workgroups));

            load_base = save_base;
            save_base += number_of_workgroups;
        }
        // 2-1. Scan: calculate the sum of each workgroup in hireachy.
        pass.set_pipeline(scan_sums_pipeline);
        for (load_base, save_base, number_of_workgroups) in rounds.iter() {
            pass.set_push_constants(SCAN_LOAD_BASE_OFFSET, bytemuck::bytes_of(load_base));
            pass.set_push_constants(SCAN_SAVE_BASE_OFFSET, bytemuck::bytes_of(save_base));

            dispatch_workgroup_ext(
                &mut pass,
                *number_of_workgroups,
                max_compute_workgroups_per_dimension,
                WORKGROUP_OFFSET_OFFSET,
            );
        }

        // 2-2. Scan: calculate the prefix sum of last workgroup.
        pass.set_pipeline(scan_last_pipeline);

        pass.set_push_constants(SCAN_LOAD_BASE_OFFSET, bytemuck::bytes_of(&load_base));
        pass.set_push_constants(SCAN_SAVE_BASE_OFFSET, bytemuck::bytes_of(&save_base));

        pass.dispatch_workgroups(1, 1, 1);

        // 2-3. Scan: calculate the prefix sum of each workgroup in hireachy.
        pass.set_pipeline(scan_prfx_pipeline);
        for (load_base, save_base, number_of_workgroups) in rounds.iter().rev() {
            pass.set_push_constants(SCAN_LOAD_BASE_OFFSET, bytemuck::bytes_of(load_base));
            pass.set_push_constants(SCAN_SAVE_BASE_OFFSET, bytemuck::bytes_of(save_base));

            dispatch_workgroup_ext(
                &mut pass,
                *number_of_workgroups,
                max_compute_workgroups_per_dimension,
                WORKGROUP_OFFSET_OFFSET,
            );
        }

        // 3. Assemble the segments to a compacted buffer.
        pass.set_pipeline(assemble_pipeline);

        dispatch_workgroup_ext(
            &mut pass,
            number_of_blocks,
            max_compute_workgroups_per_dimension,
            WORKGROUP_OFFSET_OFFSET,
        );
    }
}

/// The number of keys processed by this workgroup.
const NUMBER_OF_KEYS_OFFSET: u32 = 4;
/// The scan step reads from the `number_segs` buffer starting at this index.
const SCAN_LOAD_BASE_OFFSET: u32 = 8;
/// The scan step writes to the `number_segs` buffer starting at this index.
const SCAN_SAVE_BASE_OFFSET: u32 = 12;

const PUSH_CONSTANT_RANGES: PushConstantRange = PushConstantRange {
    stages: ShaderStages::COMPUTE,
    range: 0..16,
};

pub fn check_load_state(world: &World) -> LoadState {
    let pipeline_cache = world.resource::<PipelineCache>();
    let assemble_segs_pipeline = world.resource::<AssembleSegsPipeline>();

    let (
        count_segs_pipeline_state,
        scan_sums_pipeline_state,
        scan_last_pipeline_state,
        scan_prfx_pipeline_state,
        assemble_pipeline_state,
    ) = (
        pipeline_cache.get_compute_pipeline_state(assemble_segs_pipeline.count_segs_pipeline()),
        pipeline_cache.get_compute_pipeline_state(assemble_segs_pipeline.scan_sums_pipeline()),
        pipeline_cache.get_compute_pipeline_state(assemble_segs_pipeline.scan_last_pipeline()),
        pipeline_cache.get_compute_pipeline_state(assemble_segs_pipeline.scan_prfx_pipeline()),
        pipeline_cache.get_compute_pipeline_state(assemble_segs_pipeline.assemble_pipeline()),
    );

    if let CachedPipelineState::Err(err) = count_segs_pipeline_state {
        return LoadState::Failed(format!(
            "Failed to load assemble_segs: count_segs_pipeline: {}",
            err
        ));
    }

    if let CachedPipelineState::Err(err) = scan_sums_pipeline_state {
        return LoadState::Failed(format!(
            "Failed to load assemble_segs: scan_sums_pipeline: {}",
            err
        ));
    }

    if let CachedPipelineState::Err(err) = scan_last_pipeline_state {
        return LoadState::Failed(format!(
            "Failed to load assemble_segs: scan_last_pipeline: {}",
            err
        ));
    }

    if let CachedPipelineState::Err(err) = scan_prfx_pipeline_state {
        return LoadState::Failed(format!(
            "Failed to load assemble_segs: scan_prfx_pipeline: {}",
            err
        ));
    }

    if let CachedPipelineState::Err(err) = assemble_pipeline_state {
        return LoadState::Failed(format!(
            "Failed to load assemble_segs: assemble_pipeline: {}",
            err
        ));
    }

    if matches!(count_segs_pipeline_state, CachedPipelineState::Ok(_))
        && matches!(scan_sums_pipeline_state, CachedPipelineState::Ok(_))
        && matches!(assemble_pipeline_state, CachedPipelineState::Ok(_))
    {
        return LoadState::Loaded;
    }

    LoadState::OnLoad
}

#[cfg(test)]
mod tests {
    use bevy::{
        render::{
            render_resource::{
                BufferAddress, BufferDescriptor, BufferInitDescriptor, BufferUsages,
                CommandEncoderDescriptor, Maintain, MapMode,
            },
            renderer::RenderQueue,
            Render, RenderPlugin, RenderSet,
        },
        scene::ScenePlugin,
    };

    use crate::physics::{
        get_subgroup_size::GetSubgroupSizePlugin,
        radix_sort::{RadixSortPipeline, RadixSortPlugin},
    };

    use super::*;

    const MAX_NUMBER_OF_KEYS: u32 = 16 * 1024 * 1024;
    const NUMBER_OF_BYTES_PER_KEY: u32 = 4;

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
            .add_plugins(ScenePlugin::default())
            .add_plugins(RenderPlugin {
                synchronous_pipeline_compilation: true,
                ..default()
            })
            .add_plugins(ImagePlugin::default())
            .add_plugins(GetSubgroupSizePlugin)
            .add_plugins(RadixSortPlugin {
                settings: MAX_NUMBER_OF_KEYS.into(),
            })
            .add_plugins(AssembleSegsPlugin);

        app
    }

    fn run_assemble_segs_test(number_of_keys: u32, max_number_of_segments: u32) {
        let mut app = create_unit_test_app();

        let unit_test_system =
            move |render_device: Res<RenderDevice>,
                  render_queue: Res<RenderQueue>,
                  pipeline_cache: Res<PipelineCache>,
                  radix_sort_pipeline: Res<RadixSortPipeline>,
                  radix_bind_group: Res<RadixSortBindGroup>,
                  assemble_segs_pipeline: Res<AssembleSegsPipeline>,
                  assemble_segs_bind_group: Res<AssembleSegsBindGroup>| {
                let keys: Vec<u32> = (0..number_of_keys)
                    .map(|v| v % max_number_of_segments)
                    .collect();

                let i_keys_staging_buf =
                    render_device.create_buffer_with_data(&BufferInitDescriptor {
                        label: Some("unit-test: input keys staging buffer"),
                        usage: BufferUsages::COPY_SRC | BufferUsages::MAP_WRITE,
                        contents: bytemuck::cast_slice(keys.as_slice()),
                    });

                let o_segs_staging_buf = render_device.create_buffer(&BufferDescriptor {
                    label: Some("unit-test: output segments staging buffer"),
                    size: (max_number_of_segments * NUMBER_OF_BYTES_PER_KEY) as BufferAddress,
                    usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                });

                let o_maps_staging_buf = render_device.create_buffer(&BufferDescriptor {
                    label: Some("unit-test: output maps staging buffer"),
                    size: (max_number_of_segments * NUMBER_OF_BYTES_PER_KEY) as BufferAddress,
                    usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                });

                let o_tnum_staging_buf = render_device.create_buffer(&BufferDescriptor {
                    label: Some("unit-test: output tnum staging buffer"),
                    size: 4,
                    usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                });

                let max_compute_workgroups_per_dimension =
                    render_device.limits().max_compute_workgroups_per_dimension;

                let mut encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
                    label: Some("unit_test: radix_sort command encoder"),
                });

                encoder.copy_buffer_to_buffer(
                    &i_keys_staging_buf,
                    0,
                    radix_bind_group.eve_global_keys_buf(),
                    0,
                    (number_of_keys * NUMBER_OF_BYTES_PER_KEY) as BufferAddress,
                );

                super::super::radix_sort::run(
                    &mut encoder,
                    &pipeline_cache,
                    &radix_sort_pipeline,
                    &radix_bind_group,
                    max_compute_workgroups_per_dimension,
                    number_of_keys,
                    0..4,
                    true,
                    true,
                );

                run(
                    &mut encoder,
                    &pipeline_cache,
                    &assemble_segs_pipeline,
                    &assemble_segs_bind_group,
                    max_compute_workgroups_per_dimension,
                    number_of_keys,
                    true,
                );

                let result_size =
                    (max_number_of_segments * NUMBER_OF_BYTES_PER_KEY) as BufferAddress;

                let global_segs_buf = radix_bind_group.odd_global_keys_buf();
                encoder.copy_buffer_to_buffer(
                    global_segs_buf,
                    0,
                    &o_segs_staging_buf,
                    0,
                    result_size,
                );

                let key_seg_map_buf = radix_bind_group.odd_global_vals_buf();
                encoder.copy_buffer_to_buffer(
                    key_seg_map_buf,
                    0,
                    &o_maps_staging_buf,
                    0,
                    result_size,
                );

                let tnumber_seg_buf = assemble_segs_bind_group.tnumber_seg_buf();
                encoder.copy_buffer_to_buffer(tnumber_seg_buf, 0, &o_tnum_staging_buf, 0, 4);

                render_queue.submit([encoder.finish()]);

                let segs_slice = o_segs_staging_buf.slice(0..result_size);
                let maps_slice = o_maps_staging_buf.slice(0..result_size);
                let tnum_slice = o_tnum_staging_buf.slice(0..4);

                segs_slice.map_async(MapMode::Read, |_| ());
                maps_slice.map_async(MapMode::Read, |_| ());
                tnum_slice.map_async(MapMode::Read, |_| ());

                render_device.poll(Maintain::Wait).panic_on_timeout();

                let total_number_segs: u32;

                // assert! tnumber_seg
                {
                    let view = tnum_slice.get_mapped_range();
                    total_number_segs = bytemuck::cast_slice(&view)[0];

                    assert_eq!(
                        total_number_segs,
                        number_of_keys.min(max_number_of_segments)
                    );
                }

                // assert! global_segs
                {
                    let view = segs_slice.get_mapped_range();
                    let data: &[u32] = &bytemuck::cast_slice(&view)[0..total_number_segs as usize];

                    let answer: Vec<u32> = (0..total_number_segs)
                        // the number of keys in each segment
                        .map(|seg_index| {
                            number_of_keys / max_number_of_segments
                                + (number_of_keys % max_number_of_segments > seg_index) as u32
                        })
                        .scan(0u32, |state, x| {
                            let exclusive = *state;
                            *state += x;
                            Some(exclusive)
                        })
                        .collect();

                    assert_eq!(data, &answer);
                }

                // assert! key_seg_map
                {
                    let view = maps_slice.get_mapped_range();
                    let data: &[u32] = &bytemuck::cast_slice(&view)[0..total_number_segs as usize];

                    let answer: Vec<u32> = (0..total_number_segs).collect();

                    assert_eq!(data, &answer);
                }

                o_segs_staging_buf.unmap();
                o_maps_staging_buf.unmap();
                o_tnum_staging_buf.unmap();
            };

        app.sub_app_mut(RenderApp)
            .add_systems(Render, unit_test_system.in_set(RenderSet::Cleanup));

        run_once(&mut app);
    }

    #[test]
    fn test_bs_0() {
        run_assemble_segs_test(0, 127);
        run_assemble_segs_test(0, 256);
    }

    #[test]
    fn test_bs_1() {
        run_assemble_segs_test(1, 127);
        run_assemble_segs_test(1, 256);
    }

    #[test]
    fn test_bs_100() {
        run_assemble_segs_test(100, 127);
        run_assemble_segs_test(100, 256);
    }

    #[test]
    fn test_bs_256() {
        run_assemble_segs_test(256, 127);
        run_assemble_segs_test(256, 256);
    }

    #[test]
    fn test_bs_1000() {
        run_assemble_segs_test(1000, 127);
        run_assemble_segs_test(1000, 256);
    }

    #[test]
    fn test_bs_16x256() {
        run_assemble_segs_test(16 * 256, 127);
        run_assemble_segs_test(16 * 256, 256);
    }

    #[test]
    fn test_bs_16_000_000() {
        run_assemble_segs_test(16_000_000, 127);
        run_assemble_segs_test(16_000_000, 1048576);
    }

    #[test]
    fn test_bs_16_777_216() {
        run_assemble_segs_test(16_777_216, 127);
        run_assemble_segs_test(16_777_216, 1048576);
    }
}
