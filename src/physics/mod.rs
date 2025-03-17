pub mod spatial_hashing;

use bevy::{
    asset::{RenderAssetUsages, load_internal_asset},
    core_pipeline::core_3d::graph::{Core3d, Node3d},
    prelude::*,
    render::{
        Render, RenderApp, RenderSet,
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_asset::RenderAssets,
        render_graph::{self, RenderGraph, RenderLabel},
        render_resource::{
            BindGroup, BindGroupEntries, BindGroupLayout, BindGroupLayoutEntries, BufferUsages,
            CachedComputePipelineId, CachedPipelineState, CommandEncoder, ComputePassDescriptor,
            ComputePipelineDescriptor, PipelineCache, PushConstantRange, ShaderDefVal,
            ShaderStages,
            binding_types::{storage_buffer, storage_buffer_read_only},
        },
        renderer::{RenderContext, RenderDevice},
        storage::{GpuShaderStorageBuffer, ShaderStorageBuffer},
    },
};

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use bevy_radix_sort::{
    EVE_GLOBAL_KEYS_STORAGE_BUFFER_HANDLE, EVE_GLOBAL_VALS_STORAGE_BUFFER_HANDLE,
    GetSubgroupSizePlugin, LoadState, ODD_GLOBAL_KEYS_STORAGE_BUFFER_HANDLE,
    ODD_GLOBAL_VALS_STORAGE_BUFFER_HANDLE, RadixSortBindGroup, RadixSortPipeline, RadixSortPlugin,
    dispatch_workgroup_ext,
};
use spatial_hashing::{SpatialHashingBindGroup, SpatialHashingPipeline, SpatialHashingPlugin};

pub const NUMBER_OF_THREADS_PER_WORKGROUP: u32 = 256;

pub const PHYSICS_SHADER_HANDLE: Handle<Shader> =
    Handle::weak_from_u128(49065411481703621990197661489236519525);

/// If time % 2 == 0, then use `eve_position_with_scale_buf` as input, otherwise use `odd_position_with_scale_buf`.
pub const EVE_POSITION_WITH_SCALE_STORAGE_BUFFER_HANDLE: Handle<ShaderStorageBuffer> =
    Handle::weak_from_u128(156789234567892345678923456789234567892);
/// If time % 2 == 1, then use `odd_position_with_scale_buf` as input, otherwise use `eve_position_with_scale_buf`.
pub const ODD_POSITION_WITH_SCALE_STORAGE_BUFFER_HANDLE: Handle<ShaderStorageBuffer> =
    Handle::weak_from_u128(234567891234567891234567891234567891234);
/// If time % 2 == 0, then use `eve_packed_vfme_buf` as input, otherwise use `odd_packed_vfme_buf`.
pub const EVE_PACKED_VFME_STORAGE_BUFFER_HANDLE: Handle<ShaderStorageBuffer> =
    Handle::weak_from_u128(312345678912345678912345678912345678912);
/// If time % 2 == 1, then use `odd_packed_vfme_buf` as input, otherwise use `eve_packed_vfme_buf`.
pub const ODD_PACKED_VFME_STORAGE_BUFFER_HANDLE: Handle<ShaderStorageBuffer> =
    Handle::weak_from_u128(289346928374692837469283746928374692837);

#[derive(Debug)]
pub struct PhysicsPlugin {
    /// See [`PhysicsSettings::enable_simulation`]
    pub enable_simulation: bool,
    /// See [`PhysicsSettings::enable_spring`]
    pub enable_spring: bool,
    /// See [`PhysicsSettings::number_of_grids_one_dimension`]
    pub number_of_grids_one_dimension: u32,
    /// See [`PhysicsSettings::grid_size`]
    pub grid_size: u32,
    /// See [`PhysicsSettings::half_map_height`]
    pub half_map_height: u32,

    /// See [`PhysicsSettings::gravity`]
    pub gravity: Vec3,

    /// See [`PhysicsSettings::e_envir`]
    pub e_envir: f32,
    /// See [`PhysicsSettings::e_agent`]
    pub e_agent: f32,

    /// See [`PhysicsSettings::k_agent`]
    pub k_agent: f32,
    /// See [`PhysicsSettings::c_agent`]
    pub c_agent: f32,
    /// See [`PhysicsSettings::r_agent`]
    pub r_agent: f32,

    /// See [`PhysicsSettings::max_number_of_agents`]
    pub max_number_of_agents: u32,
    /// See [`PhysicsSettings::time_delta`]
    pub time_delta: f32,
}

impl Default for PhysicsPlugin {
    fn default() -> Self {
        Self {
            enable_simulation: true,
            enable_spring: false,
            number_of_grids_one_dimension: 128,
            grid_size: 8,
            half_map_height: 16,

            gravity: Vec3::new(0., -9.8, 0.),

            e_envir: 0.90,
            e_agent: 0.95,

            k_agent: 25.,
            c_agent: 0.1,
            r_agent: 0.9,

            max_number_of_agents: 16 * 1024,
            time_delta: 1. / 120.,
        }
    }
}

impl Plugin for PhysicsPlugin {
    fn build(&self, app: &mut App) {
        let physics_settings = PhysicsSettings::from(self);

        app.register_type::<PhysicsSettings>()
            .add_plugins(GetSubgroupSizePlugin)
            .add_plugins(RadixSortPlugin {
                settings: self.max_number_of_agents.into(),
            })
            .add_plugins(SpatialHashingPlugin)
            .add_plugins(PhysicsInnerPlugin {
                settings: physics_settings,
            });
    }
}

/// NOTE: Just for resource loading order, `bevy` does not provide a way to control the loading order of [`Plugin`],
/// so it has to be implemented manually.
struct PhysicsInnerPlugin {
    settings: PhysicsSettings,
}

impl Plugin for PhysicsInnerPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            PHYSICS_SHADER_HANDLE,
            "physics.wgsl",
            Shader::from_wgsl
        );

        create_shader_storage_buffers(app, self.settings.max_number_of_agents());

        app.insert_resource(self.settings)
            .add_plugins(ExtractResourcePlugin::<PhysicsSettings>::default());

        let render_app = app.sub_app_mut(RenderApp);

        render_app.insert_resource(self.settings).add_systems(
            Render,
            PhysicsBindGroup::initialize
                .in_set(RenderSet::PrepareBindGroups)
                .run_if(not(resource_exists::<PhysicsBindGroup>)),
        );

        if let Some(core3d) = render_app
            .world_mut()
            .resource_mut::<RenderGraph>()
            .get_sub_graph_mut(Core3d)
        {
            core3d.add_node(PhysicsNodeLabel, PhysicsNode::onload(self.settings.into()));
            core3d.add_node_edge(Node3d::Upscaling, PhysicsNodeLabel);
        }
    }

    fn finish(&self, app: &mut App) {
        app.sub_app_mut(RenderApp)
            .init_resource::<PhysicsPipeline>();
    }
}

/// TODO: function for test, delete it after testing
fn position_with_scale_data(max_number_of_agents: u32) -> Vec<Vec4> {
    const SPACING: u32 = 4;

    let log = max_number_of_agents.ilog2();
    let log_x = log / 2;
    let log_z = log - log_x;

    let count_x = 1u32 << log_x;
    let count_z = 1u32 << log_z;

    let mut position_with_scale_data = Vec::with_capacity(max_number_of_agents as usize);
    for x in 0..count_x {
        let to_x = x as f32 - 0.5 * (count_x as f32 - 1.);
        for z in 0..count_z {
            let to_z = z as f32 - 0.5 * (count_z as f32 - 1.);

            let position_with_scale =
                Vec4::new(to_x * (SPACING as f32), 4., to_z * (SPACING as f32), 1.0);
            position_with_scale_data.push(position_with_scale);
        }
    }

    position_with_scale_data
}

/// TODO: function for test, delete it after testing
fn packed_vfme_data(max_number_of_agents: u32) -> Vec<UVec4> {
    let mut rng = ChaCha8Rng::seed_from_u64(0);

    let mut packed_vfme_data = Vec::with_capacity(max_number_of_agents as usize);
    for _ in 0..max_number_of_agents {
        let velocity = Vec3::new(rng.gen_range(-0.2..0.2), 0., rng.gen_range(-0.2..0.2));
        let extforce = Vec3::new(0., 0., 0.);
        let massv = 1f32;
        let packed = pack_vfme(velocity, extforce, massv);
        packed_vfme_data.push(packed);
    }

    packed_vfme_data
}

fn pack_vfme(v: Vec3, f: Vec3, m: f32) -> UVec4 {
    let (vx, vy, vz) = (v.x as f16, v.y as f16, v.z as f16);
    let (fx, fy, fz) = (f.x as f16, f.y as f16, f.z as f16);
    let m = m as f16;
    let e = 0f16;

    let (uvx, uvy, uvz) = (
        vx.to_bits() as u32,
        vy.to_bits() as u32,
        vz.to_bits() as u32,
    );
    let (ufx, ufy, ufz) = (
        fx.to_bits() as u32,
        fy.to_bits() as u32,
        fz.to_bits() as u32,
    );
    let um = m.to_bits() as u32;
    let ue = e.to_bits() as u32;

    let cx = uvx | (uvy << 16);
    let cy = uvz | (ufx << 16);
    let cz = ufy | (ufz << 16);
    let cw = um | (ue << 16);

    UVec4::new(cx, cy, cz, cw)
}

fn create_shader_storage_buffers(app: &mut App, max_number_of_agents: u32) {
    let mut sbufs = app
        .world_mut()
        .resource_mut::<Assets<ShaderStorageBuffer>>();

    let usages = BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST;
    let size = max_number_of_agents as usize * std::mem::size_of::<Vec4>();

    let position_with_scale_data = position_with_scale_data(max_number_of_agents);
    let packed_vfme_data = packed_vfme_data(max_number_of_agents);

    let mut eve_position_with_scale_buf =
        ShaderStorageBuffer::with_size(size, RenderAssetUsages::default());
    eve_position_with_scale_buf.set_data(position_with_scale_data);
    eve_position_with_scale_buf.buffer_description.label =
        Some("physics: eve_position_with_scale buffer");
    eve_position_with_scale_buf.buffer_description.usage = usages;

    let mut eve_packed_vfme_buf =
        ShaderStorageBuffer::with_size(size, RenderAssetUsages::default());
    eve_packed_vfme_buf.set_data(packed_vfme_data);
    eve_packed_vfme_buf.buffer_description.label = Some("physics: eve_packed_vfme buffer");
    eve_packed_vfme_buf.buffer_description.usage = usages;

    let mut odd_position_with_scale_buf =
        ShaderStorageBuffer::with_size(size, RenderAssetUsages::default());
    odd_position_with_scale_buf.buffer_description.label =
        Some("physics: odd_position_with_scale buffer");
    odd_position_with_scale_buf.buffer_description.usage = usages;

    let mut odd_packed_vfme_buf =
        ShaderStorageBuffer::with_size(size, RenderAssetUsages::default());
    odd_packed_vfme_buf.buffer_description.label = Some("physics: odd_packed_vfme buffer");
    odd_packed_vfme_buf.buffer_description.usage = usages;

    sbufs.insert(
        EVE_POSITION_WITH_SCALE_STORAGE_BUFFER_HANDLE.id(),
        eve_position_with_scale_buf,
    );
    sbufs.insert(
        EVE_PACKED_VFME_STORAGE_BUFFER_HANDLE.id(),
        eve_packed_vfme_buf,
    );
    sbufs.insert(
        ODD_POSITION_WITH_SCALE_STORAGE_BUFFER_HANDLE.id(),
        odd_position_with_scale_buf,
    );
    sbufs.insert(
        ODD_PACKED_VFME_STORAGE_BUFFER_HANDLE.id(),
        odd_packed_vfme_buf,
    );
}

#[derive(Resource, ExtractResource, Debug, Clone, Copy, Reflect)]
#[reflect(Resource, Default, Debug)]
pub struct PhysicsSettings {
    /// whether or not enable physics simulation
    pub enable_simulation: bool,
    /// whether or not enable spring simulation
    pub enable_spring: bool,
    /// The number of grids in one dimension.
    ///
    /// Plase use a power of 2, such as `2^10`, `2^11`, `2^12`, etc.
    ///
    /// ## Important
    ///
    /// The count of `collision grid` is equal to `number_of_grids_one_dimension * number_of_grids_one_dimension`.
    /// Please ensure that the count of `collision grids` does not exceed `16,777,216 (2^24)`, otherwise calculation errors will occur.
    ///
    /// ## Why `16,777,216 (2^24)`?
    ///
    /// This is considered from two aspects:
    ///
    /// 1. 16,777,216 `collision grids` are sufficient to cover most application scenarios.
    /// 2. 16,777,216 only takes up 3 bytes. When executing [`RadixSortPipeline`],
    ///     only 3 passes of sub-sorting are needed, instead of 4 passes, which can save 1/4 of the overhead.
    ///
    /// TODO: Support dynamically adjusting `pass_range` of `radix sort` based on the count of `collision grid`?
    /// (Using 4 sub-sorts can reduce a lot of unnecessary code, but it will increase the performance overhead by 1/4.
    /// Perhaps by implementing incremental sorting, we can reduce the code size without increasing the overhead.)
    pub number_of_grids_one_dimension: u32,
    /// The size of one grid
    pub grid_size: u32,
    /// The half height of the map
    pub half_map_height: u32,
    /// The gravity acceleration
    pub gravity: Vec3,
    /// The coefficient of restitution for collisions between agents and environment(a boudning box)
    pub e_envir: f32,
    /// The coefficient of restitution for collisions between agents and agents
    pub e_agent: f32,
    /// The coefficient of stiffness for spring collisions between agent and agent/environment
    pub k_agent: f32,
    /// The coefficient of damping for spring collisions between agent and agent/environment
    pub c_agent: f32,
    /// Elastic Stiffness threshold, used to determine whether the `Agent` should behave as a rigid body
    /// or a spring when interacting with the environment.
    /// When `makeup < -r_agent * scale`, it behaves as a rigid body; otherwise, it behaves as a spring.
    ///
    /// Range: [0., 1.]
    pub r_agent: f32,
    /// The time interval between two physics updates.
    ///
    /// The value is 17ms, which is equivalent to 60fps.
    ///
    /// - value:  8, fps: 120
    /// - value: 17, fps: 60
    /// - value: 33, fps: 30
    /// - value: 67, fps: 15
    pub time_delta: f32,
    /// ## Important
    ///
    /// [`PhysicsSettings::max_number_of_agents`] must be greater than or equal to [`PhysicsSettings::number_of_grids_one_dimension`]^2.
    ///
    /// (`number_of_grids_one_dimension` * `number_of_grids_one_dimension` is the count of `collision grids`)
    ///
    /// This is because when computing [`SpatialHashingPipeline`], the
    /// [`EVE_GLOBAL_KEYS_STORAGE_BUFFER_HANDLE`]/[`EVE_GLOBAL_VALS_STORAGE_BUFFER_HANDLE`],
    /// [`ODD_GLOBAL_KEYS_STORAGE_BUFFER_HANDLE`]/[`ODD_GLOBAL_VALS_STORAGE_BUFFER_HANDLE`],
    ///  is used to store the start index and close index of the `collision grid`.
    /// Therefore, the number of `collision grids` cannot exceed the [`PhysicsSettings::max_number_of_agents`].
    max_number_of_agents: u32,
}

impl PhysicsSettings {
    pub fn max_number_of_agents(&self) -> u32 {
        self.max_number_of_agents
    }

    pub fn max_number_of_grids_on_dimension(&self) -> u32 {
        self.max_number_of_agents().isqrt()
    }

    pub fn map_size(&self) -> f32 {
        self.map_size_u() as f32
    }

    #[allow(dead_code)]
    pub fn half_map_size(&self) -> f32 {
        self.half_map_size_u() as f32
    }

    #[allow(dead_code)]
    pub fn map_height(&self) -> f32 {
        2. * self.half_map_height()
    }

    #[allow(dead_code)]
    pub fn half_map_height(&self) -> f32 {
        self.half_map_height as f32
    }

    pub fn map_size_u(&self) -> u32 {
        self.number_of_grids_one_dimension * self.grid_size
    }

    pub fn half_map_size_u(&self) -> u32 {
        self.map_size_u() / 2
    }
}

impl From<&PhysicsPlugin> for PhysicsSettings {
    fn from(pp: &PhysicsPlugin) -> Self {
        Self {
            enable_simulation: pp.enable_simulation,
            enable_spring: pp.enable_spring,
            number_of_grids_one_dimension: pp.number_of_grids_one_dimension,
            grid_size: pp.grid_size,
            half_map_height: pp.half_map_height,

            gravity: pp.gravity,

            e_envir: pp.e_envir,
            e_agent: pp.e_agent,

            k_agent: pp.k_agent,
            c_agent: pp.c_agent,
            r_agent: pp.r_agent,

            max_number_of_agents: pp.max_number_of_agents,
            time_delta: pp.time_delta,
        }
    }
}

impl From<PhysicsPlugin> for PhysicsSettings {
    fn from(pp: PhysicsPlugin) -> Self {
        Self::from(&pp)
    }
}

impl Default for PhysicsSettings {
    fn default() -> Self {
        PhysicsPlugin::default().into()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct CompiledConsts {
    number_of_grids_one_dimension: u32,
    grid_size: u32,
    half_map_height: u32,
    half_map_size: u32,

    gravity: Vec3,

    e_envir: f32,
    e_agent: f32,

    k_agent: f32,
    c_agent: f32,
    r_agent: f32,

    time_delta: f32,

    enable_spring: bool,
}

impl CompiledConsts {
    fn shader_defs(&self) -> Vec<ShaderDefVal> {
        vec![
            ShaderDefVal::UInt(
                "NUMBER_OF_THREADS_PER_WORKGROUP".into(),
                NUMBER_OF_THREADS_PER_WORKGROUP,
            ),
            ShaderDefVal::UInt(
                "NUMBER_OF_GRIDS_ONE_DIMENSION".into(),
                self.number_of_grids_one_dimension,
            ),
            ShaderDefVal::UInt("GRID_SIZE".into(), self.grid_size),
            ShaderDefVal::UInt("HALF_MAP_HEIGHT".into(), self.half_map_height),
            ShaderDefVal::UInt("HALF_MAP_SIZE".into(), self.half_map_size),
            ShaderDefVal::Int("GRAVITY_X".into(), (self.gravity.x * 1000.) as i32),
            ShaderDefVal::Int("GRAVITY_Y".into(), (self.gravity.y * 1000.) as i32),
            ShaderDefVal::Int("GRAVITY_Z".into(), (self.gravity.z * 1000.) as i32),
            ShaderDefVal::UInt("E_ENVIR".into(), (self.e_envir * 1000.) as u32),
            ShaderDefVal::UInt("E_AGENT".into(), (self.e_agent * 1000.) as u32),
            ShaderDefVal::UInt("K_AGENT".into(), (self.k_agent * 1000.) as u32),
            ShaderDefVal::UInt("C_AGENT".into(), (self.c_agent * 1000.) as u32),
            ShaderDefVal::UInt("R_AGENT".into(), (self.r_agent * 1000.) as u32),
            ShaderDefVal::UInt("TIME_DELTA".into(), (self.time_delta * 1000.) as u32),
            ShaderDefVal::Bool("ENABLE_SPRING".into(), self.enable_spring),
        ]
    }
}

impl From<PhysicsSettings> for CompiledConsts {
    fn from(ps: PhysicsSettings) -> Self {
        Self::from(&ps)
    }
}

impl From<&PhysicsSettings> for CompiledConsts {
    fn from(ps: &PhysicsSettings) -> Self {
        Self {
            number_of_grids_one_dimension: ps.number_of_grids_one_dimension,
            grid_size: ps.grid_size,
            half_map_height: ps.half_map_height,
            half_map_size: ps.half_map_size_u(),

            gravity: ps.gravity,

            e_envir: ps.e_envir,
            e_agent: ps.e_agent,

            k_agent: ps.k_agent,
            c_agent: ps.c_agent,
            r_agent: ps.r_agent,

            time_delta: ps.time_delta,

            enable_spring: ps.enable_spring,
        }
    }
}

impl Default for CompiledConsts {
    fn default() -> Self {
        PhysicsSettings::default().into()
    }
}

#[derive(Resource, Debug, Clone)]
pub struct PhysicsPipeline {
    init_pipeline: CachedComputePipelineId,
    loop_pipeline: CachedComputePipelineId,
    bind_group_layout: BindGroupLayout,
}

impl PhysicsPipeline {
    fn new(
        render_device: &RenderDevice,
        pipeline_cache: &PipelineCache,
        compiled_consts: &CompiledConsts,
    ) -> Self {
        let bind_group_layout = render_device.create_bind_group_layout(
            "physics bindgroup layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    // A buffer stores sorted `agent_index`
                    storage_buffer_read_only::<u32>(false),
                    // A buffer stores `collision grid` start index
                    storage_buffer_read_only::<u32>(false),
                    // A buffer stores `collision grid` close index
                    storage_buffer_read_only::<u32>(false),
                    // A buffer which is used to store `grid_index` of each agent, which is the key of radix sort to sort
                    storage_buffer::<u32>(false),
                    // A buffer read position_with_scale from
                    storage_buffer_read_only::<Vec4>(false),
                    // A buffer read packed_vfme from
                    storage_buffer_read_only::<Vec4>(false),
                    // A buffer write position_with_scale to
                    storage_buffer::<Vec4>(false),
                    // A buffer write packed_vfme to
                    storage_buffer::<Vec4>(false),
                ),
            ),
        );

        let cdefs = compiled_consts.shader_defs();

        let init_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("physics: init pipeline".into()),
            layout: vec![bind_group_layout.clone()],
            push_constant_ranges: vec![PUSH_CONSTANT_RANGES],
            shader: PHYSICS_SHADER_HANDLE,
            shader_defs: [cdefs.as_slice(), &["INIT_PIPELINE".into()]].concat(),
            entry_point: "main".into(),
            zero_initialize_workgroup_memory: false,
        });

        let loop_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("physics: loop pipeline".into()),
            layout: vec![bind_group_layout.clone()],
            push_constant_ranges: vec![PUSH_CONSTANT_RANGES],
            shader: PHYSICS_SHADER_HANDLE,
            shader_defs: [cdefs.as_slice(), &["LOOP_PIPELINE".into()]].concat(),
            entry_point: "main".into(),
            zero_initialize_workgroup_memory: false,
        });

        Self {
            init_pipeline,
            loop_pipeline,
            bind_group_layout,
        }
    }

    pub fn init_pipeline(&self) -> CachedComputePipelineId {
        self.init_pipeline
    }

    pub fn loop_pipeline(&self) -> CachedComputePipelineId {
        self.loop_pipeline
    }

    pub fn bind_group_layout(&self) -> &BindGroupLayout {
        &self.bind_group_layout
    }
}

impl FromWorld for PhysicsPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let pipeline_cache = world.resource::<PipelineCache>();
        let physics_settings = world.resource::<PhysicsSettings>();
        let compiled_consts = CompiledConsts::from(physics_settings);

        Self::new(render_device, pipeline_cache, &compiled_consts)
    }
}

#[derive(Resource, Debug, Clone)]
pub struct PhysicsBindGroup {
    /// If time % 2 == 0, then use `eve_physics_bind_group`, otherwise use `odd_physics_bind_group`.
    eve_physics_bind_group: BindGroup,
    /// If time % 2 == 1, then use `odd_physics_bind_group`, otherwise use `eve_physics_bind_group`.
    odd_physics_bind_group: BindGroup,

    time: i32,
}

impl PhysicsBindGroup {
    pub fn initialize(
        mut commands: Commands,
        physics_pipeline: Res<PhysicsPipeline>,
        render_device: Res<RenderDevice>,
        sbufs: Res<RenderAssets<GpuShaderStorageBuffer>>,
    ) {
        let bind_group_layout = physics_pipeline.bind_group_layout();

        let eve_position_with_scale_buf = sbufs
            .get(EVE_POSITION_WITH_SCALE_STORAGE_BUFFER_HANDLE.id())
            .unwrap();
        let eve_packed_vfme_buf = sbufs
            .get(EVE_PACKED_VFME_STORAGE_BUFFER_HANDLE.id())
            .unwrap();
        let odd_position_with_scale_buf = sbufs
            .get(ODD_POSITION_WITH_SCALE_STORAGE_BUFFER_HANDLE.id())
            .unwrap();
        let odd_packed_vfme_buf = sbufs
            .get(ODD_PACKED_VFME_STORAGE_BUFFER_HANDLE.id())
            .unwrap();

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

        let eve_physics_bind_group = render_device.create_bind_group(
            "physics bind group eve",
            bind_group_layout,
            &BindGroupEntries::sequential((
                // Read sorted `agent_index` from this buffer
                odd_global_vals_buf.buffer.as_entire_binding(),
                // Read `collision grid` start index from this buffer
                //
                // Because of the odd buffer is staging buffer, so the `collision grid` start index stored in the odd buffer
                eve_global_keys_buf.buffer.as_entire_binding(),
                // Read `collision grid` close index from this buffer
                //
                // Because of the odd buffer is staging buffer, so the `collision grid` close index stored in the odd buffer
                eve_global_vals_buf.buffer.as_entire_binding(),
                // Write unsorted `grid_index` of each agent to this buffer
                odd_global_keys_buf.buffer.as_entire_binding(),
                // physics resources
                eve_position_with_scale_buf.buffer.as_entire_binding(),
                eve_packed_vfme_buf.buffer.as_entire_binding(),
                odd_position_with_scale_buf.buffer.as_entire_binding(),
                odd_packed_vfme_buf.buffer.as_entire_binding(),
            )),
        );

        let odd_physics_bind_group = render_device.create_bind_group(
            "physics bind group odd",
            bind_group_layout,
            &BindGroupEntries::sequential((
                // Read sorted `agent_index` from this buffer
                eve_global_vals_buf.buffer.as_entire_binding(),
                // Read `collision grid` start index from this buffer
                //
                // Because of the even buffer is staging buffer, so the `collision grid` start index stored in the even buffer
                odd_global_keys_buf.buffer.as_entire_binding(),
                // Read `collision grid` close index from this buffer
                //
                // Because of the even buffer is staging buffer, so the `collision grid` close index stored in the even buffer
                odd_global_vals_buf.buffer.as_entire_binding(),
                // Write unsorted `grid_index` of each agent to this buffer
                eve_global_keys_buf.buffer.as_entire_binding(),
                // physics resources
                odd_position_with_scale_buf.buffer.as_entire_binding(),
                odd_packed_vfme_buf.buffer.as_entire_binding(),
                eve_position_with_scale_buf.buffer.as_entire_binding(),
                eve_packed_vfme_buf.buffer.as_entire_binding(),
            )),
        );

        let physics_bind_group = PhysicsBindGroup {
            eve_physics_bind_group,
            odd_physics_bind_group,

            time: -1,
        };

        commands.insert_resource(physics_bind_group);
    }

    #[allow(dead_code)]
    pub fn time(&self) -> i32 {
        self.time
    }

    /// Returns None if not started, otherwise alternates between Some(true) and Some(false) every tick
    pub fn is_even(&self) -> Option<bool> {
        if self.time == -1 {
            None
        } else {
            Some(self.time % 2 == 0)
        }
    }

    /// Returns None if not started,
    /// otherwise alternates between [`PhysicsBindGroup::eve_physics_bind_group`] and [`PhysicsBindGroup::odd_physics_bind_group`] every tick
    pub fn bind_group(&self) -> Option<&BindGroup> {
        match self.is_even() {
            Some(is_even) if is_even => Some(&self.eve_physics_bind_group),
            Some(is_even) if !is_even => Some(&self.odd_physics_bind_group),
            _ => None,
        }
    }

    fn next_cycle(&mut self) {
        self.time += 1;
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum PhysicsState {
    OnLoad(CompiledConsts),
    Loaded(CompiledConsts),
}

#[derive(Debug, Clone, Eq, PartialEq, Hash, RenderLabel)]
struct PhysicsNodeLabel;

impl Default for PhysicsState {
    fn default() -> Self {
        Self::OnLoad(Default::default())
    }
}

#[derive(Default, Clone, Copy, Debug, PartialEq)]
struct PhysicsNode {
    state: PhysicsState,
}

impl PhysicsNode {
    fn onload(cc: CompiledConsts) -> Self {
        Self {
            state: PhysicsState::OnLoad(cc),
        }
    }
}

impl render_graph::Node for PhysicsNode {
    fn update(&mut self, world: &mut World) {
        let physics_settings = world.resource::<PhysicsSettings>();

        if !physics_settings.enable_simulation {
            return;
        }

        match self.state {
            PhysicsState::OnLoad(cc) => {
                let physics_load_state = check_load_state(world);
                let spatial_hashing_load_state = spatial_hashing::check_load_state(world);
                let radix_sort_load_state = bevy_radix_sort::check_load_state(world);

                if let LoadState::Failed(err) = &physics_load_state {
                    panic!("{}", err);
                }

                if let LoadState::Failed(err) = &spatial_hashing_load_state {
                    panic!("{}", err);
                }

                if let LoadState::Failed(err) = &radix_sort_load_state {
                    panic!("{}", err);
                }

                if matches!(physics_load_state, LoadState::Loaded)
                    && matches!(spatial_hashing_load_state, LoadState::Loaded)
                    && matches!(radix_sort_load_state, LoadState::Loaded)
                {
                    self.state = PhysicsState::Loaded(cc);
                    world.resource_mut::<PhysicsBindGroup>().next_cycle();
                }
            }
            PhysicsState::Loaded(cc) => {
                let ncc = CompiledConsts::from(physics_settings);
                if ncc != cc {
                    log::info!("recompile physics pipeline..");

                    let pipeline_cache = world.resource::<PipelineCache>();
                    let physics_pipeline = world.resource::<PhysicsPipeline>();

                    let cdefs = ncc.shader_defs();

                    let new_init_pipeline =
                        pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                            shader_defs: [cdefs.as_slice(), &["INIT_PIPELINE".into()]].concat(),
                            ..pipeline_cache
                                .get_compute_pipeline_descriptor(physics_pipeline.init_pipeline())
                                .clone()
                        });
                    let new_loop_pipeline =
                        pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                            shader_defs: [cdefs.as_slice(), &["LOOP_PIPELINE".into()]].concat(),
                            ..pipeline_cache
                                .get_compute_pipeline_descriptor(physics_pipeline.loop_pipeline())
                                .clone()
                        });

                    let mut physics_pipeline = world.resource_mut::<PhysicsPipeline>();
                    physics_pipeline.init_pipeline = new_init_pipeline;
                    physics_pipeline.loop_pipeline = new_loop_pipeline;

                    self.state = PhysicsState::OnLoad(ncc);
                } else {
                    world.resource_mut::<PhysicsBindGroup>().next_cycle();
                }
            }
        }
    }

    /// See the work flow: ![work-flow](./physics-work-flow.png)
    fn run(
        &self,
        _graph: &mut render_graph::RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), render_graph::NodeRunError> {
        let physics_settings = world.resource::<PhysicsSettings>();

        if !physics_settings.enable_simulation || matches!(self.state, PhysicsState::OnLoad(_)) {
            return Ok(());
        }

        let max_compute_workgroups_per_dimension = {
            let render_device = world.resource::<RenderDevice>();
            render_device.limits().max_compute_workgroups_per_dimension
        };

        let pipeline_cache = world.resource::<PipelineCache>();

        let physics_pipeline = world.resource::<PhysicsPipeline>();
        let radix_sort_pipeline = world.resource::<RadixSortPipeline>();

        let spatial_hashing_pipeline = world.resource::<SpatialHashingPipeline>();
        let spatial_hashing_bind_group = world.resource::<SpatialHashingBindGroup>();

        let physics_bind_group = world.resource::<PhysicsBindGroup>();
        let radix_sort_bind_group = world.resource::<RadixSortBindGroup>();

        let number_of_agents = physics_settings.max_number_of_agents();

        let encoder = render_context.command_encoder();

        let is_even = physics_bind_group.is_even().unwrap();

        bevy_radix_sort::run(
            encoder,
            pipeline_cache,
            radix_sort_pipeline,
            radix_sort_bind_group,
            max_compute_workgroups_per_dimension,
            number_of_agents,
            0..3,
            true,
            is_even,
        );

        spatial_hashing::run(
            encoder,
            pipeline_cache,
            spatial_hashing_pipeline,
            spatial_hashing_bind_group,
            max_compute_workgroups_per_dimension,
            number_of_agents,
            !is_even,
        );

        run(
            encoder,
            pipeline_cache,
            physics_pipeline,
            physics_bind_group,
            max_compute_workgroups_per_dimension,
            number_of_agents,
        );

        Ok(())
    }
}

pub fn run(
    encoder: &mut CommandEncoder,
    pipeline_cache: &PipelineCache,
    physics_pipeline: &PhysicsPipeline,
    physics_bind_group: &PhysicsBindGroup,
    max_compute_workgroups_per_dimension: u32,
    number_of_agents: u32,
) {
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("physics pass"),
            ..default()
        });

        if physics_bind_group.time == 0 {
            let init_pipeline = pipeline_cache
                .get_compute_pipeline(physics_pipeline.init_pipeline())
                .unwrap();
            pass.set_pipeline(init_pipeline);
        } else {
            let loop_pipeline = pipeline_cache
                .get_compute_pipeline(physics_pipeline.loop_pipeline())
                .unwrap();
            pass.set_pipeline(loop_pipeline);
        }

        pass.set_bind_group(0, physics_bind_group.bind_group().unwrap(), &[]);

        pass.set_push_constants(
            NUMBER_OF_AGENTS_OFFSET,
            bytemuck::bytes_of(&number_of_agents),
        );

        let number_of_workgroups = number_of_agents.div_ceil(NUMBER_OF_THREADS_PER_WORKGROUP);
        dispatch_workgroup_ext(
            &mut pass,
            number_of_workgroups,
            max_compute_workgroups_per_dimension,
            WORKGROUP_OFFSET_OFFSET,
        );
    }
}

fn check_load_state(world: &World) -> LoadState {
    let pipeline_cache = world.resource::<PipelineCache>();
    let physics_pipeline = world.resource::<PhysicsPipeline>();

    let (init_pipeline_state, loop_pipeline_state) = (
        pipeline_cache.get_compute_pipeline_state(physics_pipeline.init_pipeline()),
        pipeline_cache.get_compute_pipeline_state(physics_pipeline.loop_pipeline()),
    );

    if let CachedPipelineState::Err(err) = init_pipeline_state {
        return LoadState::Failed(format!("Failed to load physics:init_pipeline: {:?}", err));
    }

    if let CachedPipelineState::Err(err) = loop_pipeline_state {
        return LoadState::Failed(format!("Failed to load physics:loop_pipeline: {:?}", err));
    }

    if matches!(init_pipeline_state, CachedPipelineState::Ok(_))
        && matches!(loop_pipeline_state, CachedPipelineState::Ok(_))
    {
        return LoadState::Loaded;
    }

    LoadState::OnLoad
}

/// In most cases, the parameters `x`, `y`, `z` in [`ComputePass::dispatch_workgroups(x: u32, y: u32, z: u32)`]
/// are limited to the range\[1, 65535\](the maximum number of workgroups per dimension can be queried through
/// [`Limits::max_compute_workgroups_per_dimension`]).
///
/// When dealing with a particularly large 1-dimensional array, for example, `number_of_keys` = 2^24,
/// then `number_of_workgroups` = 2^16 = 65536; So (65536, 1, 1), (1, 65536, 1), and (1, 1, 65536) all exceed
/// the valid range and will be rejected by the graphics API.
///
/// Therefore, the simplest solution is to split one `dispatch_workgroups(..)` into two or more. Here I choose two:
/// - `workgroup_offset` = 0:           dispatch_workgroups(65535, 1, 1)
/// - `workgroup_offset` = 65535,       dispatch_workgroups(1, 1, 1)
///
/// If `number_of_workgroups` is even larger, for example, `number_of_workgroups` = 2^24, then:
/// - `workgroup_offset` = 0:           dispatch_workgroups(65535, 256, 1)
/// - `workgroup_offset` = 16776960:    dispatch_workgroups(256, 1, 1)
///
/// (Complaint: This is a very annoying limitation that adds unnecessary complexity to the code, but currently there is no better solution)
const WORKGROUP_OFFSET_OFFSET: u32 = 0;
/// The number of keys to be sorted.
const NUMBER_OF_AGENTS_OFFSET: u32 = 4;

const PUSH_CONSTANT_RANGES: PushConstantRange = PushConstantRange {
    stages: ShaderStages::COMPUTE,
    range: 0..8,
};
