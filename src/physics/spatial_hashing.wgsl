@group(0) @binding(0) var<storage, read      > collision_grid_indexs: array<u32>;
@group(0) @binding(1) var<storage, read_write> collision_grid_start_indexs: array<u32>;
@group(0) @binding(2) var<storage, read_write> collision_grid_close_indexs: array<u32>;

struct PushConstants {
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
    workgroup_offset: u32,
    /// The number of agents to be simulated, which is configured in [`PhysicsSetting`].
    number_of_agents: u32,
}
var<push_constant> pc: PushConstants;

#ifdef ZEROING_PIPELINE
@compute @workgroup_size(#NUMBER_OF_THREADS_PER_WORKGROUP, 1, 1)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3u,
    @builtin(num_workgroups) num_workgroups: vec3u,
    @builtin(local_invocation_id) local_invocation_id: vec3u
) {
    let workgroup_index = get_workgroup_index(workgroup_id, num_workgroups);
    let thread_index = get_global_thread_index(workgroup_index, local_invocation_id.x);

    if thread_index >= pc.number_of_agents { return; }
    
    // NOTE: use 0xFFFFFFFFu to represent the invalid index
    collision_grid_start_indexs[thread_index] = 0xFFFFFFFFu;
}
#endif

#ifdef HASHING_PIPELINE
@compute @workgroup_size(#NUMBER_OF_THREADS_PER_WORKGROUP, 1, 1)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3u,
    @builtin(num_workgroups) num_workgroups: vec3u,
    @builtin(local_invocation_id) local_invocation_id: vec3u
) {
    let workgroup_index = get_workgroup_index(workgroup_id, num_workgroups);
    let thread_index = get_global_thread_index(workgroup_index, local_invocation_id.x);

    if thread_index >= pc.number_of_agents { return; }

    let collision_grid_index = collision_grid_indexs[thread_index];

    // write the start index of the collision grid
    var prv_collision_grid_index = 0xFFFFFFFFu - collision_grid_index;
    if thread_index > 0u { prv_collision_grid_index = collision_grid_indexs[thread_index - 1u]; }
    if collision_grid_index != prv_collision_grid_index { collision_grid_start_indexs[collision_grid_index] = thread_index; }

    // write the close index of the collision grid
    var nxt_collision_grid_index = 0xFFFFFFFFu - collision_grid_index;
    if thread_index < pc.number_of_agents - 1u { nxt_collision_grid_index = collision_grid_indexs[thread_index + 1u]; }
    if collision_grid_index != nxt_collision_grid_index { collision_grid_close_indexs[collision_grid_index] = thread_index; }
}
#endif

fn get_workgroup_index(workgroup_id: vec3u, num_workgroups: vec3u) -> u32 {
    return workgroup_id.y * num_workgroups.x + workgroup_id.x + pc.workgroup_offset;
}

fn get_global_thread_index(workgroup_index: u32, local_invocation_id_x: u32) -> u32 {
    return workgroup_index * #NUMBER_OF_THREADS_PER_WORKGROUP + local_invocation_id_x;
}