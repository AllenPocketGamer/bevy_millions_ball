@group(0) @binding(0) var<storage, read      > global_keys: array<u32>;
@group(0) @binding(1) var<storage, read_write> number_segs: array<u32>;
@group(0) @binding(2) var<storage, read_write> global_segs: array<u32>;
@group(0) @binding(3) var<storage, read_write> key_seg_map: array<u32>;
@group(0) @binding(4) var<storage, read_write> tnumber_seg: u32;        // The total number of segments

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
    /// The number of keys to be assemble, which is configured in [`PhysicsSetting`].
    number_of_keys: u32,
    /// The scan step reads from the `number_segs` buffer starting at this index.
    scan_load_base: u32,
    /// The scan step writes to the `number_segs` buffer starting at this index.
    scan_save_base: u32,
}
var<push_constant> pc: PushConstants;

const NUMBER_OF_KEYS_PER_WORKGROUP: u32 = #NUMBER_OF_THREADS_PER_WORKGROUP * #NUMBER_OF_ROWS_PER_WORKGROUP;
const NUMBER_OF_SUBGROUPS: u32  = #NUMBER_OF_THREADS_PER_WORKGROUP / #NUMBER_OF_THREADS_PER_SUBGROUP;

var<workgroup> subgroup_sums: array<u32, NUMBER_OF_SUBGROUPS>;

fn scan_exclusive(value: u32, subgroup_id: u32, subgroup_invocation_id: u32) -> u32 {
    let subgroup_prefix_sum = subgroupInclusiveAdd(value);

    if subgroup_invocation_id == #NUMBER_OF_THREADS_PER_SUBGROUP - 1u { subgroup_sums[subgroup_id] = subgroup_prefix_sum; }
    workgroupBarrier();
    
    let prev_subgroup_sum = select(0u, subgroup_sums[subgroup_invocation_id], subgroup_invocation_id < subgroup_id);
    let prev_sum = subgroupAdd(prev_subgroup_sum);

    return prev_sum + subgroup_prefix_sum - value;
}

fn sum(value: u32, subgroup_id: u32, subgroup_invocation_id: u32) -> u32 {
    var subgroup_sum = subgroupAdd(value);
    if subgroup_invocation_id == 0u { subgroup_sums[subgroup_id] = subgroup_sum; }
    workgroupBarrier();

    subgroup_sum = select(0u, subgroup_sums[subgroup_invocation_id], subgroup_invocation_id < NUMBER_OF_SUBGROUPS);

    return subgroupAdd(subgroup_sum);
}

fn get_workgroup_index(workgroup_id: vec3u, num_workgroups: vec3u) -> u32 {
    return workgroup_id.y * num_workgroups.x + workgroup_id.x + pc.workgroup_offset;
}

fn zeroing_shared(local_invocation_id_x: u32) {
    if local_invocation_id_x < NUMBER_OF_SUBGROUPS { subgroup_sums[local_invocation_id_x] = 0u; }
    workgroupBarrier();
}

#ifdef COUNT_SEGS_PIPELINE
var<workgroup> shared_number_segs: u32;

@compute @workgroup_size(#NUMBER_OF_THREADS_PER_WORKGROUP, 1, 1)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3u,
    @builtin(num_workgroups) num_workgroups: vec3u,
    @builtin(local_invocation_id) local_invocation_id: vec3u,
    @builtin(subgroup_id) subgroup_id: u32,
    @builtin(subgroup_invocation_id) subgroup_invocation_id: u32,
) {
    shared_number_segs = 0u;
    zeroing_shared(local_invocation_id.x);

    let workgroup_index = get_workgroup_index(workgroup_id, num_workgroups);

    let start_index = workgroup_index * NUMBER_OF_KEYS_PER_WORKGROUP + local_invocation_id.x;
    let close_index = min(start_index + NUMBER_OF_KEYS_PER_WORKGROUP, pc.number_of_keys);
    for (var key_index = start_index; key_index < close_index; key_index += #{NUMBER_OF_THREADS_PER_WORKGROUP}u) {
        let prv_key_index = select(0u, key_index - 1u, key_index > 0u);
        
        let key = global_keys[key_index];
        let prv_key = global_keys[prv_key_index];

        let is_seg_start = key_index == 0u || key != prv_key;

        // note: subgroup operation is not the best performance(atomicAdd is better) here, but the difference is really small.
        let local_number_segs = sum(select(0u, 1u, is_seg_start), subgroup_id, subgroup_invocation_id);
        if local_invocation_id.x == 0u { shared_number_segs += local_number_segs; }
    }

    workgroupBarrier();

    if local_invocation_id.x == 0u { number_segs[workgroup_index] = shared_number_segs; }
}
#endif

#ifdef SCAN_SUMS_PIPELINE
@compute @workgroup_size(#NUMBER_OF_THREADS_PER_WORKGROUP, 1, 1)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3u,
    @builtin(num_workgroups) num_workgroups: vec3u,
    @builtin(local_invocation_id) local_invocation_id: vec3u,
    @builtin(subgroup_id) subgroup_id: u32,
    @builtin(subgroup_invocation_id) subgroup_invocation_id: u32,
) {
    zeroing_shared(local_invocation_id.x);
    
    let workgroup_index = get_workgroup_index(workgroup_id, num_workgroups);

    let index = pc.scan_load_base + workgroup_index * #NUMBER_OF_THREADS_PER_WORKGROUP + local_invocation_id.x;
    if index >= pc.scan_save_base { return; }

    let value = number_segs[index];

    let sum = sum(value, subgroup_id, subgroup_invocation_id);

    if local_invocation_id.x == 0u {
        number_segs[pc.scan_save_base + workgroup_index] = sum;
    }
}
#endif

#ifdef SCAN_LAST_PIPELINE
@compute @workgroup_size(#NUMBER_OF_THREADS_PER_WORKGROUP, 1, 1)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3u,
    @builtin(num_workgroups) num_workgroups: vec3u,
    @builtin(local_invocation_id) local_invocation_id: vec3u,
    @builtin(subgroup_id) subgroup_id: u32,
    @builtin(subgroup_invocation_id) subgroup_invocation_id: u32,
) {
    zeroing_shared(local_invocation_id.x);

    let number_to_scan = pc.scan_save_base - pc.scan_load_base;
    let is_active = local_invocation_id.x < number_to_scan;

    let index = pc.scan_load_base + local_invocation_id.x;
    
    var value = 0u;
    if is_active { value = number_segs[index]; }

    let prefix_sum = scan_exclusive(value, subgroup_id, subgroup_invocation_id);
    
    if is_active { number_segs[index] = prefix_sum; }

    if local_invocation_id.x == #NUMBER_OF_THREADS_PER_WORKGROUP - 1u { tnumber_seg = prefix_sum + value; }
}
#endif

#ifdef SCAN_PRFX_PIPELINE
@compute @workgroup_size(#NUMBER_OF_THREADS_PER_WORKGROUP, 1, 1)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3u,
    @builtin(num_workgroups) num_workgroups: vec3u,
    @builtin(local_invocation_id) local_invocation_id: vec3u,
    @builtin(subgroup_id) subgroup_id: u32,
    @builtin(subgroup_invocation_id) subgroup_invocation_id: u32,
) {
    zeroing_shared(local_invocation_id.x);
    
    let workgroup_index = get_workgroup_index(workgroup_id, num_workgroups);

    let index = pc.scan_load_base + workgroup_index * #NUMBER_OF_THREADS_PER_WORKGROUP + local_invocation_id.x;
    if index >= pc.scan_save_base { return; }

    let value = number_segs[index];

    let sum = number_segs[pc.scan_save_base + workgroup_index];
    let prefix_sum = scan_exclusive(value, subgroup_id, subgroup_invocation_id);

    number_segs[index] = prefix_sum + sum;
}
#endif

#ifdef ASSEMBLE_PIPELINE
var<workgroup> shared_prv_number_segs: u32;

@compute @workgroup_size(#NUMBER_OF_THREADS_PER_WORKGROUP, 1, 1)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3u,
    @builtin(num_workgroups) num_workgroups: vec3u,
    @builtin(local_invocation_id) local_invocation_id: vec3u,
    @builtin(subgroup_id) subgroup_id: u32,
    @builtin(subgroup_invocation_id) subgroup_invocation_id: u32,
) {
    shared_prv_number_segs = 0u;
    zeroing_shared(local_invocation_id.x);

    let workgroup_index = get_workgroup_index(workgroup_id, num_workgroups);

    let global_prv_number_segs = number_segs[workgroup_index];

    let start_index = workgroup_index * NUMBER_OF_KEYS_PER_WORKGROUP + local_invocation_id.x;
    let close_index = min(start_index + NUMBER_OF_KEYS_PER_WORKGROUP, pc.number_of_keys);
    for (var key_index = start_index; key_index < close_index; key_index += #{NUMBER_OF_THREADS_PER_WORKGROUP}u) {
        let prv_key_index = select(0u, key_index - 1u, key_index > 0u);
        
        let key = global_keys[key_index];
        let prv_key = global_keys[prv_key_index];

        let is_seg_start = key_index == 0u || key != prv_key;
        let seg_start = select(0u, 1u, is_seg_start);

        let local_prefix_sum = scan_exclusive(seg_start, subgroup_id, subgroup_invocation_id);

        if is_seg_start {
            let seg_index = global_prv_number_segs + shared_prv_number_segs + local_prefix_sum;

            key_seg_map[key] = seg_index;
            global_segs[seg_index] = key_index;
        }

        workgroupBarrier();

        if local_invocation_id.x == #NUMBER_OF_THREADS_PER_WORKGROUP - 1u {
            shared_prv_number_segs += local_prefix_sum + seg_start;
        }
    }
}
#endif