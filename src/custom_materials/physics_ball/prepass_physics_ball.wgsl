#import bevy_pbr::{
    prepass_io::Vertex,
    mesh_view_bindings as view_bindings,
    mesh_bindings,
    view_transformations,
    pbr_deferred_functions,
    pbr_functions,
    pbr_types,
}

#import custom_shader::sdf
// #import bevy_wgsl_utils::pcg_hash as hash

@group(2) @binding(0) var<storage, read> collision_grid_indexs: array<u32>;
@group(2) @binding(1) var<storage, read> collision_grid_start_indexs: array<u32>;
@group(2) @binding(2) var<storage, read> collision_grid_close_indexs: array<u32>;
@group(2) @binding(3) var<storage, read> position_with_scale_i: array<vec4f>;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) world_position: vec4f,
    // @location(1) @interpolate(flat) agent_index: u32,
    @location(1) @interpolate(flat) position_with_scale: vec4f,
    // @location(2) @interpolate(flat) collision_grid_index: u32,
}

fn temporary_solution(sphere: vec4f) -> vec4f {
    let x = clamp(sphere.y / sphere.w, 0.0, 1.0);
    let y = 0.26 * (x - 1.0) * (x - 1.0) + 1.0;

    return vec4f(sphere.xyz, sphere.w * y);
}

@vertex
fn vertex(in: Vertex) -> VertexOutput {
    var position_with_scale = position_with_scale_i[in.instance_index];
    position_with_scale = temporary_solution(position_with_scale);
    
    let world_position = in.position * position_with_scale.w + position_with_scale.xyz;
    let clip_position = view_transformations::position_world_to_clip(world_position);

    // let collision_grid_index = collision_grid_indexs[in.instance_index];

    return VertexOutput(clip_position, vec4f(world_position, 1.0), position_with_scale);
}

#ifdef PREPASS_FRAGMENT
struct FragmentOutput {
#ifdef NORMAL_PREPASS
    @location(0) normal: vec4<f32>,
#endif

    @builtin(frag_depth) frag_depth: f32,
}
#endif //PREPASS_FRAGMENT

const K: f32 = 0.08;

@fragment
fn fragment(in: VertexOutput) -> FragmentOutput {
    let is_orthographic = view_bindings::view.clip_from_view[3].w == 1.0;
    let view_direction = pbr_functions::calculate_view(in.world_position, is_orthographic);

    let ray_origin = in.world_position.xyz;
    let ray_direction = view_direction;

    let sphere = in.position_with_scale;
    
    let info = sdf::ray_x_sphere(ray_origin, ray_direction, sphere);
    if info.d2raydir > sphere.w { discard; }

    var world_position_hit = ray_origin - info.d2sphere * ray_direction;

    let plane = vec4f(0.0, -1.0, 0.0, 0.0);

    var sdf_sphere = 0.0;
    var sdf_plane = sdf::sdf_plane(world_position_hit, plane);

    // NOTE: 可以直接用球体近似(去掉这些代码), 视觉上区别不大
    for (var i = 0u; i < 16u; i++) {
        let sdfv = sdf::smooth_max(sdf_sphere, sdf_plane, K);
        world_position_hit -= sdfv * ray_direction;

        if sdfv < 0.001 { break; }
        if i == 15u { discard; }

        sdf_sphere = sdf::sdf_sphere(world_position_hit, sphere);
        sdf_plane = sdf::sdf_plane(world_position_hit, plane);
    }

    var out: FragmentOutput;

#ifdef NORMAL_PREPASS
    let world_normal_sphere = sdf::normal_sphere(world_position_hit, sphere);
    let world_normal = sdf::gradient_smooth_max(sdf_sphere, sdf_plane, K, world_normal_sphere, plane.xyz);

    out.normal = vec4(world_normal * 0.5 + vec3(0.5), 1.0);
#endif

    // TODO: 这里有更简单的方式, 可以根据info.d2sphere来直接计算深度
    let clip_position = view_transformations::position_world_to_clip(world_position_hit);
    let frag_coord = clip_position / clip_position.w;

    out.frag_depth = frag_coord.z;

    return out;
}