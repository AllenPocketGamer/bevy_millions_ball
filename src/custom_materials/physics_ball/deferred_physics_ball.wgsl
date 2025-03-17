#import bevy_pbr::{
    prepass_io::Vertex,
    mesh_view_bindings as view_bindings,
    mesh_bindings,
    mesh_types,
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

#ifdef DEFERRED_PREPASS
    @location(2) deferred: vec4<u32>,
    @location(3) deferred_lighting_pass_id: u32,
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
    let world_normal_sphere = sdf::normal_sphere(world_position_hit, sphere);

    let plane = vec4f(0.0, -1.0, 0.0, 0.0);

    var sdf_sphere = 0.0;
    var sdf_plane = sdf::sdf_plane(world_position_hit, plane);

    for (var i = 0u; i < 16u; i++) {
        let sdfv = sdf::smooth_max(sdf_sphere, sdf_plane, K);
        world_position_hit -= sdfv * ray_direction;

        if sdfv < 0.001 { break; }
        if i == 15u { discard; }

        sdf_sphere = sdf::sdf_sphere(world_position_hit, sphere);
        sdf_plane = sdf::sdf_plane(world_position_hit, plane);
    }

    let world_normal = sdf::gradient_smooth_max(sdf_sphere, sdf_plane, K, world_normal_sphere, plane.xyz);

    // TODO: 这里有更简单的方式, 可以根据info.d2sphere来直接计算深度
    let clip_position = view_transformations::position_world_to_clip(world_position_hit);
    let frag_coord = clip_position / clip_position.w;

    let pbr_input = custom_pbr_input(
        is_orthographic,
        view_direction,
        frag_coord,
        vec4f(world_position_hit, 1.0),
        world_normal,
        vec4f(255.0, 191.0, 0.0, 255.0) / 255.0,
    );

    return deferred_output(world_normal, frag_coord.z, pbr_input);
}

fn custom_pbr_input(
    is_orthographic: bool,
    view_direction: vec3f,
    frag_coord: vec4f,
    world_position: vec4f,
    world_normal: vec3f,
    base_color: vec4f,
) -> pbr_types::PbrInput {
    var pbr_input = pbr_types::pbr_input_new();

    pbr_input.is_orthographic = is_orthographic;
    pbr_input.V = view_direction;

    pbr_input.frag_coord = frag_coord;
    pbr_input.world_position = world_position;
    pbr_input.world_normal = world_normal;
    pbr_input.N = pbr_input.world_normal;

    pbr_input.material.base_color = base_color;
    pbr_input.material.base_color = pbr_functions::alpha_discard(pbr_input.material, pbr_input.material.base_color);
    pbr_input.material.perceptual_roughness = 0.435;
    pbr_input.material.metallic = 0.00;

    pbr_input.flags |= mesh_types::MESH_FLAGS_SHADOW_RECEIVER_BIT;
    // pbr_input.material.flags |= pbr_types::STANDARD_MATERIAL_FLAGS_UNLIT_BIT;

    // pbr_input.material.specular_transmission = 0.8;
    // pbr_input.material.diffuse_transmission = 0.5;
    // pbr_input.material.reflectance = 0.25;
    // pbr_input.material.thickness = 1.25;

    return pbr_input;
}

fn deferred_output(
    world_normal: vec3f,
    frag_depth: f32,
    pbr_input: pbr_types::PbrInput
) -> FragmentOutput {
    var out: FragmentOutput;

#ifdef NORMAL_PREPASS
    out.normal = vec4f(world_normal * 0.5 + vec3f(0.5), 1.0);
#endif

#ifdef DEFERRED_PREPASS
    out.deferred = pbr_deferred_functions::deferred_gbuffer_from_pbr_input(pbr_input);
    out.deferred_lighting_pass_id = 1u;
#endif

    out.frag_depth = frag_depth;

    return out;
}