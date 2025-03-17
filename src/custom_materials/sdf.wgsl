#define_import_path custom_shader::sdf

// -------------------------------
// SMOOTH BLEND FUNCTION ---------
// -------------------------------

fn smooth_min(a: f32, b: f32, k: f32) -> f32 {
    return smooth_max(a, b, -k);
}

fn smooth_max(a: f32, b: f32, k: f32) -> f32 {
    let k_inv = 1. / k;
    return k * log2(exp2(a * k_inv) + exp2(b * k_inv));
}

fn gradient_smooth_min(a: f32, b: f32, k: f32, na: vec3f, nb: vec3f) -> vec3f {
    return gradient_smooth_max(a, b, -k, na, nb);
}

fn gradient_smooth_max(a: f32, b: f32, k: f32, na: vec3f, nb: vec3f) -> vec3f {
    let h = (b - a) / k;

    let partial_a = 1. / (1. + exp2( h));
    let partial_b = 1. / (1. + exp2(-h));
    
    return partial_a * na + partial_b * nb;
}

fn smooth_min_3(a: f32, b: f32, c: f32, k: f32) -> f32 {
    return smooth_max_3(a, b, c, -k);
}

fn smooth_max_3(a: f32, b: f32, c: f32, k: f32) -> f32 {
    let k_inv = 1. / k;
    return k * log2(exp2(a * k_inv) + exp2(b * k_inv) + exp2(c * k_inv));
}

fn gradient_smooth_min_3(a: f32, b: f32, c: f32, k:f32, na: vec3f, nb: vec3f, nc: vec3f) -> vec3f {
    return gradient_smooth_max_3(a, b, c, -k, na, nb, nc);
}

fn gradient_smooth_max_3(a: f32, b: f32, c: f32, k:f32, na: vec3f, nb: vec3f, nc: vec3f) -> vec3f {
    let k_inv = 1. / k;
    
    let a_b = (a - b) * k_inv;
    let b_c = (b - c) * k_inv;
    let c_a = (c - a) * k_inv;

    let partial_a = 1. / (1. + exp2(-a_b) + exp2(c_a));
    let partial_b = 1. / (1. + exp2(a_b) + exp2(-b_c));
    let partial_c = 1. / (1. + exp2(-c_a) + exp2(b_c));
    
    return partial_a * na + partial_b * nb + partial_c * nc;
}

fn smooth_min_4(a: f32, b: f32, c: f32, d: f32, k: f32) -> f32 {
    return smooth_max_4(a, b, c, d, -k);
}

fn smooth_max_4(a: f32, b: f32, c: f32, d: f32, k: f32) -> f32 {
    let k_inv = 1. / k;
    return k * log2(exp2(a * k_inv) + exp2(b * k_inv) + exp2(c * k_inv) + exp2(d * k_inv));
}

fn gradient_smooth_min_4(a: f32, b: f32, c: f32, d: f32, k: f32, na: vec3f, nb: vec3f, nc: vec3f, nd: vec3f) -> vec3f {
    return gradient_smooth_max_4(a, b, c, d, -k, na, nb, nc, nd);
}

fn gradient_smooth_max_4(a: f32, b: f32, c: f32, d: f32, k: f32, na: vec3f, nb: vec3f, nc: vec3f, nd: vec3f) -> vec3f {
    let k_inv = 1. / k;

    let a_b = (a - b) * k_inv;
    let a_c = (a - c) * k_inv;
    let a_d = (a - d) * k_inv;
    
    let b_c = (b - c) * k_inv;
    let b_d = (b - d) * k_inv;

    let c_d = (c - d) * k_inv;

    let partial_a = 1. / (1. + exp2(-a_b) + exp2(-a_c) + exp2(-a_d));
    let partial_b = 1. / (1. + exp2(a_b) + exp2(-b_c) + exp2(-b_d));
    let partial_c = 1. / (1. + exp2(a_c) + exp2(b_c) + exp2(-c_d));
    let partial_d = 1. / (1. + exp2(a_d) + exp2(b_d) + exp2(c_d));

    return partial_a * na + partial_b * nb + partial_c * nc + partial_d * nd;
}

// ---------------------
// 3D Geometry ---------
// ---------------------

fn sdf_sphere(position: vec3f, sphere: vec4f) -> f32 {
    return length(position - sphere.xyz) - sphere.w;
}

/// Signed distance function for a plane
/// @param p - Point to evaluate SDF at
/// @param plane - vec4f where:
///   - plane.xyz: The normal vector of the plane (must be normalized)
///   - plane.w: Signed distance from origin to plane along the normal
/// @returns Signed distance from p to the plane (positive if p is on normal side)
fn sdf_plane(position: vec3f, plane: vec4f) -> f32 {
    return dot(position, plane.xyz) - plane.w;
}

// ----------------------------
// 3D Geometry Normal ---------
// ----------------------------

fn normal_sphere(position: vec3f, sphere: vec4f) -> vec3f {
    return normalize(position - sphere.xyz);
}

fn normal_plane(plane: vec4f) -> vec3f {
    return normalize(plane.xyz);
}

// ------------------------------------
// Ray Intersection Functions ---------
// ------------------------------------

struct BoxHitInfo {
    tmin: f32,
    tmax: f32,
}

fn ray_x_box(ray_origin: vec3f, ray_direction: vec3f, box_centre: vec3f, half_size: vec3f) -> BoxHitInfo {
    let box_ray_origin = ray_origin - box_centre;
    let t1 = (-half_size - box_ray_origin) / ray_direction;
    let t2 = (half_size - box_ray_origin) / ray_direction;

    // Get the entry point (tmin) and exit point (tmax) for each pair of planes
    let tmin = min(t1, t2);
    let tmax = max(t1, t2);

    // Get the nearest entry point（t_min) and the nearest exit point(t_max)
    let t_min = max(tmin.x, max(tmin.y, tmin.z));
    let t_max = min(tmax.x, min(tmax.y, tmax.z));

    // if result < 0., the ray is not outside the box or not hit the box.
    return BoxHitInfo(t_min, t_max);
}

struct SphereHitInfo {
    d2sphere: f32,  // the distance from ray_origin to the sphere
    d2raydir: f32,  // the distance from sphere center to ray direction
}

/// If `ray_origin` is in sphere or ray does not hit sphere, return false.
///
/// The ray_direction is towards to the eye(ray_origin).
fn ray_x_sphere(ray_origin: vec3f, ray_direction: vec3f, sphere: vec4f) -> SphereHitInfo {
    let position = ray_origin - sphere.xyz;
    let radius = sphere.w;
    
    let t = dot(position, ray_direction);
    let b2 = dot(position, position) - t * t;   // the power of distance from sphere center to ray
    let c2 = radius * radius - b2;
    let d = t - sqrt(c2);                       // the distance from ray_origin to the sphere

    return SphereHitInfo(d, sqrt(b2));
}

// ---------------
// Utils ---------
// ---------------

fn get_intersection_plane(sphere_a: vec4f, sphere_b: vec4f) -> vec4f {
    let d = sphere_b.xyz - sphere_a.xyz;
    let w = 0.5 * (
        dot(sphere_b.xyz, sphere_b.xyz) - sphere_b.w * sphere_b.w -
        dot(sphere_a.xyz, sphere_a.xyz) + sphere_a.w * sphere_a.w
    );

    return vec4f(d, w);
    // return vec4f(d, w) / length(d);
}

fn facing(n_dot_v: f32, blend: f32) -> f32 {
    return pow(1. - n_dot_v, blend);
}