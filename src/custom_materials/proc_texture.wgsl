#define_import_path custom_shader::proc_texture

#import custom_shader::math::{Y_UP, Z_FORWARD}

fn transform_position_to_uv(position: vec3f, normal: vec3f) -> vec2f {
    let center = dot(position, normal) * normal;
    let vector = position - center;

    var v_axis = cross(Y_UP, normal);
    v_axis = select(Z_FORWARD, v_axis, length(v_axis) > 0.01);
    let u_axis = cross(normal, v_axis);

    let uv = vec2f(dot(vector, u_axis), dot(vector, v_axis));

    return uv;
}

fn checkerboard(uv: vec2f, size: vec2f, smoothness: f32) -> f32 {
    let uv0 = uv / size;

    let ww = fwidthFine(uv0) * smoothness;

    let dp0 = 0.5 * (uv0 + 0.5 * ww);
    let dp1 = 0.5 * (uv0 - 0.5 * ww);

    let db0 = floor(dp0) + 2. * max(dp0 - floor(dp0) - 0.5, vec2f(0.));
    let db1 = floor(dp1) + 2. * max(dp1 - floor(dp1) - 0.5, vec2f(0.));

    let i = (db0 - db1) / ww;

    return i.x * i.y + (1. - i.x) * (1. - i.y);
}

fn reference_grid(uv: vec2f, size: vec2f, thickness: vec2f, smoothness: f32) -> f32 {
    let t = thickness / size;
    let h = 0.5 * t;
    
    let uv0 = uv / size + h;

    let ww = fwidthFine(uv0) * smoothness;

    let dp0 = uv0 + 0.5 * ww;
    let dp1 = uv0 - 0.5 * ww;

    let db0 = t * floor(dp0) - h + min(dp0 - floor(dp0), t);
    let db1 = t * floor(dp1) - h + min(dp1 - floor(dp1), t);

    let i = (db0 - db1) / ww;

    return max(i.x, i.y);
}