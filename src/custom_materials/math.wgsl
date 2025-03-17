#define_import_path custom_shader::math

const X_RIGHT: vec3f = vec3f(1.0, 0.0, 0.0);
const Y_UP: vec3f = vec3f(0.0, 1.0, 0.0);
const Z_FORWARD: vec3f = vec3f(0.0, 0.0, 1.0);

fn modulo(a: f32, b: f32) -> f32 {
    return a - b * floor(a / b);
}

fn modulo2(a: vec2f, b: vec2f) -> vec2f {
    return a - b * floor(a / b);
}

fn ping_pong(x: f32, max: f32) -> f32 {
    return max - abs(modulo(x, 2.0 * max) - max);
}

fn ping_pong2(x: vec2f, max: vec2f) -> vec2f {
    return max - abs(modulo2(x, 2.0 * max) - max);
}