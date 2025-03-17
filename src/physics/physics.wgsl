/// Use [`VERLET INTEGRATION EQUATION`](https://en.wikipedia.org/wiki/Verlet_integration) to simulate simple physics effect!
const NUMBER_OF_GRIDS_ONE_DIMENSION: u32 = #NUMBER_OF_GRIDS_ONE_DIMENSION;
const GRID_SIZE: u32 = #GRID_SIZE;
const HALF_MAP_HEIGHT: f32 = f32(#HALF_MAP_HEIGHT);
const HALF_MAP_SIZE: f32 = f32(#HALF_MAP_SIZE);

const GRAVITY: vec3f = vec3f(f32(#GRAVITY_X), f32(#GRAVITY_Y), f32(#GRAVITY_Z)) / 1000.;

const E_ENVIR: f32 = f32(#E_ENVIR) / 1000.;
const E_AGENT: f32 = f32(#E_AGENT) / 1000.;

const K_AGENT: f32 = f32(#K_AGENT) / 1000.;
const C_AGENT: f32 = f32(#C_AGENT) / 1000.;
const R_AGENT: f32 = f32(#R_AGENT) / 1000.;

const T_DELTA: f32 = f32(#TIME_DELTA) / 1000.;

const INVALID: u32 = 0xFFFFFFFFu;
const CLAMP_F16: vec3f = vec3f(65504.);

@group(0) @binding(0) var<storage, read      > agent_index_sorted: array<u32>;
@group(0) @binding(1) var<storage, read      > collision_grid_start_indexs: array<u32>;
@group(0) @binding(2) var<storage, read      > collision_grid_close_indexs: array<u32>;
@group(0) @binding(3) var<storage, read_write> collision_grid_index_unsorted: array<u32>;
@group(0) @binding(4) var<storage, read      > position_with_scale_i: array<vec4f>;
@group(0) @binding(5) var<storage, read      > packed_vfme_i: array<vec4u>;
@group(0) @binding(6) var<storage, read_write> position_with_scale_o: array<vec4f>;
@group(0) @binding(7) var<storage, read_write> packed_vfme_o: array<vec4u>;

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
    /// The number of agents to be simulated.
    number_of_agents: u32,
}
var<push_constant> pc: PushConstants;

var<private> agent: Agent;
var<private> new_agent_index: u32;

@compute @workgroup_size(#NUMBER_OF_THREADS_PER_WORKGROUP, 1, 1)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3u,
    @builtin(num_workgroups) num_workgroups: vec3u,
    @builtin(local_invocation_id) local_invocation_id: vec3u
) {
    let workgroup_index = get_workgroup_index(workgroup_id, num_workgroups);

    new_agent_index = get_global_thread_index(workgroup_index, local_invocation_id.x);
    if new_agent_index >= pc.number_of_agents { return; }

    let old_agent_index = agent_index_sorted[new_agent_index];

    agent = agent_new(old_agent_index);

    let collision_grid_id = get_collision_grid_id(agent.position.xz);
    let collision_grid_index = get_collision_grid_index(collision_grid_id);

#ifdef LOOP_PIPELINE
    verlet_integration();
    // the force used in `verlet_integration` comes from last frame, so it must be cleared before the next frame.
    agent.force = vec3f(0.);
    handle_collisions(collision_grid_id, collision_grid_index);
    solve_cube_constraint();

    agent.velocity = clamp(agent.velocity, -CLAMP_F16, CLAMP_F16);
    agent.force = clamp(agent.force, -CLAMP_F16, CLAMP_F16);

    position_with_scale_o[new_agent_index] = vec4f(agent.position, agent.scale);
    packed_vfme_o[new_agent_index] = pack_vfme(agent.velocity, agent.force, agent.massv);

    collision_grid_index_unsorted[new_agent_index] = get_collision_grid_index_direct(agent.position.xz);
#endif

#ifdef INIT_PIPELINE
    position_with_scale_o[new_agent_index] = vec4f(agent.position, agent.scale);
    packed_vfme_o[new_agent_index] = pack_vfme(agent.velocity, agent.force, agent.massv);

    collision_grid_index_unsorted[new_agent_index] = collision_grid_index;
#endif
}

fn get_workgroup_index(workgroup_id: vec3u, num_workgroups: vec3u) -> u32 {
    return workgroup_id.y * num_workgroups.x + workgroup_id.x + pc.workgroup_offset;
}

fn get_global_thread_index(workgroup_index: u32, local_invocation_id_x: u32) -> u32 {
    return workgroup_index * #NUMBER_OF_THREADS_PER_WORKGROUP + local_invocation_id_x;
}

fn get_collision_grid_id(position: vec2f) -> vec2u {
    return vec2u(position + vec2f(HALF_MAP_SIZE)) / GRID_SIZE;
}

fn get_collision_grid_index(grid_id: vec2u) -> u32 {
    return grid_id.y * NUMBER_OF_GRIDS_ONE_DIMENSION + grid_id.x;
}

fn get_collision_grid_index_direct(position: vec2f) -> u32 {
    return get_collision_grid_index(get_collision_grid_id(position));
}

fn handle_collisions(collision_grid_id: vec2u, collision_grid_index: u32) {
    let old_position = agent.position;
    
    // middle
    {
        // self collision grid(must exist)
        var collision_grid_start_index = collision_grid_start_indexs[collision_grid_index];
        var collision_grid_close_index = collision_grid_close_indexs[collision_grid_index];

        // left collision grid
        if collision_grid_id.x > 0u {
            let start_index = collision_grid_start_indexs[collision_grid_index - 1u];
            if start_index != INVALID { collision_grid_start_index = start_index; }
        }

        // right collision grid
        if collision_grid_id.x < NUMBER_OF_GRIDS_ONE_DIMENSION - 1u {
            let start_index = collision_grid_start_indexs[collision_grid_index + 1u];
            if start_index != INVALID { collision_grid_close_index = collision_grid_close_indexs[collision_grid_index + 1u]; }
        }

        // collision detection calculation
        for (var i = collision_grid_start_index; i <= collision_grid_close_index; i++) {
            if (i == new_agent_index) { continue; }
            let anr_old_agent_index = agent_index_sorted[i];
            let anr_agent = agent_new(anr_old_agent_index);

            solve_sphere_constraint(anr_agent);
        }
    }

    // bottom
    if collision_grid_id.y > 0u {
        let bbase_collision_grid_index = collision_grid_index - NUMBER_OF_GRIDS_ONE_DIMENSION;
        handle_collision(collision_grid_id, bbase_collision_grid_index);
    }

    // top
    if collision_grid_id.y < NUMBER_OF_GRIDS_ONE_DIMENSION - 1u {
        let tbase_collision_grid_index = collision_grid_index + NUMBER_OF_GRIDS_ONE_DIMENSION;
        handle_collision(collision_grid_id, tbase_collision_grid_index);
    }

    // the position difference between before and after collision detection
    let diff = agent.position - old_position;
    // avoid the agent getting stuck with another agents
    if (length(diff) > agent.scale) {
        let jitter = hash3f1u(new_agent_index) * 2. - 1.;
        agent.position += normalize(jitter);
    }
}

fn handle_collision(collision_grid_id: vec2u, base_collision_grid_index: u32) {
    var collision_grid_start_index = INVALID;
    var collision_grid_close_index = 0u;

    if collision_grid_id.x > 0u {
        let start_index = collision_grid_start_indexs[base_collision_grid_index - 1u];
        if start_index != INVALID {
            collision_grid_start_index = start_index;
            collision_grid_close_index = collision_grid_close_indexs[base_collision_grid_index - 1u];
        }
    }

    {
        let start_index = collision_grid_start_indexs[base_collision_grid_index];
        if start_index != INVALID {
            if (collision_grid_start_index == INVALID) { collision_grid_start_index = start_index; }
            collision_grid_close_index = collision_grid_close_indexs[base_collision_grid_index];
        }
    }

    if collision_grid_id.x < NUMBER_OF_GRIDS_ONE_DIMENSION - 1u {
        let start_index = collision_grid_start_indexs[base_collision_grid_index + 1u];
        if start_index != INVALID {
            if (collision_grid_start_index == INVALID) { collision_grid_start_index = start_index; }
            collision_grid_close_index = collision_grid_close_indexs[base_collision_grid_index + 1u];
        }
    }

    // collision detection calculation
    for (var i = collision_grid_start_index; i <= collision_grid_close_index; i++) {
        let anr_old_agent_index = agent_index_sorted[i];
        let anr_agent = agent_new(anr_old_agent_index);

        solve_sphere_constraint(anr_agent);
    }
}

struct Agent {
    position: vec3f,
    scale: f32,
    // the position difference between current frame and last frame
    velocity: vec3f,
    massv: f32,
    force: vec3f,
}

fn agent_new(old_agent_index: u32) -> Agent {
    let position_with_scale = position_with_scale_i[old_agent_index];
    
    // unpack
    var velocity: vec3f;
    var extforce: vec3f;
    var massive: f32;
    unpack_vfme(packed_vfme_i[old_agent_index], &velocity, &extforce, &massive);

    return Agent(
        position_with_scale.xyz,
        position_with_scale.w,
        velocity,
        massive,
        extforce,
    );
}

fn pack_vfme(velocity: vec3f, force: vec3f, massv: f32) -> vec4u {
    let vxy = pack2x16float(velocity.xy);
    let vzfx = pack2x16float(vec2f(velocity.z, force.x));
    let fyz = pack2x16float(force.yz);
    let m_e = pack2x16float(vec2f(massv, 0.));

    return vec4u(vxy, vzfx, fyz, m_e);
}

fn unpack_vfme(packed: vec4u, velocity: ptr<function, vec3f>, extforce: ptr<function, vec3f>, massive: ptr<function, f32>) {
    let vxy = unpack2x16float(packed.x);
    let vzfx = unpack2x16float(packed.y);
    let fyz = unpack2x16float(packed.z);
    let m_e = unpack2x16float(packed.w);

    *velocity = vec3f(vxy, vzfx.x);
    *extforce = vec3f(vzfx.y, fyz);
    *massive = m_e.x;
}

/// Verlet-Integration Equation: $x_{t+1} = 2 x_t - x_{t-1} + a {\Delta t}^2$
///
/// In normal circumstances, $x_t$ and $x_{t-1}$ are used to represent the state of motion, 
/// but this form is not friendly for rigid body collision calculations (damping, rebound, etc.),
/// so the `verlet-integration` is slightly modified in the code: $x_{t+1} = x_t - v_t + a {\Delta t}^2$.
///
/// ($v_t = x_t - x_{t-1}$)
fn verlet_integration() {
    let acc = agent.force / agent.massv;
    agent.velocity += (GRAVITY + acc) * T_DELTA * T_DELTA;
    agent.position += agent.velocity;
}

/// The length and width and height of this Box are `pc.number_of_grid_one_dimension * GRID_SIZE`.
fn solve_cube_constraint() {
    let edge = vec3f(HALF_MAP_SIZE, HALF_MAP_HEIGHT, HALF_MAP_SIZE) - agent.scale;
    let cetr = vec3f(0., HALF_MAP_HEIGHT, 0.);
    let tpos = agent.position - cetr;

    let makeup = edge - abs(tpos);
    let itouch = makeup < vec3f(0.);
    if(any(itouch)) {
#if ENABLE_SPRING == false
        agent.position = clamp(tpos, -edge, edge) + cetr;
        // `sign` is not good because it will return 0 when the input is 0, which will cause the velocity to be 0.
        agent.velocity *= select(vec3f(1.), -vec3f(E_ENVIR), itouch);
#else
        let min_makeup = -vec3f(agent.scale * R_AGENT);
        // the spring force is calculated by Hooke's Law
        let kforce = K_AGENT * sign(tpos) * clamp(makeup, min_makeup, vec3f(0.));
        let cforce = -C_AGENT / T_DELTA * agent.velocity;
        agent.force += kforce + cforce;

        let itouch = makeup < min_makeup;
        if(any(itouch)) {
            let edge = edge - min_makeup;
            agent.position = clamp(tpos, -edge, edge) + cetr;
            let dv = -(1. + E_ENVIR) / T_DELTA * select(vec3f(0.), agent.velocity, itouch);
            let acc = dv / T_DELTA;
            agent.force += agent.massv * acc;
        }
#endif
    }
}

fn solve_sphere_constraint(anr_agent: Agent) {
    let edge = anr_agent.scale + agent.scale;
    let tpos = agent.position - anr_agent.position;
    let dist = length(tpos);

    if dist <= edge {
        let norm = normalize(tpos);

#if ENABLE_SPRING == false
        let m1 = agent.massv;
        let m2 = anr_agent.massv;

        let rvel = agent.velocity - anr_agent.velocity;
        let rspd = dot(rvel, norm);
        if rspd < 0. {
            let rvel_n = rspd * norm;
            let rvel_c = (m1 - E_AGENT * m2) * rvel_n / (m1 + m2);
            agent.velocity = anr_agent.velocity + (rvel - rvel_n) + rvel_c;
        }
        agent.position += 0.5 * (edge - dist) * norm;
#else
        let makeup = edge - dist;

        let kforce = K_AGENT * makeup * norm;
        let cforce = -C_AGENT / T_DELTA * agent.velocity;
        agent.force += kforce + cforce;

        let limit = makeup - R_AGENT * agent.scale;
        if limit > 0. { agent.position += limit * norm; }
#endif
    }
}

/// The algorithm comes from [James_Harnett](https://www.shadertoy.com/view/4dVBzz)
fn hash3f1u(n: u32) -> vec3f {
    let v = n * (n ^ (n >> 15u));
    return vec3f(v * vec3u(0x1u, 0x1ffu, 0x3ffffu)) / f32(0xFFFFFFFFu);
}

// fn loop_in_range(x: u32, a: u32, n: u32) -> u32 {
//     return x - (x - a) / n * n;
// }