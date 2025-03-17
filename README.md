# Bevy Millions Ball

![Bevy Millions Ball](https://img.shields.io/badge/Bevy-0.15.3-blue)
![License](https://img.shields.io/badge/license-MIT-green)

![bevy_millions_ball_showcase](./assets/videos/bevy_millions_ball_showcase.gif)

A high-performance collision detection system built with the Bevy game engine, capable of simulating **millions of spheres** in real-time.

## Features

- **Massive Scale**: Efficiently handles over 1,000,000 sphere collisions simultaneously
- **GPU-Accelerated**: Leverages compute shaders for parallel collision detection and resolution
- **Spatial Hashing**: Implements uniform grid partitioning for O(n) collision detection
- **High Performance**: Optimized algorithms for real-time physics simulation
- **Configurable**: Adjustable physics parameters through an intuitive UI
- **Extensible**: Designed to support other primitive shapes beyond spheres

## Technical Implementation

This project demonstrates advanced techniques in real-time physics simulation:

- **Uniform Grid Partitioning**: Divides the 3D space into a grid for efficient spatial queries
- **Parallel Computing**: Utilizes GPU compute shaders for massive parallelization
- **[Radix Sort](https://github.com/AllenPocketGamer/bevy_radix_sort)**: Implements efficient sorting for spatial hashing
- **Double Buffering**: Uses ping-pong buffers for efficient data processing
- **Spring Physics**: Optional spring-based collision response for more realistic simulations

## Performance Optimizations

- **WGSL Compute Shaders**: Custom shaders for maximum GPU utilization
- **Memory Efficiency**: Optimized data structures and memory layout
- **Workgroup Optimization**: Tuned workgroup sizes for optimal GPU performance
- **Minimal CPU Overhead**: Physics calculations performed entirely on the GPU

## Requirements

- Rust 2021 edition or later
- GPU with compute shader support
- Bevy 0.15.3 compatible system

## Usage

```bash
# Clone the repository
git clone https://github.com/AllenPocketGamer/bevy_millions_ball.git
cd bevy_millions_ball

# Run the example
cargo run --release
```

## Controls

- **Camera**: Pan, orbit, and zoom with mouse controls
- **Physics Settings**: Adjust simulation parameters through the UI panel
- **Simulation**: Toggle simulation on/off and adjust time scale

## Configuration

The physics system is highly configurable through the UI:

- **Grid Size**: Adjust the spatial partitioning resolution
- **Restitution**: Configure elasticity of collisions
- **Spring Physics**: Enable/disable and tune spring parameters
- **Gravity**: Modify the gravity vector
- **Time Step**: Adjust simulation speed

## Performance Testing

To test the performance limits of your system, you can modify the number of spheres in the simulation:

1. Open `src/main.rs` in your code editor
2. Locate the `PhysicsPlugin` configuration in the `main()` function:

```rust
.add_plugins(PhysicsPlugin {
    max_number_of_agents: 1024 * 1024, // 1 million spheres
    number_of_grids_one_dimension: 1024,
    half_map_height: 8,
    e_agent: 0.8,
    e_envir: 0.8,
    ..default()
})
```

3. Adjust the `max_number_of_agents` parameter to increase or decrease the number of spheres:
   - For lower-end systems, try `256 * 256` (65,536 spheres)
   - For mid-range systems, try `512 * 512` (262,144 spheres)
   - For high-end systems, try `2048 * 2048` (4 million spheres)

4. You may also need to adjust `number_of_grids_one_dimension` to maintain performance as you change the number of agents

Note: Performance will vary based on your GPU capabilities. Monitor your frame rate and adjust accordingly.

## How It Works

The collision detection system uses a spatial hashing approach with uniform grid partitioning:

1. **Spatial Hashing**: Each sphere is assigned to grid cells based on its position
2. **Parallel Processing**: Grid cells are processed in parallel on the GPU
3. **Collision Detection**: Only spheres in the same or neighboring cells are checked for collisions
4. **Physics Resolution**: Collisions are resolved using either rigid body or spring physics

This approach reduces the complexity from O(n²) to O(n), making it possible to simulate millions of objects in real-time.

## Author

[AllenPocketGamer](https://github.com/AllenPocketGamer)

---

*This project was created as a demonstration of high-performance physics simulation capabilities using the Bevy game engine.*
