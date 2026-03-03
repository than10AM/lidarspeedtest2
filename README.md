# Lidar Raycast Benchmark

This project benchmarks raycasting performance using NVIDIA's Warp framework, simulating lidar sweeps over a generated 3D scene containing thousands of randomly placed cubes.

## Why

This project is a minimal, focused environment to test the raw performance capability of GPU-accelerated raycasting using [NVIDIA Warp](https://github.com/NVIDIA/warp). It is particularly useful for assessing whether Warp handles high-volume geometric queries (e.g. simulating a fast-spinning dense Lidar) efficiently.

Specifically, it measures the time taken to:
1. Fire a dense array of rays (2048x128 resolution, representing one full 360-degree sweep of a 128-channel lidar).
2. Perform ray-mesh intersections against a scene with randomly generated cubes.
3. Compute math operations on large arrays to simulate overhead or additional data processing.
4. Read the collision data back from GPU to CPU memory and pack it optimally into byte formats.

This yields a "Rays per Second" metric that includes memory transfer overhead, offering a realistic view of real-time pointcloud generation performance.

## Prerequisites

- Python 3.8+
- An NVIDIA GPU with CUDA Toolkit installed
- Appropriate GPU Drivers

## Running the benchmark

The easiest way to run the benchmark is to use the included `run.sh` script.

This script will automatically:
1. Create a Python virtual environment (`venv`).
2. Activate the virtual environment.
3. Install the required dependencies from `requirements.txt`.
4. Execute the benchmarking script.

```bash
./run.sh
```

Alternatively, you can run the steps manually:

```bash
# Create the virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install the requirements
pip install -r requirements.txt

# Run the benchmark script
python warp_raycast_test.py
```
