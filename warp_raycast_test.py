import warp as wp
import numpy as np
import time

wp.init()

def create_cubes(num_cubes=500):
    # A single unit cube centered at origin
    base_verts = np.array([
        [-0.5, -0.5, -0.5],
        [ 0.5, -0.5, -0.5],
        [ 0.5,  0.5, -0.5],
        [-0.5,  0.5, -0.5],
        [-0.5, -0.5,  0.5],
        [ 0.5, -0.5,  0.5],
        [ 0.5,  0.5,  0.5],
        [-0.5,  0.5,  0.5]
    ], dtype=np.float32)
    
    # 12 triangles (2 per face)
    base_indices = np.array([
        0, 1, 2,  0, 2, 3,  # Front
        1, 5, 6,  1, 6, 2,  # Right
        5, 4, 7,  5, 7, 6,  # Back
        4, 0, 3,  4, 3, 7,  # Left
        3, 2, 6,  3, 6, 7,  # Top
        4, 5, 1,  4, 1, 0   # Bottom
    ], dtype=np.int32)
    
    all_verts = []
    all_indices = []
    
    np.random.seed(42)
    for i in range(num_cubes):
        scale = np.random.uniform(0.5, 3.0, size=(1, 3))
        # Random position in a -50 to 50 box
        pos = np.random.uniform(-50.0, 50.0, size=(1, 3))
        
        # Keep cubes away from origin where the camera is
        while np.linalg.norm(pos) < 5.0:
            pos = np.random.uniform(-50.0, 50.0, size=(1, 3))
            
        verts = base_verts * scale + pos
        
        all_verts.append(verts)
        all_indices.append(base_indices + i * 8)
        
    all_verts = np.vstack(all_verts)
    all_indices = np.concatenate(all_indices)
    
    return all_verts, all_indices


@wp.kernel
def raycast_kernel_xyz(
    mesh: wp.uint64,
    width: int,
    height: int,
    fov_v_rad: float,
    big_arrays: wp.array(dtype=float, ndim=2),
    output_xyz: wp.array(dtype=wp.vec3),
    output_meta: wp.array(dtype=float, ndim=3)
):
    tid = wp.tid()
    
    j = tid % width
    i = tid // width
    
    if i >= height:
        return
        
    # j: 0 to width-1 maps to 0 to 2*PI
    theta = (float(j) / float(width)) * 2.0 * wp.pi
    
    # i: 0 to height-1 maps to -fov_v_rad/2 to fov_v_rad/2
    phi = (float(i) / float(height - 1) - 0.5) * fov_v_rad
    
    # Spherical to cartesian direction
    cos_phi = wp.cos(phi)
    dir_x = cos_phi * wp.cos(theta)
    dir_y = wp.sin(phi)
    dir_z = cos_phi * wp.sin(theta)
    
    dir = wp.vec3(dir_x, dir_y, dir_z)
    pos = wp.vec3(0.0, 0.0, 0.0)
    
    # Query mesh
    t = float(0.0)
    u = float(0.0)
    v = float(0.0)
    sign = float(0.0)
    n = wp.vec3()
    f = int(0)
    
    hit = wp.mesh_query_ray(mesh, pos, dir, 1000.0, t, u, v, sign, n, f)
    
    # 50+ arbitrary operations on 15 huge arrays
    val = float(0.0)
    for k in range(5):
        idx = (tid * 5 + k) % 2621440
        v0 = big_arrays[0, idx]
        v1 = big_arrays[1, idx]
        v2 = big_arrays[2, idx]
        v3 = big_arrays[3, idx]
        v4 = big_arrays[4, idx]
        v5 = big_arrays[5, idx]
        v6 = big_arrays[6, idx]
        v7 = big_arrays[7, idx]
        v8 = big_arrays[8, idx]
        v9 = big_arrays[9, idx]
        v10 = big_arrays[10, idx]
        v11 = big_arrays[11, idx]
        v12 = big_arrays[12, idx]
        v13 = big_arrays[13, idx]
        v14 = big_arrays[14, idx]
        
        val = val + wp.sin(v0) + wp.cos(v1) * v2 - wp.exp(v3) / (v4 + 2.0)
        val = val + wp.sin(v5) + wp.cos(v6) * v7 - v8 / (v9 + 2.0)
        val = val + wp.sin(v10) + wp.cos(v11) * v12 - v13 / (v14 + 2.0)

    if hit:
        # Save exact world position of the hit for the pointcloud
        hit_pos = pos + dir * t
        output_xyz[tid] = hit_pos
        
        output_meta[i, j, 0] = t
        output_meta[i, j, 1] = u
        output_meta[i, j, 2] = v
        output_meta[i, j, 3] = float(f)
        output_meta[i, j, 4] = val
    else:
        # If no hit, store a point far away or NaN (using a distant point here so it doesn't break structs)
        output_xyz[tid] = wp.vec3(0.0, 0.0, 0.0) # Or could use NaNs or filtering later
        
        output_meta[i, j, 0] = -1.0
        output_meta[i, j, 1] = 0.0
        output_meta[i, j, 2] = 0.0
        output_meta[i, j, 3] = -1.0
        output_meta[i, j, 4] = val

@wp.kernel
def raycast_kernel_v2_xyz(
    mesh: wp.uint64,
    width: int,
    height: int,
    fov_v_rad: float,
    big_arrays: wp.array(dtype=float, ndim=2),
    output_xyz: wp.array(dtype=wp.vec3),
    output_meta: wp.array(dtype=float, ndim=3)
):
    tid = wp.tid()
    
    j = tid % width
    i = tid // width
    
    if i >= height:
        return
        
    # j: 0 to width-1 maps to 0 to 2*PI
    theta = (float(j) / float(width)) * 2.0 * wp.pi
    
    # i: 0 to height-1 maps to -fov_v_rad/2 to fov_v_rad/2
    phi = (float(i) / float(height - 1) - 0.5) * fov_v_rad
    
    # Spherical to cartesian direction
    cos_phi = wp.cos(phi)
    dir_x = cos_phi * wp.cos(theta)
    dir_y = wp.sin(phi)
    dir_z = cos_phi * wp.sin(theta)
    
    dir = wp.vec3(dir_x, dir_y, dir_z)
    pos = wp.vec3(0.0, 0.0, 0.0)
    
    # Query mesh (newer API)
    query = wp.mesh_query_ray(mesh, pos, dir, 1000.0)
    
    # 50+ arbitrary operations on 15 huge arrays
    val = float(0.0)
    for k in range(10):
        idx = (tid * 5 + k) % 2621440
        v0 = big_arrays[0, idx]
        v1 = big_arrays[1, idx]
        v2 = big_arrays[2, idx]
        v3 = big_arrays[3, idx]
        v4 = big_arrays[4, idx]
        v5 = big_arrays[5, idx]
        v6 = big_arrays[6, idx]
        v7 = big_arrays[7, idx]
        v8 = big_arrays[8, idx]
        v9 = big_arrays[9, idx]
        v10 = big_arrays[10, idx]
        v11 = big_arrays[11, idx]
        v12 = big_arrays[12, idx]
        v13 = big_arrays[13, idx]
        v14 = big_arrays[14, idx]
        
        val = val + wp.sin(v0) + wp.cos(v1) * v2 - wp.exp(v3) / (v4 + 2.0)
        val = val + wp.sin(v5) + wp.cos(v6) * v7 - v8 / (v9 + 2.0)
        val = val + wp.sin(v10) + wp.cos(v11) * v12 - v13 / (v14 + 2.0)

    if query.result:
        hit_pos = pos + dir * query.t
        output_xyz[tid] = hit_pos
        
        output_meta[i, j, 0] = query.t
        output_meta[i, j, 1] = query.u
        output_meta[i, j, 2] = query.v
        output_meta[i, j, 3] = float(query.face)
        output_meta[i, j, 4] = val
    else:
        output_xyz[tid] = wp.vec3(0.0, 0.0, 0.0)
        
        output_meta[i, j, 0] = -1.0
        output_meta[i, j, 1] = 0.0
        output_meta[i, j, 2] = 0.0
        output_meta[i, j, 3] = -1.0
        output_meta[i, j, 4] = val


def benchmark():
    print("Generating 500 cubes...")
    verts, indices = create_cubes(1000)
    
    wp_verts = wp.array(verts, dtype=wp.vec3, device="cuda")
    wp_indices = wp.array(indices, dtype=int, device="cuda")
    
    mesh = wp.Mesh(points=wp_verts, indices=wp_indices)
    
    width = 2048
    height = 128
    fov_v_deg = 45.0
    fov_v_rad = fov_v_deg * np.pi / 180.0
    
    # Test array to store 5 variables per pixel: shape (height, width, 5)
    output_meta = wp.zeros((height, width, 5), dtype=float, device="cuda")
    # Linear array to store XYZ hit positions directly for fast struct packing
    output_xyz = wp.zeros(width * height, dtype=wp.vec3, device="cuda")

    print("Loading 15 big arrays (10MB float32 each)...")
    arr_size = 2621440 # ~10MB of float32
    # Create random arrays to simulate 'loading'
    big_arrays_np = np.random.rand(15, arr_size).astype(np.float32)
    big_arrays = wp.array(big_arrays_np, dtype=float, device="cuda")
    
    # Try identifying which kernel to use based on Warp version
    kernel_to_use = raycast_kernel_xyz
    try:
        if hasattr(wp, "MeshQueryRay"):
            kernel_to_use = raycast_kernel_v2_xyz
    except:
        pass
        
    print(f"Using kernel: {kernel_to_use.__name__}")
        
    print("Warming up kernel...")
    # Warmup
    wp.launch(
        kernel=kernel_to_use,
        dim=width * height,
        inputs=[mesh.id, width, height, fov_v_rad, big_arrays, output_xyz, output_meta],
        device="cuda"
    )
    wp.synchronize()
    
    num_iters = 100
    print(f"Benchmarking {num_iters} iterations with new readbacks...")
    
    start_time = time.time()
    for _ in range(num_iters):
        wp.launch(
            kernel=kernel_to_use,
            dim=width * height,
            inputs=[mesh.id, width, height, fov_v_rad, big_arrays, output_xyz, output_meta],
            device="cuda"
        )
        wp.synchronize_device() # ensure kernel is done before readback
        
        # Super fast readback: converts wp.vec3 array on GPU directly to 
        # a contiguous numpy array of float32s of shape (N, 3)
        xyz_np = output_xyz.numpy()
        meta_np = output_meta.numpy()
        
        # Filter valid points
        hit_mask = (meta_np[:, :, 0] >= 0.0).flatten()
        valid_points = xyz_np[hit_mask]
        
        # Pack precisely to bytes
        packed_bytes = valid_points.astype(np.float32).tobytes()
        
    wp.synchronize()
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time_ms = (total_time / num_iters) * 1000.0
    
    print("-" * 50)
    print(f"Resolution: {width} x {height} ({width*height} rays)")
    print(f"Total time for {num_iters} iters : {total_time:.4f} s")
    print(f"Average time per iteration   : {avg_time_ms:.2f} ms")
    print(f"Rays per second              : {((width*height) / (avg_time_ms/1000.0)):.2e}")
    print(f"Valid hit points (last iter) : {len(valid_points)}")
    print(f"Packed bytes size (last iter): {len(packed_bytes)} bytes")
    print("-" * 50)

if __name__ == '__main__':
    benchmark()
