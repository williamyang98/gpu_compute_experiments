import wgpu
from functools import reduce
import math
import numpy as np
import time
import sys
from pprint import pprint

# x, y, z
grid_size = [32, 256, 512]
total_cells = grid_size[0]*grid_size[1]*grid_size[2]
workgroup_size = [1, 2, 128]
n_dims = 3
total_shader_loops = 32
total_compute_passes = 3

dispatch_size = [math.ceil(g/l) for g,l in zip(grid_size, workgroup_size)]
print(f"=== PARAMETERS ===")
print(f"grid_size={grid_size}")
print(f"total_cells={total_cells}")
print(f"workgroup_size={workgroup_size}")
print(f"dispatch_size={dispatch_size}")
print(f"total_shader_loops={total_shader_loops}")
print(f"total_compute_passes={total_compute_passes}")
sys.stdout.flush()

flops_per_cell = 12
shader_source = f"""
@group(0) @binding(0)
var<storage,read> x: array<f32>;

@group(0) @binding(1)
var<storage,read_write> y: array<f32>;

fn get_offset(i: vec3<u32>) -> u32 {{
    let x: u32 = i.x % {grid_size[0]};
    let y: u32 = i.y % {grid_size[1]};
    let z: u32 = i.z % {grid_size[2]};
    let offset: u32 = z + y*{grid_size[2]} + x*{grid_size[2]*grid_size[1]};
    return offset*{n_dims};
}}

@compute
@workgroup_size({",".join(map(str, workgroup_size))})
fn main(@builtin(global_invocation_id) i0: vec3<u32>) {{
    if (i0.x >= {grid_size[0]}) {{ return; }}
    if (i0.y >= {grid_size[1]}) {{ return; }}
    if (i0.z >= {grid_size[2]}) {{ return; }}
    let i = get_offset(i0);
    let iz = get_offset(i0 + vec3(0,0,1));
    let iy = get_offset(i0 + vec3(0,1,0));
    let ix = get_offset(i0 + vec3(1,0,0));
    for (var j: u32 = 0; j < {total_shader_loops}; j = j+1) {{
        y[i+0] += (x[i+2]-x[iy+2]) - (x[i+1]-x[iz+1]);
        y[i+1] += (x[i+0]-x[iz+0]) - (x[i+2]-x[ix+2]);
        y[i+2] += (x[i+1]-x[ix+1]) - (x[i+0]-x[iy+0]);
    }}
}}
"""

# input/output cpu buffers
x_cpu = np.zeros(grid_size + [n_dims,], dtype=np.float32)
y_cpu = np.zeros(grid_size + [n_dims,], dtype=np.float32)

x = np.reshape(x_cpu, (n_dims*total_cells,))
x[:] = (-1.0*np.arange(0, n_dims*total_cells, dtype=np.float32) + 0.5) % 0.88490

# get wgpu device
adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
device = adapter.request_device_sync(
    # Request a device with the timestamp_query feature, so we can profile our computation
    required_features=[wgpu.FeatureName.timestamp_query]
)
print(f"> selected wgpu device", flush=True)
pprint(adapter.info)
sys.stdout.flush()

# input/output gpu buffers
x_gpu = device.create_buffer_with_data(data=x_cpu.data, usage=wgpu.BufferUsage.STORAGE)
y_gpu = device.create_buffer(
    size=y_cpu.data.nbytes,
    usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
)
y_gpu_readback = device.create_buffer(
    size=y_cpu.data.nbytes,
    usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST
)

# create pipeline
cshader = device.create_shader_module(code=shader_source)
binding_layouts = [
    {
        "binding": 0,
        "visibility": wgpu.ShaderStage.COMPUTE,
        "buffer": {
            "type": wgpu.BufferBindingType.read_only_storage,
        },
    },
    {
        "binding": 1,
        "visibility": wgpu.ShaderStage.COMPUTE,
        "buffer": {
            "type": wgpu.BufferBindingType.storage,
        },
    },
]
bind_group_layout = device.create_bind_group_layout(entries=binding_layouts)
pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[bind_group_layout])
compute_pipeline = device.create_compute_pipeline(
    layout=pipeline_layout,
    compute={"module": cshader, "entry_point": "main"},
)

# query buffers
total_queries = total_compute_passes*2
query_set = device.create_query_set(type=wgpu.QueryType.timestamp, count=total_queries)
query_cpu = np.zeros((total_queries,), dtype=np.uint64)
query_gpu = device.create_buffer(
    size=query_cpu.data.nbytes,
    usage=wgpu.BufferUsage.QUERY_RESOLVE | wgpu.BufferUsage.COPY_SRC,
)
query_gpu_readback = device.create_buffer(
    size=query_cpu.data.nbytes,
    usage=wgpu.BufferUsage.MAP_READ | wgpu.BufferUsage.COPY_DST
)

# execute commands
bindings = [
    {
        "binding": 0,
        "resource": {"buffer": x_gpu, "offset": 0, "size": x_gpu.size},
    },
    {
        "binding": 1,
        "resource": {"buffer": y_gpu, "offset": 0, "size": y_gpu.size},
    },
]
bind_group = device.create_bind_group(layout=bind_group_layout, entries=bindings)
print(f"> encode commands", flush=True)
command_encoder = device.create_command_encoder()
for i in range(total_compute_passes):
    timestamp_offset = i*2
    compute_pass = command_encoder.begin_compute_pass(
        timestamp_writes={
            "query_set": query_set,
            "beginning_of_pass_write_index": timestamp_offset,
            "end_of_pass_write_index": timestamp_offset+1,
        }
    )
    compute_pass.set_pipeline(compute_pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups(*dispatch_size)  # x y z
    compute_pass.end()
command_encoder.resolve_query_set(
    query_set=query_set,
    first_query=0,
    query_count=total_queries,
    destination=query_gpu,
    destination_offset=0,
)
command_encoder.copy_buffer_to_buffer(
    source=query_gpu,
    source_offset=0,
    destination=query_gpu_readback,
    destination_offset=0,
    size=query_gpu.size
)
command_encoder.copy_buffer_to_buffer(
    source=y_gpu,
    source_offset=0,
    destination=y_gpu_readback,
    destination_offset=0,
    size=y_gpu.size
)
device.queue.submit([command_encoder.finish()])
print(f"> submit commands", flush=True)

# read y_gpu
y_gpu_readback.map_sync(wgpu.MapMode.READ, 0, y_gpu.size)
y_gpu_memview = y_gpu_readback.read_mapped(buffer_offset=0, size=y_gpu.size, copy=True)
y_gpu_readback.unmap()
y_gpu_out = np.frombuffer(y_gpu_memview, dtype=np.float32)
y_gpu_out = np.reshape(y_gpu_out, grid_size+[n_dims,])
print(f"> read y_gpu", flush=True)

# read queries
query_gpu_readback.map_sync(wgpu.MapMode.READ, 0, query_gpu.size)
query_gpu_memview = query_gpu_readback.read_mapped(buffer_offset=0, size=query_gpu.size, copy=True)
query_gpu_readback.unmap()
query_cpu = np.frombuffer(query_gpu_memview, dtype=np.uint64)
query_cpu = np.reshape(query_cpu, (total_queries,))
print(f"> read queries", flush=True)

gpu_delta_ns = (query_cpu[1::2] - query_cpu[0::2]).astype(np.float32) * 10 # NOTE: need to rescale results
gpu_delta_ns_avg = np.mean(gpu_delta_ns)
gpu_cell_rate = total_cells / (gpu_delta_ns_avg*1e-9) * total_shader_loops
gpu_flops = gpu_cell_rate*flops_per_cell
print(f"=== GPU MEASUREMENTS ===")
print(f"gpu_delta_avg={gpu_delta_ns_avg*1e-3:.3f} us")
print(f"gpu_cell_rate={gpu_cell_rate*1e-6:.3f} M/s")
print(f"gpu_flops={gpu_flops*1e-9:.3f} GFlops")
sys.stdout.flush()

# get cpu results
def cpu_shader(x, y, wrap_around=True):
    if not wrap_around:
        y[:,:-1,:-1,0] += (x[:,:-1,:-1,2]-x[:,1:,:-1,2]) - (x[:,:-1,:-1,1]-x[:,:-1,1:,1])
        y[:-1,:,:-1,1] += (x[:-1,:,:-1,0]-x[:-1,:,1:,0]) - (x[:-1,:,:-1,2]-x[1:,:,:-1,2])
        y[:-1,:-1,:,2] += (x[:-1,:-1,:,1]-x[1:,:-1,:,1]) - (x[:-1,:-1,:,0]-x[:-1,1:,:,0])
    else:
        y[:,:-1,:,0] += (x[:,:-1,:,2]-x[:,1:,:,2])
        y[:,-1,:,0] += (x[:,-1,:,2]-x[:,0,:,2])
        y[:,:,:-1,0] -= (x[:,:,:-1,1]-x[:,:,1:,1])
        y[:,:,-1,0] -= (x[:,:,-1,1]-x[:,:,0,1])

        y[:,:,:-1,1] += (x[:,:,:-1,0]-x[:,:,1:,0])
        y[:,:,-1,1] += (x[:,:,-1,0]-x[:,:,0,0])
        y[:-1,:,:,1] -= (x[:-1,:,:,2]-x[1:,:,:,2])
        y[-1,:,:,1] -= (x[-1,:,:,2]-x[0,:,:,2])

        y[:-1,:,:,2] += (x[:-1,:,:,1]-x[1:,:,:,1])
        y[-1,:,:,2] += (x[-1,:,:,1]-x[0,:,:,1])
        y[:,:-1,:,2] -= (x[:,:-1,:,0]-x[:,1:,:,0])
        y[:,-1,:,2] -= (x[:,-1,:,0]-x[:,0,:,0])

y_cpu_out = np.zeros(y_cpu.shape, dtype=y_cpu.dtype)

cpu_delta_ns = np.zeros((total_compute_passes,), dtype=np.uint64)
for i in range(total_compute_passes):
    start_ns = time.perf_counter_ns()
    for _ in range(total_shader_loops):
        cpu_shader(x_cpu, y_cpu_out, wrap_around=True)
    end_ns = time.perf_counter_ns()
    delta_ns = end_ns - start_ns
    cpu_delta_ns[i] = delta_ns
cpu_delta_ns_avg = np.mean(cpu_delta_ns)
cpu_cell_rate = total_cells / (cpu_delta_ns_avg*1e-9) * total_shader_loops
cpu_flops = cpu_cell_rate*flops_per_cell
print(f"=== CPU MEASUREMENTS ===")
print(f"cpu_delta_avg={cpu_delta_ns_avg*1e-3:.3f} us")
print(f"cpu_cell_rate={cpu_cell_rate*1e-6:.3f} M/s")
print(f"cpu_flops={cpu_flops*1e-9:.3f} GFlops")
print(f"gpu/cpu = {gpu_cell_rate/cpu_cell_rate:.2f}x")
sys.stdout.flush()

# compare gpu and cpu results to check they are the same
error = y_cpu_out - y_gpu_out
#error = error[:-1,:-1,:-1,:] # skip last dimension on curl
error_max = np.max(error)
error_min = np.min(error)
error_abs = np.abs(error)
error_avg = np.mean(error)
error_abs_avg = np.mean(error_abs)

print(f"=== ERROR MEASUREMENTS ===")
print(f"error_min={error_min:.3e}")
print(f"error_max={error_max:.3e}")
print(f"error_avg={error_avg:.3e}")
print(f"error_abs_avg={error_abs_avg:.3e}")
sys.stdout.flush()

n_read = 1
print(y_gpu_out[:n_read, :n_read, :n_read, :])
print(y_cpu_out[:n_read, :n_read, :n_read, :])
print(y_gpu_out[-n_read:, -n_read:, -n_read:,:])
print(y_cpu_out[-n_read:, -n_read:, -n_read:,:])

