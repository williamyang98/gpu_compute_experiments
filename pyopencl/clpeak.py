from collections import namedtuple
from pprint import pprint
from tabulate import tabulate
import argparse
import math
import numpy as np
import pyopencl as cl
import sys
import time

def create_program_source(typename, mad_func, total_mads):
    T = typename
    N = total_mads
    MAD = mad_func

    MAX_DIVISOR = 16*16
    assert(N % MAX_DIVISOR == 0)

    return f"""
#if defined(cl_khr_fp16)
  #pragma OPENCL EXTENSION cl_khr_fp16 : enable
  #define HALF_AVAILABLE
#endif

#define MAD_4(x, y)     x={MAD("y","x")};     y={MAD("x","y")};     x={MAD("y","x")};     y={MAD("x","y")};
#define MAD_16(x, y)    MAD_4(x, y);        MAD_4(x, y);        MAD_4(x, y);        MAD_4(x, y);
#define MAD_64(x, y)    MAD_16(x, y);       MAD_16(x, y);       MAD_16(x, y);       MAD_16(x, y);

__kernel void compute_1(__global {T} *ptr, {T} _A) {{
    {T} x = _A;
    {T} y = ({T})get_local_id(0);
    for(int i = 0; i < {N//(16*1)}; i++) {{
        MAD_16(x, y);
    }}
    ptr[get_global_id(0)] = y;
}}


__kernel void compute_2(__global {T} *ptr, {T} _A) {{
    {T}2 x = ({T}2)(_A, (_A+1));
    {T}2 y = ({T}2)get_local_id(0);
    for(int i = 0; i < {N//(16*2)}; i++) {{
        MAD_16(x, y);
    }}
    ptr[get_global_id(0)] = (y.S0) + (y.S1);
}}

__kernel void compute_4(__global {T} *ptr, {T} _A) {{
    {T}4 x = ({T}4)(_A, (_A+1), (_A+2), (_A+3));
    {T}4 y = ({T}4)get_local_id(0);
    for(int i = 0; i < {N//(16*4)}; i++) {{
        MAD_16(x, y);
    }}
    ptr[get_global_id(0)] = (y.S0) + (y.S1) + (y.S2) + (y.S3);
}}

__kernel void compute_8(__global {T} *ptr, {T} _A) {{
    {T}8 x = ({T}8)(_A, (_A+1), (_A+2), (_A+3), (_A+4), (_A+5), (_A+6), (_A+7));
    {T}8 y = ({T}8)get_local_id(0);
    for(int i = 0; i < {N//(16*8)}; i++) {{
        MAD_16(x, y);
    }}
    ptr[get_global_id(0)] = (y.S0) + (y.S1) + (y.S2) + (y.S3) + (y.S4) + (y.S5) + (y.S6) + (y.S7);
}}

__kernel void compute_16(__global {T} *ptr, {T} _A) {{
    {T}16 x = ({T}16)(_A, (_A+1), (_A+2), (_A+3), (_A+4), (_A+5), (_A+6), (_A+7), (_A+8), (_A+9), (_A+10), (_A+11), (_A+12), (_A+13), (_A+14), (_A+15));
    {T}16 y = ({T}16)get_local_id(0);
    for(int i = 0; i < {N//(16*16)}; i++) {{
        MAD_16(x, y);
    }}

    {T}2 t = (y.S01) + (y.S23) + (y.S45) + (y.S67) + (y.S89) + (y.SAB) + (y.SCD) + (y.SEF);
    ptr[get_global_id(0)] = t.S0 + t.S1;
}}
"""

def get_platform_info(platform):
    keys = [
        "EXTENSIONS",
        # "EXTENSIONS_WITH_VERSION",
        "HOST_TIMER_RESOLUTION",
        "NAME",
        # "NUMERIC_VERSION",
        "PROFILE",
        "VENDOR",
        "VERSION",
    ]
    info = {}
    for key in keys:
        try:
            value = platform.get_info(getattr(cl.platform_info, key))
            info.setdefault(key.lower(), value)
        except:
            pass
    return info

def get_device_info(device):
    keys = [
        "ADDRESS_BITS",
        "ATOMIC_FENCE_CAPABILITIES",
        "ATOMIC_MEMORY_CAPABILITIES",
        "ATTRIBUTE_ASYNC_ENGINE_COUNT_NV",
        "AVAILABLE",
        "AVAILABLE_ASYNC_QUEUES_AMD",
        "BOARD_NAME_AMD",
        "BUILT_IN_KERNELS",
        "BUILT_IN_KERNELS_WITH_VERSION",
        "COMPILER_AVAILABLE",
        "COMPUTE_CAPABILITY_MAJOR_NV",
        "COMPUTE_CAPABILITY_MINOR_NV",
        "DEVICE_ENQUEUE_CAPABILITIES",
        "DOUBLE_FP_CONFIG",
        "DRIVER_VERSION",
        "ENDIAN_LITTLE",
        "ERROR_CORRECTION_SUPPORT",
        "EXECUTION_CAPABILITIES",
        "EXTENSIONS",
        "EXTENSIONS_WITH_VERSION",
        "EXT_MEM_PADDING_IN_BYTES_QCOM",
        "GENERIC_ADDRESS_SPACE_SUPPORT",
        "GFXIP_MAJOR_AMD",
        "GFXIP_MINOR_AMD",
        "GLOBAL_FREE_MEMORY_AMD",
        "GLOBAL_MEM_CACHELINE_SIZE",
        "GLOBAL_MEM_CACHE_SIZE",
        "GLOBAL_MEM_CACHE_TYPE",
        "GLOBAL_MEM_CHANNELS_AMD",
        "GLOBAL_MEM_CHANNEL_BANKS_AMD",
        "GLOBAL_MEM_CHANNEL_BANK_WIDTH_AMD",
        "GLOBAL_MEM_SIZE",
        "GLOBAL_VARIABLE_PREFERRED_TOTAL_SIZE",
        "GPU_OVERLAP_NV",
        "HALF_FP_CONFIG",
        "HOST_UNIFIED_MEMORY",
        "ILS_WITH_VERSION",
        "IL_VERSION",
        "IMAGE2D_MAX_HEIGHT",
        "IMAGE2D_MAX_WIDTH",
        "IMAGE3D_MAX_DEPTH",
        "IMAGE3D_MAX_HEIGHT",
        "IMAGE3D_MAX_WIDTH",
        "IMAGE_BASE_ADDRESS_ALIGNMENT",
        "IMAGE_MAX_ARRAY_SIZE",
        "IMAGE_MAX_BUFFER_SIZE",
        "IMAGE_PITCH_ALIGNMENT",
        "IMAGE_SUPPORT",
        "INTEGRATED_MEMORY_NV",
        "KERNEL_EXEC_TIMEOUT_NV",
        "LINKER_AVAILABLE",
        "LOCAL_MEM_BANKS_AMD",
        "LOCAL_MEM_SIZE",
        "LOCAL_MEM_SIZE_PER_COMPUTE_UNIT_AMD",
        "LOCAL_MEM_TYPE",
        "MAX_ATOMIC_COUNTERS_EXT",
        "MAX_CLOCK_FREQUENCY",
        "MAX_COMPUTE_UNITS",
        "MAX_CONSTANT_ARGS",
        "MAX_CONSTANT_BUFFER_SIZE",
        "MAX_GLOBAL_VARIABLE_SIZE",
        "MAX_MEM_ALLOC_SIZE",
        "MAX_NUM_SUB_GROUPS",
        "MAX_ON_DEVICE_EVENTS",
        "MAX_ON_DEVICE_QUEUES",
        "MAX_PARAMETER_SIZE",
        "MAX_PIPE_ARGS",
        "MAX_READ_IMAGE_ARGS",
        "MAX_READ_WRITE_IMAGE_ARGS",
        "MAX_SAMPLERS",
        "MAX_WORK_GROUP_SIZE",
        "MAX_WORK_GROUP_SIZE_AMD",
        "MAX_WORK_ITEM_DIMENSIONS",
        "MAX_WORK_ITEM_SIZES",
        "MAX_WRITE_IMAGE_ARGS",
        "MEM_BASE_ADDR_ALIGN",
        "ME_VERSION_INTEL",
        "MIN_DATA_TYPE_ALIGN_SIZE",
        "NAME",
        "NATIVE_VECTOR_WIDTH_CHAR",
        "NATIVE_VECTOR_WIDTH_DOUBLE",
        "NATIVE_VECTOR_WIDTH_FLOAT",
        "NATIVE_VECTOR_WIDTH_HALF",
        "NATIVE_VECTOR_WIDTH_INT",
        "NATIVE_VECTOR_WIDTH_LONG",
        "NATIVE_VECTOR_WIDTH_SHORT",
        "NON_UNIFORM_WORK_GROUP_SUPPORT",
        "NUMERIC_VERSION",
        "NUM_SIMULTANEOUS_INTEROPS_INTEL",
        "OPENCL_C_ALL_VERSIONS",
        "OPENCL_C_FEATURES",
        "OPENCL_C_VERSION",
        "PAGE_SIZE_QCOM",
        "PARENT_DEVICE",
        "PARTITION_AFFINITY_DOMAIN",
        "PARTITION_MAX_SUB_DEVICES",
        "PARTITION_PROPERTIES",
        "PARTITION_TYPE",
        "PCIE_ID_AMD",
        "PCI_BUS_ID_NV",
        "PCI_DOMAIN_ID_NV",
        "PCI_SLOT_ID_NV",
        "PIPE_MAX_ACTIVE_RESERVATIONS",
        "PIPE_MAX_PACKET_SIZE",
        "PIPE_SUPPORT",
        "PLATFORM",
        "PREFERRED_CONSTANT_BUFFER_SIZE_AMD",
        "PREFERRED_GLOBAL_ATOMIC_ALIGNMENT",
        "PREFERRED_INTEROP_USER_SYNC",
        "PREFERRED_LOCAL_ATOMIC_ALIGNMENT",
        "PREFERRED_PLATFORM_ATOMIC_ALIGNMENT",
        "PREFERRED_VECTOR_WIDTH_CHAR",
        "PREFERRED_VECTOR_WIDTH_DOUBLE",
        "PREFERRED_VECTOR_WIDTH_FLOAT",
        "PREFERRED_VECTOR_WIDTH_HALF",
        "PREFERRED_VECTOR_WIDTH_INT",
        "PREFERRED_VECTOR_WIDTH_LONG",
        "PREFERRED_VECTOR_WIDTH_SHORT",
        "PREFERRED_WORK_GROUP_SIZE_AMD",
        "PREFERRED_WORK_GROUP_SIZE_MULTIPLE",
        "PRINTF_BUFFER_SIZE",
        "PROFILE",
        "PROFILING_TIMER_OFFSET_AMD",
        "PROFILING_TIMER_RESOLUTION",
        "QUEUE_ON_DEVICE_MAX_SIZE",
        "QUEUE_ON_DEVICE_PREFERRED_SIZE",
        "QUEUE_ON_DEVICE_PROPERTIES",
        "QUEUE_ON_HOST_PROPERTIES",
        "QUEUE_PROPERTIES",
        "REFERENCE_COUNT",
        "REGISTERS_PER_BLOCK_NV",
        "SIMD_INSTRUCTION_WIDTH_AMD",
        "SIMD_PER_COMPUTE_UNIT_AMD",
        "SIMD_WIDTH_AMD",
        "SIMULTANEOUS_INTEROPS_INTEL",
        "SINGLE_FP_CONFIG",
        "SPIR_VERSIONS",
        "SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS",
        "SVM_CAPABILITIES",
        "THREAD_TRACE_SUPPORTED_AMD",
        "TOPOLOGY_AMD",
        "TYPE",
        "VENDOR",
        "VENDOR_ID",
        "VERSION",
        "WARP_SIZE_NV",
        "WAVEFRONT_WIDTH_AMD",
        "WORK_GROUP_COLLECTIVE_FUNCTIONS_SUPPORT",
    ]
    info = {}
    for key in keys:
        try:
            value = device.get_info(getattr(cl.device_info, key))
            info.setdefault(key.lower(), value)
        except:
            pass
    return info

class NanoTimer:
    def __init__(self):
        self.start_ns = None
        self.end_ns = None

    def __enter__(self):
        self.start_ns = time.time_ns()
        return self

    def __exit__(self, *args):
        self.end_ns = time.time_ns()        

    def get_delta_ns(self):
        return self.end_ns - self.start_ns

def convert_to_si_prefix(x):
    if x >= 1e12: return ("T", x*1e-12)
    if x >= 1e9:  return ("G", x*1e-9)
    if x >= 1e6:  return ("M", x*1e-6)
    if x >= 1e3:  return ("k", x*1e-3)
    if x >= 1e0:  return ("" , x*1e0)
    if x >= 1e-3: return ("m", x*1e3)
    if x >= 1e-6: return ("u", x*1e6)
    if x >= 1e-9: return ("n", x*1e9)
    else:         return ("n", x*1e9)

KernelVariant = namedtuple("KernelVariant", ["name", "typename", "nbytes", "init_value", "mad_func", "total_mads"])
FLOAT_MAD = lambda x,y: f"mad({x},{y},{x})"
INTEGER_MAD = lambda x,y: f"({x}*{y})+{x}"
FAST_INTEGER_MAD = lambda x,y: f"mad24({x},{y},{x})"
KERNEL_VARIANTS = [
    KernelVariant("f64", "double", 8, np.float64(1.3), FLOAT_MAD, 512),
    KernelVariant("f32", "float", 4, np.float32(1.3), FLOAT_MAD, 2048),
    KernelVariant("f16", "half", 2, np.float16(1.3), FLOAT_MAD, 4096),
    KernelVariant("s64", "long", 8, np.int64(4), INTEGER_MAD, 512),
    KernelVariant("s32", "int", 4, np.int32(4), INTEGER_MAD, 1024),
    KernelVariant("s16", "short", 2, np.int16(4), INTEGER_MAD, 2048),
    KernelVariant("s8", "char", 1, np.int8(4), INTEGER_MAD, 4096),
    KernelVariant("fast_s32", "int", 4, np.int32(4), FAST_INTEGER_MAD, 2048),
]
KERNEL_NAMES = [v.name for v in KERNEL_VARIANTS]
KERNEL_LOOKUP = {v.name:v for v in KERNEL_VARIANTS}

class FancyKernelResultPrinter:
    def __init__(self):
        self.is_terminated = False

    def print_header(self):
        print("+----------+-------+--------------+\n"\
              "| name     | width | speed        |\n"\
              "+==========+=======+==============+", flush=True)

    def print_row(self, variant, width, iops):
        prefix, si_iops = convert_to_si_prefix(iops)
        si_iops = f"{si_iops:.2f}"
        iop_string = f"{si_iops:>6} {prefix}IOPS"
        print(f"| {variant.name:<8} | {str(width):<5} | {iop_string:<12} |", flush=True)
        self.is_terminated = False

    def print_divider(self):
        if self.is_terminated: return
        self.is_terminated = True
        print("+----------+-------+--------------+", flush=True)

    def __enter__(self):
        self.print_header()
        return self

    def __exit__(self, *args):
        self.print_divider()

class BasicKernelResultPrinter:
    def __init__(self):
        return

    def print_header(self):
        print("name, width, speed", flush=True)

    def print_row(self, variant, width, iops):
        print(f"{variant.name}, {width}, {iops:.0f}", flush=True)

    def print_divider(self):
        return

    def __enter__(self):
        self.print_header()
        return self

    def __exit__(self, *args):
        self.print_divider()

def run_benchmarks(platform, platform_info, device, device_info, kernel_variants, fancy_print):
    context = cl.Context(devices=[device])

    print("> Creating compute programs")
    program_sources = []
    programs = []
    for v in kernel_variants:
        program_source = create_program_source(v.typename, mad_func=v.mad_func, total_mads=v.total_mads)
        gpu_program = cl.Program(context, program_source).build()
        program_sources.append(program_source)
        programs.append(gpu_program)

    print("> Running kernels")
    total_iters = 32
    total_warmup = 4
    max_compute_units = device_info["max_compute_units"]
    workgroups_per_compute_unit = 2048
    max_workgroup_threads = device_info["max_work_group_size"]
    max_mem_alloc_size = device_info["max_mem_alloc_size"]

    sys.stdout.flush() # busy loop prevent stdout flush

    printer = FancyKernelResultPrinter() if fancy_print else BasicKernelResultPrinter()
    with printer:
        for kernel_variant, program in zip(kernel_variants, programs):
            sizeof_type = kernel_variant.nbytes
            max_global_threads = max_compute_units*workgroups_per_compute_unit*max_workgroup_threads
            max_global_threads = min(max_global_threads*sizeof_type, max_mem_alloc_size) // sizeof_type
            max_global_threads = (max_global_threads//max_workgroup_threads) * max_workgroup_threads

            y_gpu = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, max_global_threads*sizeof_type)

            kernels = [
                (program.compute_1, 1),
                (program.compute_2, 2),
                (program.compute_4, 4),
                (program.compute_8, 8),
                (program.compute_16, 16),
            ]

            iops_per_mad = 2
            mads_per_thread = kernel_variant.total_mads
            iops_per_thread = iops_per_mad*mads_per_thread
            A = kernel_variant.init_value

            global_size = max_global_threads
            workgroup_size = max_workgroup_threads

            for kernel, width in kernels:
                kernel.set_arg(0, y_gpu)
                kernel.set_arg(1, A)

                with cl.CommandQueue(context) as queue:
                    for _ in range(total_warmup):
                        cl.enqueue_nd_range_kernel(queue, kernel, (global_size,), (workgroup_size,))
                        queue.flush()

                queue = cl.CommandQueue(context)
                with NanoTimer() as timer:
                    for _ in range(total_iters):
                        cl.enqueue_nd_range_kernel(queue, kernel, (global_size,), (workgroup_size,))
                        queue.flush()
                    queue.finish()
                delta_ns = timer.get_delta_ns()

                iops = global_size*iops_per_thread*total_iters / (delta_ns*1e-9)
                printer.print_row(kernel_variant, width, iops)
            printer.print_divider()

def list_platforms(platforms):
    print(f"> Listing {len(platforms)} platforms")
    platform_infos = list(map(get_platform_info, platforms))
    table = []
    headers = ["name", "vendor", "version"]
    for info in platform_infos:
        table.append([info.get(key, "Unknown") for key in headers])
    print(tabulate(table, headers=headers, showindex="always", tablefmt="grid"))

def list_devices(devices):
    print(f"> Listing {len(devices)} devices")
    device_infos = list(map(get_device_info, devices))
    table = []
    headers = ["name", "vendor", "version"]
    for info in device_infos:
        table.append([info.get(key, "Unknown") for key in headers])
    print(tabulate(table, headers=headers, showindex="always", tablefmt="grid"))

def list_kernels(kernels):
    print(f"> Listing {len(kernels)} kernels")
    table = []
    headers = ["name", "typename", "total_mads"]
    for k in kernels:
        table.append([k.name, k.typename, k.total_mads])
    print(tabulate(table, headers=headers, tablefmt="outline"))

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--platform", type=int, default=0, help="Platform index")
    parser.add_argument("--device", type=int, default=0, help="Device index")
    parser.add_argument("--list-platforms", action="store_true", help="List platforms")
    parser.add_argument("--list-devices", action="store_true", help="List devices")
    parser.add_argument("--list-kernels", action="store_true", help="List kernels")
    parser.add_argument("--kernels", default=None, help=f"[{','.join(KERNEL_NAMES)}]")
    parser.add_argument("--print-csv", action="store_true", help="Print results as csv")
    args = parser.parse_args()

    if args.list_kernels:
        list_kernels(KERNEL_VARIANTS)
        return

    if args.kernels == None:
        selected_kernel_names = KERNEL_NAMES
    else:
        selected_kernel_names = [name.strip() for name in args.kernels.split(',')]

    selected_kernels = []
    for name in selected_kernel_names:
        kernel = KERNEL_LOOKUP.get(name)
        if kernel == None:
            print(f"Unknown kernel {name} skipping")
        else:
            selected_kernels.append(kernel)

    if len(selected_kernels) == 0:
        print("Please select at least one kernel")
        list_kernels(KERNEL_VARIANTS)
        return
 
    platforms = cl.get_platforms()

    if args.list_platforms:
        list_platforms(platforms)
        return

    if args.platform >= len(platforms):
        print(f"Platform index {args.platform} is invalid for {len(platforms)} platforms")
        list_platforms(platforms)
        return

    platform = platforms[args.platform]
    platform_info = get_platform_info(platform)

    print(f"> Selected platform ({args.platform})")
    headers = ["name", "vendor", "version"]
    table = [[key, platform_info.get(key, "Unknown")] for key in headers]
    print(tabulate(table, tablefmt="outline"))

    devices = platform.get_devices()
    if args.list_devices:
        list_devices(devices)
        return

    if args.device >= len(devices):
        print(f"Device index {args.device} is invalid for {len(devices)} devices")
        list_devices(devices)
        return

    device = devices[args.device]
    device_info = get_device_info(device)

    print(f"> Selected device ({args.device})")
    headers = ["name", "vendor", "version"]
    table = [[key, device_info.get(key, "Unknown")] for key in headers]
    print(tabulate(table, tablefmt="outline"))

    list_kernels(selected_kernels)
    fancy_print = not args.print_csv
    run_benchmarks(platform, platform_info, device, device_info, selected_kernels, fancy_print)

if __name__ == "__main__":
    main()
