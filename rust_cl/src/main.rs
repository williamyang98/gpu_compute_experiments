use std::{ffi::c_void, ptr::null_mut};
use std::sync::{Condvar, Mutex, Arc, mpsc};
use std::time::{Instant, Duration};
use opencl3::{
    command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE}, 
    context::Context, device::Device, 
    error_codes::ClError, 
    event::Event,
    kernel::{ExecuteKernel, Kernel},
    memory::{Buffer, CL_MEM_READ_WRITE},
    platform::get_platforms, 
    program::Program,
    types::{cl_bool, cl_event},
};
use ndarray::{Array1, Array3, Array4, s};
use ndarray_npy::write_npy;
use threadpool::ThreadPool;

fn main() -> Result<(), String> {
    run().map_err(|err| err.to_string())
}

enum ReadbackBufferState {
    Pending(Event, usize),
    Closed,
    Empty,
}

impl ReadbackBufferState {
    fn is_closed(&self) -> bool {
        match self {
            Self::Closed => true,
            _ => false,
        }
    }

    fn is_empty(&self) -> bool {
        match self {
            Self::Empty => true,
            _ => false,
        }
    }

    fn is_pending(&self) -> bool {
        match self {
            Self::Pending(_, _) => true,
            _ => false,
        }
    }
}

struct ReadbackBuffer<T, U> {
    gpu_buffer: Mutex<Buffer<T>>,
    cpu_buffer: Mutex<U>,
    state: Mutex<ReadbackBufferState>,
    signal_state: Condvar,
}

impl<T,U> ReadbackBuffer<T,U> {
    fn new(gpu_buffer: Buffer<T>, cpu_buffer: U) -> Self {
        Self {
            gpu_buffer: Mutex::new(gpu_buffer),
            cpu_buffer: Mutex::new(cpu_buffer),
            state: Mutex::new(ReadbackBufferState::Empty),
            signal_state: Condvar::new(),
        }
    }

    fn wait_busy(&self) {
        let mut state = self.state.lock().unwrap();
        while state.is_empty()  {
            state = self.signal_state.wait(state).unwrap();
        }
    }

    fn free(&self) {
        let mut state = self.state.lock().unwrap();
        *state = ReadbackBufferState::Empty;
        self.signal_state.notify_one();
    }
 
    fn acquire(&self) {
        let mut state = self.state.lock().unwrap();
        while state.is_pending() {
            state = self.signal_state.wait(state).unwrap();
        }
    }

    fn make_busy(&self, ev: Event, iter: usize) {
        let mut state = self.state.lock().unwrap();
        *state = ReadbackBufferState::Pending(ev, iter);
        self.signal_state.notify_one();
    }

    fn close(&self) {
        let mut state = self.state.lock().unwrap();
        *state = ReadbackBufferState::Closed;
        self.signal_state.notify_one();
    }
}

fn run() -> Result<(), ClError> {
    let mut selected_platform = None;
    let mut selected_device = None;

    let platforms = get_platforms()?;
    for (platform_index, platform) in platforms.iter().enumerate() {
        selected_platform = Some(platform);
        println!("Platform {0}: {1}", platform_index, platform.name().unwrap_or("Unknown".to_owned()));
        let device_ids = platform.get_devices(opencl3::device::CL_DEVICE_TYPE_ALL)?;
        let devices: Vec<Device> = device_ids.iter().map(|id| Device::new(*id)).collect();
        for (device_index, device) in devices.iter().enumerate() {
            println!("  Device {0}: {1}", device_index, device.name().unwrap_or("Unknown".to_owned()));
            if platform_index == 0 && device_index == 0 {
                selected_device = Some(*device);
            }
        }
    }

    let platform = selected_platform.expect("No available opencl platform");
    let device = selected_device.expect("No available opencl device");
    let context = Arc::new(Context::from_device(&device)?);

    let max_compute_units: usize = device.max_compute_units()? as usize;
    let workgroups_per_compute_unit: usize = 2048;
    let max_workgroup_threads: usize = device.max_work_group_size()?;
    let max_mem_alloc_size: usize = device.max_mem_alloc_size()? as usize;
    let f32_size = size_of::<f32>();
    let max_global_threads = {
        let n: usize = max_compute_units*workgroups_per_compute_unit*max_workgroup_threads;
        let n = usize::min(n*f32_size, max_mem_alloc_size) / f32_size;
        let n = (n / max_workgroup_threads) * max_workgroup_threads;
        n
    };


    let n_dims: usize = 3;
    let grid_size = Array1::from(vec![16, 256, 512]);
    let (n_x, n_y, n_z) = (grid_size[0], grid_size[1], grid_size[2]);
    let workgroup_size = Array1::from(vec![1, 1, 256]);
    let total_workgroup_size: usize = workgroup_size.iter().product();
    assert!(total_workgroup_size <= max_workgroup_threads);

    let total_cells: usize = grid_size.iter().product();
    let dispatch_size = &grid_size / &workgroup_size;
    let global_size = &dispatch_size * &workgroup_size;

    let c_0: f32 = 299792458.0;
    let pi = std::f32::consts::PI;
    let mu_0: f32 = 4.0*pi*1e-7;
    let e_0: f32 = 1.0/(mu_0 * c_0.powi(2));
    let Z_0: f32 = mu_0*c_0;
    let d_xyz: f32 = 1e-3;
    let dt: f32 = 1e-12;

    let mut e_k = Array3::<f32>::from_elem((n_x, n_y, n_z), e_0);
    let mut sigma_k = Array3::<f32>::from_elem((n_x, n_y, n_z), 0.0);

    {
        let sigma_0: f32 = 1e8;
        let i = s![.., 30..90, 30..40];
        sigma_k.slice_mut(i).fill(sigma_0);
    }

    let a0_cpu: Array3<f32> = 1.0/(1.0 + &sigma_k/&e_k * dt);
    let a1_cpu: Array3<f32> = 1.0/(&e_k * d_xyz) * dt;
    let b0_cpu: f32 = 1.0/(&mu_0 * d_xyz) * dt;
    let mut E_cpu = Array4::<f32>::zeros((n_x, n_y, n_z, n_dims));
    let mut H_cpu = Array4::<f32>::zeros((n_x, n_y, n_z, n_dims));
    let mut cE_cpu = Array4::<f32>::zeros((n_x, n_y, n_z, n_dims));
    let mut cH_cpu = Array4::<f32>::zeros((n_x, n_y, n_z, n_dims));
    {
        let w: usize = 10;
        let a: Array1<f32> = ndarray::linspace(0.0, 2.0*pi, w*2).collect();
        let a = 0.53836 - 0.46164*a.cos();
        let c = n_z/2;
        let i = s![5..=6,..,c-w..c+w,0];
        E_cpu
            .slice_mut(i)
            .axis_iter_mut(ndarray::Axis(1))
            .for_each(|mut row| row.assign(&a));
        H_cpu
            .slice_mut(i)
            .axis_iter_mut(ndarray::Axis(1))
            .for_each(|mut row| row.assign(&a));
    }

    let mut E_gpu = unsafe { Buffer::<f32>::create(&context, CL_MEM_READ_WRITE, E_cpu.len(), null_mut::<c_void>())? };
    let mut H_gpu = unsafe { Buffer::<f32>::create(&context, CL_MEM_READ_WRITE, H_cpu.len(), null_mut::<c_void>())? };
    let mut cE_gpu = unsafe { Buffer::<f32>::create(&context, CL_MEM_READ_WRITE, cE_cpu.len(), null_mut::<c_void>())? };
    let mut cH_gpu = unsafe { Buffer::<f32>::create(&context, CL_MEM_READ_WRITE, cH_cpu.len(), null_mut::<c_void>())? };
    let mut a0_gpu = unsafe { Buffer::<f32>::create(&context, CL_MEM_READ_WRITE, a0_cpu.len(), null_mut::<c_void>())? };
    let mut a1_gpu = unsafe { Buffer::<f32>::create(&context, CL_MEM_READ_WRITE, a1_cpu.len(), null_mut::<c_void>())? };
    {
        let queue_props =  0;
        let queue = CommandQueue::create_default(&context, queue_props)?;
        let _ev = unsafe { queue.enqueue_write_buffer(&mut E_gpu, false as cl_bool, 0, E_cpu.as_slice().unwrap(), &[]) }?;
        let _ev = unsafe { queue.enqueue_write_buffer(&mut H_gpu, false as cl_bool, 0, H_cpu.as_slice().unwrap(), &[]) }?;
        let _ev = unsafe { queue.enqueue_write_buffer(&mut cE_gpu, false as cl_bool, 0, cE_cpu.as_slice().unwrap(), &[]) }?;
        let _ev = unsafe { queue.enqueue_write_buffer(&mut cH_gpu, false as cl_bool, 0, cH_cpu.as_slice().unwrap(), &[]) }?;
        let _ev = unsafe { queue.enqueue_write_buffer(&mut a0_gpu, false as cl_bool, 0, a0_cpu.as_slice().unwrap(), &[]) }?;
        let _ev = unsafe { queue.enqueue_write_buffer(&mut a1_gpu, false as cl_bool, 0, a1_cpu.as_slice().unwrap(), &[]) }?;
        queue.finish()?;
    }

    let mut E_out_readbacks: Vec<Arc<ReadbackBuffer<f32, Array4<f32>>>> = vec![];
    const TOTAL_READBACK_BUFFERS: usize = 3;
    for _ in 0..TOTAL_READBACK_BUFFERS {
        let mut E_cpu = Array4::<f32>::zeros((n_x, n_y, n_z, n_dims));
        let E_gpu = unsafe { Buffer::<f32>::create(&context, CL_MEM_READ_WRITE, E_cpu.len(), null_mut::<c_void>())? };
        E_out_readbacks.push(Arc::new(ReadbackBuffer::new(E_gpu, E_cpu)));
    }
    let mut E_out_readback_index: usize = 0;

    let pool = ThreadPool::new(TOTAL_READBACK_BUFFERS); 
    for (index, E_out_readback) in E_out_readbacks.iter().enumerate() {
        pool.execute({
            let E_out_readback = E_out_readback.clone();
            let context = context.clone();
            move || {
                let queue_props = 0;
                let timer = Instant::now();
                let queue = CommandQueue::create_default(&context, queue_props).unwrap();
                println!("[readback][{0}] Created queue {1} us", index, timer.elapsed().as_micros());

                loop {
                    E_out_readback.wait_busy();
                    let state = E_out_readback.state.lock().unwrap();
                    let (ev_copy, curr_iter) = match &*state {
                        ReadbackBufferState::Empty => panic!("Shouldn't get empty data"),
                        ReadbackBufferState::Closed => {
                            println!("[readback][{0}] Closing readback thread", index);
                            break;
                        },
                        ReadbackBufferState::Pending(ev_copy, curr_iter) => (ev_copy, *curr_iter),
                    };

                    let timer = Instant::now();
                    let ev_read = unsafe { 
                        let gpu_buffer = E_out_readback.gpu_buffer.lock().unwrap();
                        let mut cpu_buffer = E_out_readback.cpu_buffer.lock().unwrap();
                        queue.enqueue_read_buffer(&gpu_buffer, false as cl_bool, 0, cpu_buffer.as_slice_mut().unwrap(), &[ev_copy.get()]) 
                    }.unwrap();
                    drop(state);

                    ev_read.wait().unwrap();
                    queue.finish().unwrap();
                    println!("[readback][{0}] Read cpu buffer {1} us", index, timer.elapsed().as_micros());

                    let timer = Instant::now();
                    let cpu_buffer = E_out_readback.cpu_buffer.lock().unwrap();
                    write_npy(format!("./data/E_cpu_{0}.npy", curr_iter), &*cpu_buffer).unwrap();
                    println!("[readback][{0}] Wrote to file {1} us", index, timer.elapsed().as_micros());

                    E_out_readback.free();
                }
            }
        });
    }

    let shader_src: &'static str = include_str!("./shader.cl");
    let mut program = Program::create_from_source(&context, shader_src)?;
    program.build(context.devices(), "")?;
    let mut kernel_update_E = Kernel::create(&program, "update_E")?;
    let mut kernel_update_H = Kernel::create(&program, "update_H")?;

    const TOTAL_LOOPS: usize = 8192;
    // const RECORD_STRIDE: usize = 32;
    const RECORD_STRIDE: usize = 16;
    let global_timer = Instant::now();
    {
        // let queue_props =  0;
        let queue_props =  CL_QUEUE_PROFILING_ENABLE;
        let queue = CommandQueue::create_default(&context, queue_props)?;
        let mut last_ev_update_H: Option<Event> = None;
        let mut wait_update_H: Vec<cl_event> = vec![];
        for curr_iter in 0..TOTAL_LOOPS {
            wait_update_H.clear();
            if let Some(ev) = last_ev_update_H {
                wait_update_H.push(ev.get());
            }

            unsafe {
                let ev_update_E = ExecuteKernel::new(&kernel_update_E)
                    .set_arg(&E_gpu)
                    .set_arg(&H_gpu)
                    .set_arg(&cH_gpu)
                    .set_arg(&a0_gpu)
                    .set_arg(&a1_gpu)
                    .set_arg(&(n_x as i32))
                    .set_arg(&(n_y as i32))
                    .set_arg(&(n_z as i32))
                    .set_global_work_sizes(global_size.as_slice().unwrap())
                    .set_local_work_sizes(workgroup_size.as_slice().unwrap())
                    .set_event_wait_list(&wait_update_H)
                    .enqueue_nd_range(&queue)?;
                let ev_update_H = ExecuteKernel::new(&kernel_update_H)
                    .set_arg(&E_gpu)
                    .set_arg(&H_gpu)
                    .set_arg(&cE_gpu)
                    .set_arg(&b0_cpu)
                    .set_arg(&(n_x as i32))
                    .set_arg(&(n_y as i32))
                    .set_arg(&(n_z as i32))
                    .set_global_work_sizes(global_size.as_slice().unwrap())
                    .set_local_work_sizes(workgroup_size.as_slice().unwrap())
                    .set_event_wait_list(&[ev_update_E.get()])
                    .enqueue_nd_range(&queue)?;
                last_ev_update_H = Some(ev_update_H);

                if curr_iter % RECORD_STRIDE == 0 {
                    let index = E_out_readback_index; 
                    let E_out_readback = &mut E_out_readbacks[index];
                    E_out_readback_index = (E_out_readback_index+1) % TOTAL_READBACK_BUFFERS;
                    let timer = Instant::now();
                    E_out_readback.acquire();
                    println!("[main][{0}] Waited for readback buffer {1} for {2} us", curr_iter, index, timer.elapsed().as_micros());
                    let ev_copy = { 
                        let mut gpu_buffer = E_out_readback.gpu_buffer.lock().unwrap();
                        let cpu_buffer = E_out_readback.cpu_buffer.lock().unwrap();
                        let size = cpu_buffer.len();
                        drop(cpu_buffer);
                        queue.enqueue_copy_buffer(&E_gpu, &mut gpu_buffer, 0, 0, size, &[ev_update_E.get()]) 
                    }?;
                    E_out_readback.make_busy(ev_copy, curr_iter);
                }

                queue.flush()?;
            }

        }
        queue.finish()?;
    }
    {
        let elapsed = global_timer.elapsed();
        let elapsed_secs: f64 = (elapsed.as_nanos() as f64)*1e-9;
        let cell_rate = ((total_cells * TOTAL_LOOPS) as f64)/elapsed_secs * 1e-6;
        println!("total_cells={0}", total_cells);
        println!("total_loops={0}", TOTAL_LOOPS);
        println!("cell_rate={0:.3} M/s", cell_rate);
    }

    E_out_readbacks.iter().for_each(|x| {
        x.acquire();
        x.close();
    });
    pool.join();

    Ok(())
}

