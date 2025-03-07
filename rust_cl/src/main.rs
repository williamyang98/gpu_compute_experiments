use std::{ffi::c_void, ptr::null_mut};
use std::sync::Arc;
use std::time::{Instant, Duration};
use opencl3::{
    command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE}, 
    context::Context, device::Device, 
    error_codes::ClError, 
    event::Event,
    memory::{Buffer, CL_MEM_READ_WRITE},
    platform::get_platforms, 
    types::cl_bool,
};
use ndarray::{Array1, Array4, s};
use ytdlp_server::{
    simulation::{Simulation, SimulationCpuData},
    constants as C,
    pool::{PoolChannel, PoolEntry},
};
use std::thread::JoinHandle;
use std::io::{BufWriter, Write};
use std::fs::File;
use ytdlp_server::chrome_trace::{Trace, TraceEvent};

fn main() -> Result<(), String> {
    let chrome_trace = run().map_err(|err| err.to_string())?;
    let json_string = serde_json::to_string_pretty(&chrome_trace).unwrap();
    let file = File::create("./trace.json").unwrap();
    let mut writer = BufWriter::new(file);
    writer.write_all(json_string.as_bytes()).unwrap();
    writer.flush().unwrap();
    Ok(())
}

struct FieldReadbackBuffer {
    grid_size: Array1<usize>,
    data_cpu: Array4<f32>,
    data_gpu: Buffer<f32>,
}

impl FieldReadbackBuffer {
    fn new(context: &Context, grid_size: Array1<usize>) -> Result<Self, ClError> {
        let n_dims = 3;
        let (n_x, n_y, n_z) = (grid_size[0], grid_size[1], grid_size[2]);
        let data_cpu = Array4::<f32>::zeros((n_x, n_y, n_z, n_dims));
        let data_gpu = unsafe { Buffer::<f32>::create(&context, CL_MEM_READ_WRITE, data_cpu.len(), null_mut::<c_void>()) }?;
        Ok(Self {
            grid_size,
            data_cpu,
            data_gpu,
        })
    }
}

struct FieldReadbackData {
    ev_copy: Event,
    curr_iter: usize,
}

#[derive(Default, Clone, Copy)]
struct TraceSpan {
    start: Duration,
    end: Duration,
}

impl TraceSpan {
    fn elapsed(&self) -> Duration {
        self.end - self.start
    }
}

impl TraceSpan {
    fn from_event(ev: &Event, ns_per_tick: u64) -> Result<Self, ClError> {
        let ticks_start: u64 = ev.profiling_command_start()?;
        let ticks_end: u64 = ev.profiling_command_end()?;
        let ns_start = ticks_start*ns_per_tick;
        let ns_end = ticks_end*ns_per_tick;
        let start = Duration::from_nanos(ns_start);
        let end = Duration::from_nanos(ns_end);
        Ok(Self{ start, end })
    }
}

#[derive(Default)]
struct TraceFieldReadback {
    thread_id: usize,
    gpu_copy: Vec<TraceSpan>,
    gpu_read: Vec<TraceSpan>,
}

#[derive(Default)]
struct TraceGridStep {
    gpu_update_e_field: Vec<TraceSpan>,
    gpu_update_h_field: Vec<TraceSpan>,
}

fn run() -> Result<Trace, ClError> {
    let global_timer = Instant::now();
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

    let ns_per_tick: u64 = device.profiling_timer_resolution()? as u64;
    let n_dims: usize = 3;
    let grid_size = Array1::from(vec![16, 256, 512]);
    let (n_x, n_y, n_z) = (grid_size[0], grid_size[1], grid_size[2]);

    let mut simulation_data = SimulationCpuData::new(grid_size.clone());
    init_simulation_data(&mut simulation_data);
    simulation_data.bake_constants();

    let mut simulation = Simulation::new(grid_size.clone(), &context)?;

    {
        let queue = CommandQueue::create_default(&context, 0)?;
        simulation.upload_data(&queue, &simulation_data)?;
        queue.finish()?;
    }

    const TOTAL_READBACK_BUFFERS: usize = 3;
    const TOTAL_STEPS: usize = 512;
    const RECORD_STRIDE: usize = 32;
    const IS_RECORD: bool = true;

    let mut e_field_out_buffers: Vec<Arc<PoolEntry<FieldReadbackBuffer, Option<FieldReadbackData>>>> = vec![];
    for _ in 0..TOTAL_READBACK_BUFFERS {
        let readback_buffer = FieldReadbackBuffer::new(&context, grid_size.clone())?;
        let pool_entry = PoolEntry::new(readback_buffer);
        e_field_out_buffers.push(Arc::new(pool_entry));
    }

    let mut readback_threads: Vec<Option<JoinHandle<TraceFieldReadback>>> = vec![];
    for (index, e_field_out) in e_field_out_buffers.iter().enumerate() {
        let handle = std::thread::spawn({
            let e_field_out = e_field_out.clone();
            let context = context.clone();
            let mut trace_field_readback = TraceFieldReadback {
                thread_id: index,
                ..TraceFieldReadback::default()
            };
            let mut evs_copy: Vec<Event> = vec![];
            let mut evs_read: Vec<Event> = vec![];

            move || {
                let queue_props = CL_QUEUE_PROFILING_ENABLE;
                let queue = CommandQueue::create_default(&context, queue_props).unwrap();

                loop {
                    // wait for gpu to gpu read
                    e_field_out.wait_pending_or_close();

                    let mut lock_state = e_field_out.state.lock().unwrap();
                    let readback_data = match &mut *lock_state {
                        PoolChannel::Closed => break,
                        PoolChannel::Empty => panic!("Shouldn't get empty data"),
                        PoolChannel::Pending(readback_data) => readback_data.take().expect("Missing data in readback channel"),
                    };

                    // read from gpu to cpu
                    let mut lock_value = e_field_out.value.lock().unwrap();
                    let readback_buffer = &mut *lock_value;
                    let ev_read = unsafe { 
                        queue.enqueue_read_buffer(
                            &readback_buffer.data_gpu, 
                            false as cl_bool, 0, 
                            readback_buffer.data_cpu.as_slice_mut().unwrap(), 
                            &[readback_data.ev_copy.get()],
                        )
                    }.unwrap();

                    // wait for read to finish
                    ev_read.wait().unwrap();
                    queue.finish().unwrap();
                    evs_copy.push(readback_data.ev_copy);
                    evs_read.push(ev_read);
                    // dump cpu data
                    use ndarray_npy::write_npy;
                    write_npy(format!("./data/E_cpu_{0}.npy", readback_data.curr_iter), &readback_buffer.data_cpu).unwrap();
                    // mark gpu/cpu readback buffers as available
                    drop(lock_state);
                    e_field_out.signal_free();
                }
                trace_field_readback.gpu_copy.extend(
                    evs_copy.iter().map(|ev| TraceSpan::from_event(ev, ns_per_tick).unwrap())
                );
                trace_field_readback.gpu_read.extend(
                    evs_read.iter().map(|ev| TraceSpan::from_event(ev, ns_per_tick).unwrap())
                );
                trace_field_readback
            }
        });
        readback_threads.push(Some(handle));
    }

    let mut evs_update_e_field: Vec<Event> = vec![];
    let mut evs_update_h_field: Vec<Event> = vec![];
    evs_update_e_field.reserve(TOTAL_STEPS);
    evs_update_h_field.reserve(TOTAL_STEPS);

    {
        let max_workgroup_threads: usize = device.max_work_group_size()?;
        let workgroup_size = Array1::from(vec![1, 1, 256]);
        let total_workgroup_size: usize = workgroup_size.iter().product();
        assert!(total_workgroup_size <= max_workgroup_threads);


        // let queue_props = 0;
        let queue_props = CL_QUEUE_PROFILING_ENABLE;
        let queue = CommandQueue::create_default(&context, queue_props)?;
        let mut e_field_out_index: usize = 0;

        let mut trace_loop = TraceSpan::default();
        trace_loop.start = global_timer.elapsed();
        for curr_iter in 0..TOTAL_STEPS {
            let [ev_update_e_field, ev_update_h_field] = simulation.step(&queue, workgroup_size.clone(), &[])?;
            if curr_iter % RECORD_STRIDE == 0 && IS_RECORD {
                // grab available buffer
                let index = e_field_out_index; 
                e_field_out_index = (e_field_out_index+1) % TOTAL_READBACK_BUFFERS;
                let e_field_out = &mut e_field_out_buffers[index];
                e_field_out.wait_empty();
                // readback on gpu
                let mut readback_buffer = e_field_out.value.lock().unwrap();
                let size = readback_buffer.data_cpu.len();
                let ev_copy = unsafe { queue.enqueue_copy_buffer(&simulation.e_field, &mut readback_buffer.data_gpu, 0, 0, size, &[ev_update_h_field.get()]) }?;
                e_field_out.signal_pending(Some(FieldReadbackData { ev_copy, curr_iter }));
            }
            unsafe { queue.enqueue_barrier_with_wait_list(&[ev_update_h_field.get()]) }?;

            // trace step events
            evs_update_e_field.push(ev_update_e_field);
            evs_update_h_field.push(ev_update_h_field);
            queue.flush()?;
        }
        queue.finish()?;
        trace_loop.end = global_timer.elapsed();

        // benchmark performance
        {
            let elapsed = trace_loop.elapsed();
            let elapsed_secs: f64 = (elapsed.as_nanos() as f64)*1e-9;
            let total_cells: usize = simulation.grid_size.iter().product();
            let cell_rate = ((total_cells * TOTAL_STEPS) as f64)/elapsed_secs * 1e-6;
            println!("total_cells={0}", total_cells);
            println!("total_loops={0}", TOTAL_STEPS);
            println!("cell_rate={0:.3} M/s", cell_rate);
        }
    }
    // wait for readback threads to finish
    e_field_out_buffers.iter().for_each(|buffer| {
        buffer.wait_empty();
        buffer.signal_close();
    });

    let traces_readback: Vec<TraceFieldReadback> = readback_threads
        .iter_mut()
        .map(|handle| {
            let handle = handle.take().expect("spawned thread should produce a handle");
            handle.join().expect("readback thread should join gracefully")
        })
        .collect();
    let mut trace_grid_step = TraceGridStep::default();
    trace_grid_step.gpu_update_e_field.extend(
        evs_update_e_field.iter().map(|ev| TraceSpan::from_event(ev, ns_per_tick).unwrap())
    );
    trace_grid_step.gpu_update_h_field.extend(
        evs_update_h_field.iter().map(|ev| TraceSpan::from_event(ev, ns_per_tick).unwrap())
    );

    let mut chrome_trace = Trace::default();
    for trace_field_readback in traces_readback.iter() {
        for span in trace_field_readback.gpu_copy.iter() {
            chrome_trace.events.push(TraceEvent {
                name: "copy".to_owned(),
                process_id: 0,
                thread_id: trace_field_readback.thread_id as u64 + 1,
                us_start: span.start.as_micros(),
                us_duration: span.elapsed().as_micros(),
                category: "X".to_owned(),
            });
        }
        for span in trace_field_readback.gpu_read.iter() {
            chrome_trace.events.push(TraceEvent {
                name: "read".to_owned(),
                process_id: 0,
                thread_id: trace_field_readback.thread_id as u64 + 1,
                us_start: span.start.as_micros(),
                us_duration: span.elapsed().as_micros(),
                category: "X".to_owned(),
            });
        }
    }
    {
        for span in trace_grid_step.gpu_update_e_field.iter() {
            chrome_trace.events.push(TraceEvent {
                name: "update_e".to_owned(),
                process_id: 0,
                thread_id: 0,
                us_start: span.start.as_micros(),
                us_duration: span.elapsed().as_micros(),
                category: "X".to_owned(),
            });
        }
        for span in trace_grid_step.gpu_update_h_field.iter() {
            chrome_trace.events.push(TraceEvent {
                name: "update_h".to_owned(),
                process_id: 0,
                thread_id: 0,
                us_start: span.start.as_micros(),
                us_duration: span.elapsed().as_micros(),
                category: "X".to_owned(),
            });
        }
    }

    Ok(chrome_trace)
}

fn init_simulation_data(data: &mut SimulationCpuData) {
    let grid_size = &data.grid_size;
    let (n_x, n_y, n_z) = (grid_size[0], grid_size[1], grid_size[2]);
    let d_xyz: f32 = 1e-3;
    let dt: f32 = 1e-12;

    data.d_xyz = d_xyz;
    data.dt = dt;
    {
        let sigma_0: f32 = 1e8;
        let i = s![.., 30..90, 30..40];
        data.sigma_k.slice_mut(i).fill(sigma_0);
    }

    {
        let w: usize = 10;
        let a: Array1<f32> = ndarray::linspace(0.0, 2.0*C::PI, w*2).collect();
        // hann window
        let a = 0.53836 - 0.46164*a.cos();
        let c = n_z/2;
        let i = s![5..=6,..,c-w..c+w,0];
        data.e_field
            .slice_mut(i)
            .axis_iter_mut(ndarray::Axis(1))
            .for_each(|mut row| row.assign(&a));
        data.h_field
            .slice_mut(i)
            .axis_iter_mut(ndarray::Axis(1))
            .for_each(|mut row| row.assign(&a));
    }
}
