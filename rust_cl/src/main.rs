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
    chrome_trace::{TraceSpan, Trace, TraceEvent},
    readback::{TraceFieldReadback, FieldReadbackBuffer, FieldReadbackData, FieldReadbackBufferArray},
};
use std::thread::JoinHandle;
use std::io::{BufWriter, Write};
use std::fs::File;

fn main() -> Result<(), String> {
    let chrome_trace = run().map_err(|err| err.to_string())?;
    let json_string = serde_json::to_string_pretty(&chrome_trace).unwrap();
    let file = File::create("./trace.json").unwrap();
    let mut writer = BufWriter::new(file);
    writer.write_all(json_string.as_bytes()).unwrap();
    writer.flush().unwrap();
    Ok(())
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

    let _platform = selected_platform.expect("No available opencl platform");
    let device = selected_device.expect("No available opencl device");
    let context = Arc::new(Context::from_device(&device)?);

    let ns_per_tick: u64 = device.profiling_timer_resolution()? as u64;
    let grid_size = Array1::from(vec![16, 256, 512]);

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
    const TOTAL_STEPS: usize = 8192;
    const RECORD_STRIDE: usize = 32;
    const IS_RECORD: bool = true;

    let mut e_field_readback_array = FieldReadbackBufferArray::new(context.clone(), grid_size.clone(), TOTAL_READBACK_BUFFERS)?;
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
                let mut buffer = e_field_readback_array.get_free_buffer();
                // readback on gpu
                let mut readback_buffer = buffer.value.lock().unwrap();
                let size = readback_buffer.data_cpu.len();
                let ev_copy = unsafe { queue.enqueue_copy_buffer(&simulation.e_field, &mut readback_buffer.data_gpu, 0, 0, size, &[ev_update_h_field.get()]) }?;
                buffer.signal_pending(Some(FieldReadbackData { ev_copy, curr_iter }));
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

    e_field_readback_array.join();

    let mut trace_grid_step = TraceGridStep::default();
    trace_grid_step.gpu_update_e_field.extend(
        evs_update_e_field.iter().map(|ev| TraceSpan::from_event(ev, ns_per_tick).unwrap())
    );
    trace_grid_step.gpu_update_h_field.extend(
        evs_update_h_field.iter().map(|ev| TraceSpan::from_event(ev, ns_per_tick).unwrap())
    );

    let mut chrome_trace = Trace::default();
    chrome_trace.events.extend(e_field_readback_array.get_chrome_trace_events());

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
        let i = s![0..n_x, 30..90, 30..40];
        data.sigma_k.slice_mut(i).fill(sigma_0);
    }

    {
        let w: usize = 10;
        let a: Array1<f32> = ndarray::linspace(0.0, 2.0*C::PI, w*2).collect();
        // hann window
        let a = 0.53836 - 0.46164*a.cos();
        let c = n_z/2;
        let i = s![5..=6,0..n_y,c-w..c+w,0];
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
