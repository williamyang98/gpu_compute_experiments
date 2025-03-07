use log::{info, debug};
use std::sync::{Arc, Mutex};
use std::rc::Rc;
use std::cell::RefCell;
use std::time::Instant;
use std::sync::mpsc::channel;
use threadpool::ThreadPool;
use opencl3::{
    command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE}, 
    context::Context, device::Device, 
    error_codes::ClError, 
    event::Event,
};
use ndarray::{Array1,ArrayView4, s};
use super::{
    simulation::{Simulation, SimulationCpuData},
    constants as C,
    chrome_trace::{TraceSpan, TraceEvent},
    readback::{ReadbackData, ReadbackBufferArray, ReadbackHandlerFactory},
};

#[derive(Default)]
pub struct TraceFieldReadback {
    pub e_field_copy: Vec<TraceSpan>,
    pub e_field_read: Vec<TraceSpan>,
}

#[derive(Default)]
pub struct TraceSimulationStep {
    pub update_e_field: Vec<TraceSpan>,
    pub update_h_field: Vec<TraceSpan>,
}

#[derive(Default)]
pub struct AppGpuTrace {
    pub steps: Arc<Mutex<TraceSimulationStep>>,
    pub readbacks: Rc<RefCell<Vec<Arc<Mutex<TraceFieldReadback>>>>>,
}

impl AppGpuTrace {
    pub fn get_chrome_events(&self) -> Vec<TraceEvent> {
        let mut events = vec![];
        for (thread_id, trace_readback) in self.readbacks.borrow().iter().enumerate() {
            let trace_readback = trace_readback.lock().unwrap();
            for span in trace_readback.e_field_copy.iter() {
                events.push(TraceEvent {
                    name: "copy".to_owned(),
                    process_id: 0,
                    thread_id: thread_id as u64 + 1,
                    us_start: span.start.as_micros(),
                    us_duration: span.elapsed().as_micros(),
                    category: "X".to_owned(),
                })
            }
            for span in trace_readback.e_field_read.iter() {
                events.push(TraceEvent {
                    name: "read".to_owned(),
                    process_id: 0,
                    thread_id: thread_id as u64 + 1,
                    us_start: span.start.as_micros(),
                    us_duration: span.elapsed().as_micros(),
                    category: "X".to_owned(),
                })
            }
        }

        let steps = self.steps.lock().unwrap();
        for span in steps.update_e_field.iter() {
            events.push(TraceEvent {
                name: "update_e".to_owned(),
                process_id: 0,
                thread_id: 0,
                us_start: span.start.as_micros(),
                us_duration: span.elapsed().as_micros(),
                category: "X".to_owned(),
            });
        }
        for span in steps.update_h_field.iter() {
            events.push(TraceEvent {
                name: "update_h".to_owned(),
                process_id: 0,
                thread_id: 0,
                us_start: span.start.as_micros(),
                us_duration: span.elapsed().as_micros(),
                category: "X".to_owned(),
            });
        }
        events
    }
}

pub struct App {
    device: Arc<Device>,
    context: Arc<Context>,
    grid_size: Array1<usize>,
    cpu_data: SimulationCpuData,
    simulation: Simulation,
    pub gpu_trace: AppGpuTrace,
    e_field_readback_array: ReadbackBufferArray<f32>,
    thread_pool: ThreadPool,
}

impl App {
    pub fn new(device: Arc<Device>) -> Result<Self, ClError> {
        let context = Arc::new(Context::from_device(&device)?);

        let grid_size = Array1::from(vec![16, 256, 512]);
        let ns_per_tick: u64 = device.profiling_timer_resolution()? as u64;
        let n_dims = 3;
        let (n_x, n_y, n_z) = (grid_size[0], grid_size[1], grid_size[2]);
        let total_cells: usize = grid_size.iter().product();

        let cpu_data = SimulationCpuData::new(grid_size.clone());
        let simulation = Simulation::new(grid_size.clone(), &context)?;

        const TOTAL_READBACK_BUFFERS: usize = 3;
        let gpu_trace = AppGpuTrace::default();
        gpu_trace.readbacks.borrow_mut().resize_with(TOTAL_READBACK_BUFFERS, || {
            Arc::new(Mutex::new(TraceFieldReadback::default()))
        });

        let create_readback_handler: ReadbackHandlerFactory<f32> = Box::new({
            let trace_readbacks = gpu_trace.readbacks.clone();
            move |thread_id: usize| Box::new({
                let trace_readbacks = trace_readbacks.borrow_mut();
                let trace_readback = trace_readbacks[thread_id].clone();
                move |ev_copy: Event, ev_read: Event, _thread_id: usize, curr_iter: usize, data: &[f32]| {
                    debug!("Received data from thread={0}, iter={1}, data_size={2}", thread_id, curr_iter, data.len());
                    use ndarray_npy::write_npy;
                    let data = ArrayView4::from_shape((n_x,n_y,n_z,n_dims), data).unwrap();
                    write_npy(format!("./data/E_cpu_{0}.npy", curr_iter), &data).unwrap();

                    let trace_copy = TraceSpan::from_event(&ev_copy, ns_per_tick).unwrap();
                    let trace_read = TraceSpan::from_event(&ev_read, ns_per_tick).unwrap();

                    let mut trace_readback = trace_readback.lock().unwrap();
                    trace_readback.e_field_copy.push(trace_copy);
                    trace_readback.e_field_read.push(trace_read);
                }
            })
        });
        let e_field_readback_array = ReadbackBufferArray::<f32>::new(context.clone(), total_cells*n_dims, TOTAL_READBACK_BUFFERS, Some(create_readback_handler))?;

        let thread_pool = ThreadPool::new(8);

        Ok(Self {
            device,
            context,
            grid_size,
            cpu_data,
            simulation,
            gpu_trace,
            e_field_readback_array,
            thread_pool,
        })
    }

    pub fn init_simulation_data(&mut self) {
        let data = &mut self.cpu_data;
        let grid_size = &data.grid_size;
        let (n_x, n_y, n_z) = (grid_size[0], grid_size[1], grid_size[2]);
        let d_xyz: f32 = 1e-3;
        let dt: f32 = 1e-12;

        data.d_xyz = d_xyz;
        data.dt = dt;
        data.sigma_k.fill(0.0);
        data.e_field.fill(0.0);
        data.h_field.fill(0.0);
        data.e_k.fill(C::E_0);
        data.mu_k.fill(C::MU_0);
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

        data.bake_constants();
    }

    pub fn upload_simulation_data(&mut self) -> Result<(), ClError> {
        debug!("Uploading initial simulation conditions");
        let queue = CommandQueue::create_default(&self.context, 0)?;
        self.simulation.upload_data(&queue, &self.cpu_data)?;
        queue.finish()?;
        Ok(())
    }
 
    pub fn run(&mut self) -> Result<(), ClError> {
        let grid_size = &self.grid_size;
        let ns_per_tick: u64 = self.device.profiling_timer_resolution()? as u64;
        let total_cells: usize = grid_size.iter().product();

        // convert step events to trace points
        let (tx_step_events, rx_step_events) = channel::<(Event, Event)>();
        self.thread_pool.execute({
            let trace = self.gpu_trace.steps.clone();
            move || {
                for (ev_update_e_field, ev_update_h_field) in rx_step_events {
                    ev_update_e_field.wait().unwrap();
                    ev_update_h_field.wait().unwrap();

                    let trace_update_e_field = TraceSpan::from_event(&ev_update_e_field, ns_per_tick).unwrap();
                    let trace_update_h_field = TraceSpan::from_event(&ev_update_h_field, ns_per_tick).unwrap();

                    let mut trace = trace.lock().unwrap();
                    trace.update_e_field.push(trace_update_e_field);
                    trace.update_h_field.push(trace_update_h_field);
                }
            }
        });

        const TOTAL_STEPS: usize = 8192;
        const RECORD_STRIDE: usize = 32;
        const IS_RECORD: bool = true;

        // calculate workgroup size and global size for program
        let max_workgroup_threads: usize = self.device.max_work_group_size()?;
        let workgroup_size = Array1::from(vec![1, 1, 256]);
        let total_workgroup_size: usize = workgroup_size.iter().product();
        assert!(total_workgroup_size <= max_workgroup_threads);

        // let queue_props = 0;
        let queue_props = CL_QUEUE_PROFILING_ENABLE;
        let queue = CommandQueue::create_default(&self.context, queue_props)?;
        let global_timer = Instant::now();
        for curr_iter in 0..TOTAL_STEPS {
            let [ev_update_e_field, ev_update_h_field] = self.simulation.step(&queue, workgroup_size.clone(), &[])?;
            if curr_iter % RECORD_STRIDE == 0 && IS_RECORD {
                // grab available buffer
                let buffer = self.e_field_readback_array.get_free_buffer();
                // readback on gpu
                let mut readback_buffer = buffer.value.lock().unwrap();
                let size = readback_buffer.data_cpu.len();
                let ev_copy = unsafe { queue.enqueue_copy_buffer(&self.simulation.e_field, &mut readback_buffer.data_gpu, 0, 0, size, &[ev_update_h_field.get()]) }?;
                buffer.signal_pending(Some(ReadbackData { ev_copy, curr_iter }));
            }
            unsafe { queue.enqueue_barrier_with_wait_list(&[ev_update_h_field.get()]) }?;
            tx_step_events.send((ev_update_e_field, ev_update_h_field)).unwrap();
            queue.flush()?;
        }
        queue.finish()?;
        // benchmark performance
        let elapsed = global_timer.elapsed();
        let elapsed_secs: f64 = (elapsed.as_nanos() as f64)*1e-9;
        let cell_rate = ((total_cells * TOTAL_STEPS) as f64)/elapsed_secs * 1e-6;
        info!("total_cells={0}", total_cells);
        info!("total_loops={0}", TOTAL_STEPS);
        info!("cell_rate={0:.3} M/s", cell_rate);

        self.e_field_readback_array.join();
        drop(tx_step_events);
        self.thread_pool.join();

        Ok(())
    }
}
