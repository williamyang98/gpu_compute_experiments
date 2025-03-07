use log::{info, debug};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use std::{ffi::c_void, ptr::null_mut};
use crossbeam_channel::bounded;
use threadpool::ThreadPool;
use opencl3::{
    command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE}, 
    context::Context, device::Device, 
    error_codes::ClError, 
    event::Event,
    types::cl_bool,
    memory::{Buffer, CL_MEM_READ_WRITE},
};
use ndarray::{Array1,ArrayView4, s};
use super::{
    simulation::{Simulation, SimulationCpuData},
    constants as C,
    chrome_trace::{TraceSpan, TraceEvent},
};

struct ReadbackBuffer<T> {
    data_cpu: Vec<T>,
    data_gpu: Buffer<T>,
}

impl<T: Sized + Default + Clone> ReadbackBuffer<T> {
    fn new(context: &Context, size: usize) -> Result<Self, ClError> {
        let data_cpu: Vec<T> = vec![T::default(); size];
        let data_gpu = unsafe { Buffer::<T>::create(context, CL_MEM_READ_WRITE, size, null_mut::<c_void>()) }?;
        Ok(Self {
            data_cpu,
            data_gpu,
        })
    }
}

struct ReadbackRequest {
    ev_copy: Event,
    curr_iter: usize,
}

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
    pub readbacks: Vec<Arc<Mutex<TraceFieldReadback>>>,
}

impl AppGpuTrace {
    pub fn get_chrome_events(&self) -> Vec<TraceEvent> {
        let mut events = vec![];
        for (thread_id, trace_readback) in self.readbacks.iter().enumerate() {
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
    e_field_readback_array: Vec<Arc<Mutex<ReadbackBuffer<f32>>>>,
    thread_pool: ThreadPool,
}

impl App {
    pub fn new(device: Arc<Device>) -> Result<Self, ClError> {
        let context = Arc::new(Context::from_device(&device)?);

        let grid_size = Array1::from(vec![16, 256, 512]);
        let n_dims = 3;
        let total_cells: usize = grid_size.iter().product();

        let cpu_data = SimulationCpuData::new(grid_size.clone());
        let simulation = Simulation::new(grid_size.clone(), &context)?;

        let mut gpu_trace = AppGpuTrace::default();
        gpu_trace.readbacks.resize_with(TOTAL_READBACK_BUFFERS, || {
            Arc::new(Mutex::new(TraceFieldReadback::default()))
        });

        const TOTAL_READBACK_BUFFERS: usize = 3;
        let e_field_readback_array = (0..TOTAL_READBACK_BUFFERS)
            .map(|_| Arc::new(Mutex::new(ReadbackBuffer::<f32>::new(&context, total_cells*n_dims).unwrap())))
            .collect();

        // readback threads, event processing from main thread
        let thread_pool = ThreadPool::new(TOTAL_READBACK_BUFFERS + 1);

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
        let total_cells: usize = grid_size.iter().product();

        let n_dims = 3;
        let ns_per_tick: u64 = self.device.profiling_timer_resolution()? as u64;
        let (n_x, n_y, n_z) = (grid_size[0], grid_size[1], grid_size[2]);

        // convert step events to trace points
        let (tx_step_events, rx_step_events) = bounded::<(Event, Event)>(128);
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

        // read data from gpu to cpu
        let total_readback_buffers = self.e_field_readback_array.len();
        let mut tx_readback_requests = vec![];

        let (tx_readback_available, rx_readback_available) = bounded::<usize>(total_readback_buffers);
        for (thread_id, buffer) in self.e_field_readback_array.iter().enumerate() {
            self.thread_pool.execute({
                let tx_readback_available = tx_readback_available.clone();
                let (tx_readback_request, rx_readback_request) = bounded::<ReadbackRequest>(1);
                tx_readback_requests.push(tx_readback_request);
                let buffer_mutex = buffer.clone();
                let trace_readback = self.gpu_trace.readbacks[thread_id].clone();
                let queue_props = CL_QUEUE_PROFILING_ENABLE;
                let queue = CommandQueue::create_default(&self.context, queue_props).unwrap();
                move || {
                    for data in rx_readback_request {
                        let curr_iter = data.curr_iter;
                        let mut buffer_lock = buffer_mutex.lock().unwrap();
                        let buffer = &mut *buffer_lock;
                        let total_elems = buffer.data_cpu.len();

                        let ev_read = unsafe {
                            queue.enqueue_read_buffer(
                                &buffer.data_gpu,
                                false as cl_bool, 0, 
                                buffer.data_cpu.as_mut_slice(),
                                &[data.ev_copy.get()],
                            ).unwrap()
                        };
                        ev_read.wait().unwrap();

                        debug!("Received data from thread={0}, iter={1}, data_size={2}", thread_id, curr_iter, total_elems);
                        {
                            use ndarray_npy::write_npy;
                            let data = ArrayView4::from_shape((n_x,n_y,n_z,n_dims), buffer.data_cpu.as_slice()).unwrap();
                            write_npy(format!("./data/E_cpu_{0}.npy", curr_iter), &data).unwrap();
                        }

                        let trace_copy = TraceSpan::from_event(&data.ev_copy, ns_per_tick).unwrap();
                        let trace_read = TraceSpan::from_event(&ev_read, ns_per_tick).unwrap();

                        let mut trace_readback = trace_readback.lock().unwrap();
                        trace_readback.e_field_copy.push(trace_copy);
                        trace_readback.e_field_read.push(trace_read);

                        drop(buffer_lock);
                        let _is_main_thread_closed = tx_readback_available.send(thread_id).is_err();
                    }
                    queue.finish().unwrap();
                }
            })
        }
        // prepopulate queue with existing buffers
        for thread_id in 0..total_readback_buffers {
            tx_readback_available.send(thread_id).unwrap();
        }
        drop(tx_readback_available);

        const TOTAL_STEPS: usize = 2048;
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
                // enqueue gpu to gpu copy and let gpu to cpu happen in readback thread
                if let Ok(thread_id) = rx_readback_available.recv() {
                    let buffer_mutex = &self.e_field_readback_array[thread_id];
                    let mut buffer_lock = buffer_mutex.lock().unwrap();
                    let buffer = &mut *buffer_lock;
                    let ev_copy = unsafe { 
                        queue.enqueue_copy_buffer(
                            &self.simulation.e_field,
                            &mut buffer.data_gpu, 
                            0, 0, buffer.data_cpu.len(),
                            &[ev_update_h_field.get()]
                        ) 
                    }?;
                    drop(buffer_lock);
                    tx_readback_requests[thread_id].send(ReadbackRequest { ev_copy, curr_iter }).unwrap();
                }
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

        drop(tx_step_events);
        drop(tx_readback_requests);
        drop(rx_readback_available);
        self.thread_pool.join();

        Ok(())
    }
}
