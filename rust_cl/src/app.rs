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
use ndarray::{Array1,ArrayView4,Array4,s};
use super::{
    simulation::{Simulation, SimulationCpuData},
    constants as C,
    chrome_trace::{TraceSpan, TraceEvent},
};

pub enum UserEvent {
    SetProgress { curr_step: usize, total_steps: usize },
    GridDownload { data: Arc<Mutex<Array4<f32>>>, thread_id: usize, curr_iter: usize },
}

pub struct ReadbackBuffer<T> {
    queue: CommandQueue,
    data_cpu: Vec<T>,
    data_gpu: Buffer<T>,
}

impl<T: Sized + Default + Clone> ReadbackBuffer<T> {
    fn new(context: &Context, size: usize) -> Result<Self, ClError> {
        let queue_props = CL_QUEUE_PROFILING_ENABLE;
        let queue = CommandQueue::create_default(context, queue_props).unwrap();
        let data_cpu: Vec<T> = vec![T::default(); size];
        let data_gpu = unsafe { Buffer::<T>::create(context, CL_MEM_READ_WRITE, size, null_mut::<c_void>()) }?;
        Ok(Self {
            queue,
            data_cpu,
            data_gpu,
        })
    }
}

impl<T> Drop for ReadbackBuffer<T> {
    fn drop(&mut self) {
        self.queue.finish().unwrap();
    }
}

struct ReadbackRequest {
    ev_copy: Event,
    ev_read: Event,
    curr_iter: usize,
}

enum SimulationEvent {
    ApplySignal { event: Event, curr_iter: usize },
    Step { update_e_field: Event, update_h_field: Event, curr_iter: usize },
}

#[derive(Default)]
pub struct TraceFieldReadback {
    pub e_field_copy: Vec<TraceSpan>,
    pub e_field_read: Vec<TraceSpan>,
}

#[derive(Default)]
pub struct TraceSimulationStep {
    pub apply_signal: Vec<TraceSpan>,
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
                    us_start: span.start.as_nanos(),
                    us_duration: span.elapsed().as_nanos(),
                    category: "X".to_owned(),
                })
            }
            for span in trace_readback.e_field_read.iter() {
                events.push(TraceEvent {
                    name: "read".to_owned(),
                    process_id: 0,
                    thread_id: thread_id as u64 + 1,
                    us_start: span.start.as_nanos(),
                    us_duration: span.elapsed().as_nanos(),
                    category: "X".to_owned(),
                })
            }
        }

        let steps = self.steps.lock().unwrap();
        for span in steps.apply_signal.iter() {
            events.push(TraceEvent {
                name: "apply_signal".to_owned(),
                process_id: 0,
                thread_id: 0,
                us_start: span.start.as_nanos(),
                us_duration: span.elapsed().as_nanos(),
                category: "X".to_owned(),
            });
        }
        for span in steps.update_e_field.iter() {
            events.push(TraceEvent {
                name: "update_e".to_owned(),
                process_id: 0,
                thread_id: 0,
                us_start: span.start.as_nanos(),
                us_duration: span.elapsed().as_nanos(),
                category: "X".to_owned(),
            });
        }
        for span in steps.update_h_field.iter() {
            events.push(TraceEvent {
                name: "update_h".to_owned(),
                process_id: 0,
                thread_id: 0,
                us_start: span.start.as_nanos(),
                us_duration: span.elapsed().as_nanos(),
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
    user_events: crossbeam_channel::Sender<UserEvent>,
}

impl App {
    pub fn new(device: Arc<Device>, user_events: crossbeam_channel::Sender<UserEvent>) -> Result<Self, ClError> {
        let context = Arc::new(Context::from_device(&device)?);

        let grid_size = Array1::from(vec![16, 256, 512]);
        let n_dims = 3;
        let total_cells: usize = grid_size.iter().product();

        let cpu_data = SimulationCpuData::new(grid_size.clone());
        let simulation = Simulation::new(grid_size.clone(), &context)?;

        let mut gpu_trace = AppGpuTrace::default();
        const TOTAL_READBACK_BUFFERS: usize = 1;
        gpu_trace.readbacks.resize_with(TOTAL_READBACK_BUFFERS, || {
            Arc::new(Mutex::new(TraceFieldReadback::default()))
        });

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
            user_events,
        })
    }

    pub fn init_simulation_data(&mut self) {
        let data = &mut self.cpu_data;
        let grid_size = &data.grid_size;
        let (n_x, n_y, n_z) = (grid_size[0], grid_size[1], grid_size[2]);
        let courant_number: f32 = 1.0/f32::sqrt(3.0) * 0.99;
        let d_xyz: f32 = 1e-5;
        // make sure each time step doesn't result in light travelling more than one grid scale (scaled to courant factor)
        // step_factor = (c*dt)/d_xyz
        // let dt: f32 = 1e-14; // produces a step factor of 0.3
        let dt = (courant_number * d_xyz)/C::C_0; 
        log::info!("Using a time step of dt={:.3e}", dt);

        data.d_xyz = d_xyz;
        data.dt = dt;
        data.sigma_k.fill(0.0);
        data.e_field.fill(0.0);
        data.h_field.fill(0.0);
        data.e_k.fill(C::E_0);
        data.mu_k = C::MU_0;

        // 16,256,512

        let border: usize = 40;
        let height: usize = 4;
        let width: usize = 10;
        let diff_spacing: usize = 5;
        let thickness: usize = 1;
        // add conductors
        if true {
            let sigma_0: f32 = 1e8;
            // ground plane
            let i = s![n_x/2-height/2-thickness..n_x/2-height/2, border..n_y-border, border..n_z-border];
            data.sigma_k.slice_mut(i).fill(sigma_0);
            // transmission line
            let i = s![n_x/2+height/2..n_x/2+height/2+thickness, n_y/2-width/2..n_y/2+width/2, border*2..n_z-border*2];
            // let i = s![14..=15, border..n_y-border, border..n_z-border];
            data.sigma_k.slice_mut(i).fill(sigma_0);

            // differential transmission line
            let i = s![n_x/2+height/2..n_x/2+height/2+thickness, n_y/2-width/2+width+diff_spacing..n_y/2+width/2+width+diff_spacing, border*2..n_z-border*2];
            data.sigma_k.slice_mut(i).fill(sigma_0);
        }

        // add dielectric
        if true {
            let border: usize = 40;
            let e_k = C::E_0*4.1;
            let i = s![n_x/2-height/2-2..n_x/2+height/2+2, border..n_y-border, border..n_z-border];
            data.e_k.slice_mut(i).fill(e_k);
        }

        // add termination resistor
        if true {
            let R: f32 = 41.6;
            // let R: f32 = 37.0;
            let l = (height as f32) * d_xyz;
            let A = ((width as f32)*d_xyz) * ((thickness as f32)*d_xyz);
            let sigma = l/(R*A);
            log::info!("Feedline has conductivity of {:.3e}", sigma);

            let z_start = border*2;
            let i = s![n_x/2-height/2..n_x/2+height/2, n_y/2-width/2..n_y/2+width/2, z_start..z_start+thickness];
            data.sigma_k.slice_mut(i).fill(sigma);
            let i = s![n_x/2-height/2..n_x/2+height/2, n_y/2-width/2+width+diff_spacing..n_y/2+width/2+width+diff_spacing, z_start..z_start+thickness];
            data.sigma_k.slice_mut(i).fill(sigma);

            let z_end = n_z-border*2;
            let i = s![n_x/2-height/2..n_x/2+height/2, n_y/2-width/2..n_y/2+width/2, z_end-thickness..z_end];
            data.sigma_k.slice_mut(i).fill(sigma);
            let i = s![n_x/2-height/2..n_x/2+height/2, n_y/2-width/2+width+diff_spacing..n_y/2+width/2+width+diff_spacing, z_end-thickness..z_end];
            data.sigma_k.slice_mut(i).fill(sigma);
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
 
    pub fn run(&mut self, total_steps: usize, record_stride: Option<usize>) -> Result<(), ClError> {
        let grid_size = &self.grid_size;
        let total_cells: usize = grid_size.iter().product();

        let n_dims = 3;
        let ns_per_tick: u64 = self.device.profiling_timer_resolution()? as u64;
        let (n_x, n_y, n_z) = (grid_size[0], grid_size[1], grid_size[2]);

        // convert step events to trace points
        let (tx_step_events, rx_step_events) = bounded::<SimulationEvent>(128);
        self.thread_pool.execute({
            let trace = self.gpu_trace.steps.clone();
            move || {
                for event in rx_step_events {
                    match event {
                        SimulationEvent::ApplySignal { event, curr_iter }  => {
                            event.wait().unwrap();
                            let trace_event = TraceSpan::from_event(&event, ns_per_tick).unwrap();
                            let mut trace = trace.lock().unwrap();
                            trace.apply_signal.push(trace_event);
                        },
                        SimulationEvent::Step { update_e_field, update_h_field, curr_iter } => {
                            update_e_field.wait().unwrap();
                            update_h_field.wait().unwrap();

                            let trace_update_e_field = TraceSpan::from_event(&update_e_field, ns_per_tick).unwrap();
                            let trace_update_h_field = TraceSpan::from_event(&update_h_field, ns_per_tick).unwrap();

                            let mut trace = trace.lock().unwrap();
                            trace.update_e_field.push(trace_update_e_field);
                            trace.update_h_field.push(trace_update_h_field);
                        },
                    }
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

                let grid_shape = (n_x,n_y,n_z,n_dims);
                let grid_data = Arc::new(Mutex::new(Array4::<f32>::zeros(grid_shape)));

                let user_events = self.user_events.clone();
                move || {
                    for data in rx_readback_request {
                        data.ev_copy.wait().unwrap();
                        data.ev_read.wait().unwrap();

                        let curr_iter = data.curr_iter;
                        let mut buffer_lock = buffer_mutex.lock().unwrap();
                        let buffer = &mut *buffer_lock;
                        let total_elems = buffer.data_cpu.len();

                        debug!("Received data from thread={0}, iter={1}, data_size={2}", thread_id, curr_iter, total_elems);
                        {
                            let grid_view = ArrayView4::from_shape(grid_shape, buffer.data_cpu.as_slice()).unwrap();
                            grid_data.lock().unwrap().assign(&grid_view);
                            let _ = user_events.send(UserEvent::GridDownload { 
                                data: grid_data.clone(),
                                thread_id,
                                curr_iter: data.curr_iter,
                            });
                            // use ndarray_npy::write_npy;
                            // write_npy(format!("./data/E_cpu_{0}.npy", curr_iter), &data).unwrap();
                        }

                        let trace_copy = TraceSpan::from_event(&data.ev_copy, ns_per_tick).unwrap();
                        let trace_read = TraceSpan::from_event(&data.ev_read, ns_per_tick).unwrap();

                        let mut trace_readback = trace_readback.lock().unwrap();
                        trace_readback.e_field_copy.push(trace_copy);
                        trace_readback.e_field_read.push(trace_read);

                        drop(buffer_lock);
                        let _is_main_thread_closed = tx_readback_available.send(thread_id).is_err();
                    }
                }
            })
        }
        // prepopulate queue with existing buffers
        for thread_id in 0..total_readback_buffers {
            tx_readback_available.send(thread_id).unwrap();
        }
        drop(tx_readback_available);

        // calculate workgroup size and global size for program
        let max_workgroup_threads: usize = self.device.max_work_group_size()?;
        let workgroup_size = Array1::from(vec![1, 1, 256]);
        let total_workgroup_size: usize = workgroup_size.iter().product();
        assert!(total_workgroup_size <= max_workgroup_threads);

        let signal_length: usize = 256;
        let mut signal = Array1::<f32>::zeros(signal_length);
        for (i, v) in signal.iter_mut().enumerate() {
            let dt = 3.1415*(i as f32)/((signal_length-1) as f32);
            let w = dt.sin();
            let w = w*w;
            let a = 1.0;
            *v = w*a;
        }

        // let queue_props = 0;
        let queue_props = CL_QUEUE_PROFILING_ENABLE;
        let queue = CommandQueue::create_default(&self.context, queue_props)?;
        let global_timer = Instant::now();
        for curr_iter in 0..total_steps {
            if let Some(value) = signal.get(curr_iter).copied() {
                let border: usize = 40;
                let width: usize = 10;
                let height: usize = 4;
                let diff_spacing: usize = 5;
                let thickness: usize = 1;

                let offset: [usize;3] = [n_x/2-height/2, n_y/2-width/2, n_z-2*border-1];
                let size: [usize;3] = [height, width, thickness];
                let ev0 = self.simulation.apply_voltage_source(&queue, value, &offset, &size, &[])?;
                let offset: [usize;3] = [n_x/2-height/2, n_y/2-width/2+width+diff_spacing, n_z-2*border-1];
                let ev1 = self.simulation.apply_voltage_source(&queue, -value, &offset, &size, &[])?;
                unsafe { queue.enqueue_barrier_with_wait_list(&[ev0.get(), ev1.get()]) }?;
                tx_step_events.send(SimulationEvent::ApplySignal { event: ev0, curr_iter }).unwrap();
                tx_step_events.send(SimulationEvent::ApplySignal { event: ev1, curr_iter }).unwrap();
            }
            let [ev_update_e_field, ev_update_h_field] = self.simulation.step(&queue, workgroup_size.clone(), &[])?;
            let is_record = record_stride.map(|stride| curr_iter % stride == 0).unwrap_or(false);
            if is_record {
                // process results in readback thread
                if let Ok(thread_id) = rx_readback_available.recv() {
                    let mut buffer_lock = self.e_field_readback_array[thread_id].lock().unwrap();
                    let buffer = &mut *buffer_lock;
                    // copy gpu to gpu on separate context
                    let ev_copy = unsafe {
                        buffer.queue.enqueue_copy_buffer(
                            &self.simulation.e_field,
                            &mut buffer.data_gpu, 
                            0, 0, buffer.data_cpu.len() * std::mem::size_of::<f32>(),
                            &[ev_update_e_field.get()]
                        )
                    }?;
                    // TODO: Attempt to synchronise copy/step so that we don't have race condition during copy
                    //       Can we do this in a better way without stalling main thread??
                    unsafe { queue.enqueue_barrier_with_wait_list(&[ev_update_h_field.get()]) }?;
                    // copy gpu to cpu on separate context
                    let ev_read = unsafe {
                        buffer.queue.enqueue_read_buffer(
                            &buffer.data_gpu,
                            false as cl_bool, 0, 
                            buffer.data_cpu.as_mut_slice(),
                            &[ev_copy.get()],
                        )
                    }?;
                    unsafe { buffer.queue.enqueue_barrier_with_wait_list(&[ev_read.get()]) }?;
                    tx_readback_requests[thread_id].send(
                        ReadbackRequest {
                            ev_copy,
                            ev_read,
                            curr_iter,
                        }
                    ).unwrap();
                }
            } else {
                unsafe { queue.enqueue_barrier_with_wait_list(&[ev_update_h_field.get()]) }?;
            }
            tx_step_events.send(SimulationEvent::Step {
                update_e_field: ev_update_e_field, 
                update_h_field: ev_update_h_field,
                curr_iter
            }).unwrap();
            queue.flush()?;
            let _ = self.user_events.send(UserEvent::SetProgress {
                curr_step: curr_iter+1,
                total_steps,
            });

        }
        queue.finish()?;
        // benchmark performance
        let elapsed = global_timer.elapsed();
        let elapsed_secs: f64 = (elapsed.as_nanos() as f64)*1e-9;
        let step_rate = ((total_steps) as f64)/elapsed_secs;
        let cell_rate = ((total_cells * total_steps) as f64)/elapsed_secs * 1e-6;
        info!("total_cells={0}", total_cells);
        info!("total_loops={0}", total_steps);
        info!("step_rate={0:.1} steps/s", step_rate);
        info!("cell_rate={0:.3} M/s", cell_rate);

        drop(tx_step_events);
        drop(tx_readback_requests);
        drop(rx_readback_available);
        self.thread_pool.join();

        Ok(())
    }
}

