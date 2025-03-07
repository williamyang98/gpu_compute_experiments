use std::{ffi::c_void, ptr::null_mut};
use std::sync::Arc;
use opencl3::{
    command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE}, 
    context::Context, 
    device::Device, 
    error_codes::ClError, 
    event::Event,
    memory::{Buffer, CL_MEM_READ_WRITE},
    types::cl_bool,
};
use ndarray::{Array1, Array4};
use super::{
    pool::{PoolChannel, PoolEntry},
};
use std::thread::JoinHandle;
use super::chrome_trace::{Trace, TraceSpan, TraceEvent};

#[derive(Default)]
pub struct TraceFieldReadback {
    pub thread_id: usize,
    pub gpu_copy: Vec<TraceSpan>,
    pub gpu_read: Vec<TraceSpan>,
}

pub struct FieldReadbackBuffer {
    pub grid_size: Array1<usize>,
    pub data_cpu: Array4<f32>,
    pub data_gpu: Buffer<f32>,
}

impl FieldReadbackBuffer {
    pub fn new(context: &Context, grid_size: Array1<usize>) -> Result<Self, ClError> {
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

pub struct FieldReadbackData {
    pub ev_copy: Event,
    pub curr_iter: usize,
}

pub struct FieldReadbackBufferArray {
    pub buffers: Vec<Arc<PoolEntry<FieldReadbackBuffer, Option<FieldReadbackData>>>>,
    join_handles: Vec<Option<JoinHandle<TraceFieldReadback>>>,
    pub traces_readback: Vec<TraceFieldReadback>,
    read_index: usize,
    size: usize,
}

impl FieldReadbackBufferArray {
    pub fn new(context: Arc<Context>, grid_size: Array1<usize>, size: usize) -> Result<Self, ClError> {
        assert!(size > 0);
        // create buffers
        let mut buffers: Vec<Arc<PoolEntry<FieldReadbackBuffer, Option<FieldReadbackData>>>> = vec![];
        for _ in 0..size {
            let buffer = FieldReadbackBuffer::new(&context, grid_size.clone())?;
            let pool_entry = PoolEntry::new(buffer);
            buffers.push(Arc::new(pool_entry));
        }

        let device: Device = Device::from(context.devices()[0]);
        let ns_per_tick: u64 = device.profiling_timer_resolution()? as u64;

        // create readback thread for each buffer
        let mut join_handles: Vec<Option<JoinHandle<TraceFieldReadback>>> = vec![];
        for (index, buffer) in buffers.iter().enumerate() {
            let handle = std::thread::spawn({
                let buffer = buffer.clone();
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
                        buffer.wait_pending_or_close();

                        let mut lock_state = buffer.state.lock().unwrap();
                        let readback_data = match &mut *lock_state {
                            PoolChannel::Closed => break,
                            PoolChannel::Empty => panic!("Shouldn't get empty data"),
                            PoolChannel::Pending(readback_data) => readback_data.take().expect("Missing data in readback channel"),
                        };

                        // read from gpu to cpu
                        let mut lock_value = buffer.value.lock().unwrap();
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
                        buffer.signal_free();
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
            join_handles.push(Some(handle));
        }

        Ok(Self {
            buffers,
            join_handles,
            traces_readback: vec![],
            read_index: 0,
            size,
        })
    }

    pub fn get_free_buffer(&mut self) -> &mut Arc<PoolEntry<FieldReadbackBuffer, Option<FieldReadbackData>>> {
        let index = self.read_index; 
        self.read_index = (self.read_index+1) % self.size;
        let buffer = &mut self.buffers[index];
        buffer.wait_empty();
        buffer
    }

    pub fn join(&mut self) {
        self.buffers.iter().for_each(|buffer| {
            buffer.wait_empty();
            buffer.signal_close();
        });

        self.traces_readback.extend(self.join_handles
            .iter_mut()
            .filter_map(|handle| handle.take())
            .map(|handle| {
                handle.join().expect("readback thread should join gracefully")
            })
        );
    }

    pub fn get_chrome_trace_events(&self) -> Vec<TraceEvent> {
        let mut events = vec![];
        for trace_readback in self.traces_readback.iter() {
            for span in trace_readback.gpu_copy.iter() {
                events.push(TraceEvent {
                    name: "copy".to_owned(),
                    process_id: 0,
                    thread_id: trace_readback.thread_id as u64 + 1,
                    us_start: span.start.as_micros(),
                    us_duration: span.elapsed().as_micros(),
                    category: "X".to_owned(),
                })
            }
            for span in trace_readback.gpu_read.iter() {
                events.push(TraceEvent {
                    name: "read".to_owned(),
                    process_id: 0,
                    thread_id: trace_readback.thread_id as u64 + 1,
                    us_start: span.start.as_micros(),
                    us_duration: span.elapsed().as_micros(),
                    category: "X".to_owned(),
                })
            }
        }
        events
    }
}



