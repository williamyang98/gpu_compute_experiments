use std::{ffi::c_void, ptr::null_mut};
use std::sync::Arc;
use opencl3::{
    command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE}, 
    context::Context, 
    error_codes::ClError, 
    event::Event,
    memory::{Buffer, CL_MEM_READ_WRITE},
    types::cl_bool,
};
use super::pool::{PoolChannel, PoolEntry};
use std::thread::JoinHandle;

pub struct ReadbackBuffer<T> {
    pub data_cpu: Vec<T>,
    pub data_gpu: Buffer<T>,
}

impl<T: Sized + Default + Clone> ReadbackBuffer<T> {
    pub fn new(context: &Context, size: usize) -> Result<Self, ClError> {
        let data_cpu: Vec<T> = vec![T::default(); size];
        let data_gpu = unsafe { Buffer::<T>::create(context, CL_MEM_READ_WRITE, size, null_mut::<c_void>()) }?;
        Ok(Self {
            data_cpu,
            data_gpu,
        })
    }
}

pub struct ReadbackData {
    pub ev_copy: Event,
    pub curr_iter: usize,
}

pub trait ReadbackHandler<T> {
    fn handle(&mut self, event_copy: Event, event_read: Event, thread_id: usize, curr_iter: usize, data: &[T]);
}

impl<T, F> ReadbackHandler<T> for F
where F: FnMut(Event, Event, usize, usize, &[T])
{
    fn handle(&mut self, event_copy: Event, event_read: Event, thread_id: usize, curr_iter: usize, data: &[T]) {
        self(event_copy, event_read, thread_id, curr_iter, data);
    }
}

pub struct ReadbackBufferArray<T> {
    buffers: Vec<Arc<PoolEntry<ReadbackBuffer<T>, Option<ReadbackData>>>>,
    join_handles: Vec<Option<JoinHandle<()>>>,
    read_index: usize,
}

pub type ReadbackHandlerFactory<T> = Box<dyn FnMut(usize) -> Box<dyn ReadbackHandler<T> + Send>>;

impl<T> ReadbackBufferArray<T> 
where
    T: Sized + Clone + Default + Send + 'static
{
    pub fn new(context: Arc<Context>, size: usize, total_buffers: usize, mut create_handler: Option<ReadbackHandlerFactory<T>>) -> Result<Self, ClError> {
        assert!(size > 0);
        assert!(total_buffers > 0);
        // create buffers
        let mut buffers: Vec<Arc<PoolEntry<ReadbackBuffer<T>, Option<ReadbackData>>>> = vec![];
        for _ in 0..total_buffers {
            let buffer = ReadbackBuffer::<T>::new(&context, size)?;
            let pool_entry = PoolEntry::new(buffer);
            buffers.push(Arc::new(pool_entry));
        }

        // create readback thread for each buffer
        let mut join_handles: Vec<Option<JoinHandle<()>>> = vec![];
        for (index, buffer) in buffers.iter().enumerate() {
            let handle = std::thread::spawn({
                let buffer = buffer.clone();
                let context = context.clone();
                let mut handler = None;
                if let Some(ref mut create_handler) = create_handler {
                    handler = Some(create_handler(index)); 
                }

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
                                readback_buffer.data_cpu.as_mut_slice(),
                                &[readback_data.ev_copy.get()],
                            )
                        }.unwrap();

                        // wait for read to finish
                        ev_read.wait().unwrap();
                        queue.finish().unwrap();
                        // dump cpu data
                        if let Some(ref mut handler) = handler {
                            handler.handle(readback_data.ev_copy, ev_read, index, readback_data.curr_iter, readback_buffer.data_cpu.as_slice());
                        }
                        // mark gpu/cpu readback buffers as available
                        drop(lock_state);
                        buffer.signal_free();
                    }
                }
            });
            join_handles.push(Some(handle));
        }

        Ok(Self {
            buffers,
            join_handles,
            read_index: 0,
        })
    }

    pub fn get_free_buffer(&mut self) -> &mut Arc<PoolEntry<ReadbackBuffer<T>, Option<ReadbackData>>> {
        let index = self.read_index; 
        self.read_index = (self.read_index+1) % self.buffers.len();
        let buffer = &mut self.buffers[index];
        buffer.wait_empty();
        buffer
    }

    pub fn join(&mut self) {
        self.buffers.iter().for_each(|buffer| {
            buffer.wait_empty();
            buffer.signal_close();
        });

        for handle in self.join_handles.iter_mut() {
            if let Some(handle) = handle.take() {
                handle.join().expect("readback thread should join gracefully")
            }
        }
    }
}
