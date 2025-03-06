use std::{ffi::c_void, ptr::null_mut};
use opencl3::{
    command_queue::CommandQueue, 
    context::Context,
    error_codes::ClError, 
    event::Event,
    kernel::{ExecuteKernel, Kernel},
    memory::{Buffer, CL_MEM_READ_WRITE},
    program::Program,
    types::{cl_bool, cl_event},
};
use ndarray::{Array1, Array3, Array4};
use super::constants as C;

pub struct SimulationCpuData {
    pub grid_size: Array1<usize>,
    pub e_field: Array4<f32>,
    pub h_field: Array4<f32>,
    pub sigma_k: Array3<f32>,
    pub e_k: Array3<f32>,
    pub mu_k: Array3<f32>,
    pub dt: f32,
    pub d_xyz: f32,
    pub a0: Array3<f32>,
    pub a1: Array3<f32>,
    pub b0: Array3<f32>,
}

impl SimulationCpuData {
    pub fn new(grid_size: Array1<usize>) -> Self {
        let (n_x, n_y, n_z) = (grid_size[0], grid_size[1], grid_size[2]);
        let n_dims = 3;

        Self {
            grid_size,
            e_field: Array4::<f32>::zeros((n_x, n_y, n_z, n_dims)),
            h_field: Array4::<f32>::zeros((n_x, n_y, n_z, n_dims)),
            sigma_k: Array3::<f32>::zeros((n_x, n_y, n_z)),
            e_k: Array3::<f32>::from_elem((n_x, n_y, n_z), C::E_0),
            mu_k: Array3::<f32>::from_elem((n_x, n_y, n_z), C::MU_0),
            dt: 1e-12,
            d_xyz: 1e-3,
            a0: Array3::<f32>::zeros((n_x, n_y, n_z)),
            a1: Array3::<f32>::zeros((n_x, n_y, n_z)),
            b0: Array3::<f32>::zeros((n_x, n_y, n_z)),
        }
    }

    pub fn bake_constants(&mut self) {
        self.a0.assign(&(1.0/(1.0 + &self.sigma_k/&self.e_k * self.dt)));
        self.a1.assign(&(1.0/(&self.e_k * self.d_xyz) * self.dt));
        self.b0.assign(&(1.0/(&self.mu_k * self.d_xyz) * self.dt));
    }
}


pub struct Simulation {
    pub grid_size: Array1<usize>,
    pub e_field: Buffer<f32>,
    pub h_field: Buffer<f32>,
    pub a0: Buffer<f32>,
    pub a1: Buffer<f32>,
    pub b0: Buffer<f32>,
    pub program_src: &'static str,
    _program: Program,
    kernel_update_e_field: Kernel,
    kernel_update_h_field: Kernel,
}

impl Simulation {
    pub fn new(grid_size: Array1<usize>, context: &Context) -> Result<Self, ClError> {
        let n_dims: usize = 3;
        let total_cells: usize = grid_size.iter().product();

        let host_ptr = null_mut::<c_void>();
        let mem_flags = CL_MEM_READ_WRITE;

        let e_field = unsafe { Buffer::<f32>::create(&context, mem_flags, total_cells*n_dims, host_ptr) }?;
        let h_field = unsafe { Buffer::<f32>::create(&context, mem_flags, total_cells*n_dims, host_ptr) }?;
        let a0 = unsafe { Buffer::<f32>::create(&context, mem_flags, total_cells, host_ptr) }?;
        let a1 = unsafe { Buffer::<f32>::create(&context, mem_flags, total_cells, host_ptr) }?;
        let b0 = unsafe { Buffer::<f32>::create(&context, mem_flags, total_cells, host_ptr) }?;

        let program_src: &'static str = include_str!("./shader.cl");
        let mut program = Program::create_from_source(&context, program_src)?;
        program.build(context.devices(), "")?;
        let kernel_update_e_field = Kernel::create(&program, "update_E")?;
        let kernel_update_h_field = Kernel::create(&program, "update_H")?;

        Ok(Self {
            grid_size,
            e_field,
            h_field,
            a0,
            a1,
            b0,
            program_src,
            _program: program,
            kernel_update_e_field,
            kernel_update_h_field,
        })
    }

    pub fn upload_data(&mut self, queue: &CommandQueue, data: &SimulationCpuData) -> Result<(), ClError> {
        let is_block = false as cl_bool;
        let _ = unsafe { queue.enqueue_write_buffer(&mut self.e_field, is_block, 0, data.e_field.as_slice().unwrap(), &[]) }?;
        let _ = unsafe { queue.enqueue_write_buffer(&mut self.h_field, is_block, 0, data.h_field.as_slice().unwrap(), &[]) }?;
        let _ = unsafe { queue.enqueue_write_buffer(&mut self.a0, is_block, 0, data.a0.as_slice().unwrap(), &[]) }?;
        let _ = unsafe { queue.enqueue_write_buffer(&mut self.a1, is_block, 0, data.a1.as_slice().unwrap(), &[]) }?;
        let _ = unsafe { queue.enqueue_write_buffer(&mut self.a1, is_block, 0, data.a1.as_slice().unwrap(), &[]) }?;
        let _ = unsafe { queue.enqueue_write_buffer(&mut self.b0, is_block, 0, data.b0.as_slice().unwrap(), &[]) }?;
        Ok(())
    }

    pub fn step(&mut self, queue: &CommandQueue, workgroup_size: Array1<usize>, wait_events: &[cl_event]) -> Result<[Event; 2], ClError> {
        let dispatch_size = &self.grid_size / &workgroup_size;
        let global_size = &dispatch_size * &workgroup_size;

        let (n_x, n_y, n_z) = (self.grid_size[0], self.grid_size[1], self.grid_size[2]);
        // let n_dims: usize = 3;

        unsafe {
            // e-field update
            let ev_update_e_field = ExecuteKernel::new(&self.kernel_update_e_field)
                .set_arg(&mut self.e_field)
                .set_arg(&self.h_field)
                .set_arg(&self.a0)
                .set_arg(&self.a1)
                .set_arg(&(n_x as i32))
                .set_arg(&(n_y as i32))
                .set_arg(&(n_z as i32))
                .set_global_work_sizes(global_size.as_slice().unwrap())
                .set_local_work_sizes(workgroup_size.as_slice().unwrap())
                .set_event_wait_list(&wait_events)
                .enqueue_nd_range(&queue)?;
            // h-field update
            let ev_update_h_field = ExecuteKernel::new(&self.kernel_update_h_field)
                .set_arg(&self.e_field)
                .set_arg(&mut self.h_field)
                .set_arg(&self.b0)
                .set_arg(&(n_x as i32))
                .set_arg(&(n_y as i32))
                .set_arg(&(n_z as i32))
                .set_global_work_sizes(global_size.as_slice().unwrap())
                .set_local_work_sizes(workgroup_size.as_slice().unwrap())
                .set_event_wait_list(&[ev_update_e_field.get()])
                .enqueue_nd_range(&queue)?;
            Ok([ev_update_e_field, ev_update_h_field])
        }
    }
}
