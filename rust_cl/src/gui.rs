use ndarray::{Array1, Array4};
use std::{borrow::Cow, num::NonZero, sync::{Arc, Mutex}};
use super::app::UserEvent;
use threadpool::ThreadPool;
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct CopyParams {
    size_x: u32,
    size_y: u32,
    size_z: u32,
    copy_x: u32,
}

struct CopyGridToTextureShader {
    params: CopyParams,
    params_uniform: wgpu::Buffer,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    bind_group_layout: wgpu::BindGroupLayout,
    compute_pipeline: wgpu::ComputePipeline,
}

impl CopyGridToTextureShader {
    fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        let program_src = std::fs::read_to_string("./src/read_buffer_to_texture.wgsl").expect("Shader file should exist");
        let program_src = Cow::Borrowed(program_src.as_str());

        let params = CopyParams::default();
        let params_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("copy_grid_to_texture_params_uniform"),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
            size: std::mem::size_of::<CopyParams>() as u64,
        });
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("copy_grid_to_texture_shader_module"),
            source: wgpu::ShaderSource::Wgsl(program_src),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("copy_grid_to_texture_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("copy_grid_to_texture_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("copy_grid_to_texture_compute_pipeline"),
            layout: Some(&pipeline_layout),
            entry_point: Some("main"),
            module: &shader_module,
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            device,
            queue,
            params,
            params_uniform,
            bind_group_layout,
            compute_pipeline,
        }
    }

    fn create_compute_pass(&mut self, 
        encoder: &mut wgpu::CommandEncoder,
        grid_buffer: wgpu::BufferBinding,
        texture: &wgpu::TextureView,
        grid_size: Array1<usize>, copy_x: usize,
    ) {
        let workgroup_size: [usize; 2] = [1, 256];
        let dispatch_size = [
            (grid_size[1] as f32 / workgroup_size[0] as f32).ceil() as usize,
            (grid_size[2] as f32 / workgroup_size[1] as f32).ceil() as usize,
        ];
        self.params.size_x = grid_size[0] as u32;
        self.params.size_y = grid_size[1] as u32;
        self.params.size_z = grid_size[2] as u32;
        self.params.copy_x = copy_x as u32;
        self.queue.write_buffer(&self.params_uniform, 0, bytemuck::cast_slice(&[self.params]));

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("copy_grid_to_texture_bind_group"),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.params_uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(grid_buffer),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(texture),
                },
            ],
            layout: &self.bind_group_layout,
        });

        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("copy_grid_to_texture_compute_pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&self.compute_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(dispatch_size[0] as u32, dispatch_size[1] as u32, 1);
        drop(compute_pass);
    }
}

pub struct AppGui {
    name: String,
    age: usize,
    curr_step: usize,
    total_steps: usize,
    total_grid_downloads: usize,
    grid_data_iter: usize,
    grid_buffer: Arc<wgpu::Buffer>,
    grid_texture: Arc<wgpu::Texture>,
    grid_texture_view: Arc<wgpu::TextureView>,
    grid_texture_id: egui::TextureId,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    copy_shader: CopyGridToTextureShader,
}

impl AppGui {
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>, renderer: &mut egui_wgpu::Renderer) -> Self {
        let grid_size = [16, 256, 512, 3];
        let total_elems: usize = grid_size.iter().product();
        let grid_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("grid_buffer"),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
            size: (total_elems * std::mem::size_of::<f32>()) as u64,
        });
        let grid_buffer = Arc::new(grid_buffer);

        let grid_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("grid_texture"),
            size: wgpu::Extent3d { width: grid_size[2] as u32, height: grid_size[1] as u32, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let grid_texture = Arc::new(grid_texture);
        let grid_texture_view = Arc::new(grid_texture.create_view(&wgpu::TextureViewDescriptor::default()));
        let grid_texture_id = renderer.register_native_texture(&device, &grid_texture_view, wgpu::FilterMode::Nearest);

        let copy_shader = CopyGridToTextureShader::new(device.clone(), queue.clone());

        Self {
            name: "Arthur".to_owned(),
            age: 42,
            curr_step: 0,
            total_steps: 0,
            total_grid_downloads: 0,
            grid_data_iter: 0,
            grid_buffer,
            grid_texture,
            grid_texture_view,
            grid_texture_id,
            device,
            queue,
            copy_shader,
        }
    }

    pub fn on_command_encoder(&mut self, encoder: &mut wgpu::CommandEncoder, renderer: &mut egui_wgpu::Renderer) {
        let grid_size = Array1::<usize>::from(vec![16, 256, 512]);
        let copy_x = grid_size[0]/2;
        self.copy_shader.create_compute_pass(
            encoder,
            self.grid_buffer.as_entire_buffer_binding(),
            &self.grid_texture_view,
            grid_size,
            copy_x,
        );
        renderer.update_egui_texture_from_wgpu_texture(&self.device, &self.grid_texture_view, wgpu::FilterMode::Linear, self.grid_texture_id);
    }

    pub fn render(&mut self, context: &egui::Context) {
        egui::CentralPanel::default().show(context, |ui| {
            ui.heading("My egui Application");
            ui.horizontal(|ui| {
                let name_label = ui.label("Your name: ");
                ui.text_edit_singleline(&mut self.name)
                    .labelled_by(name_label.id);
            });
            ui.add(egui::Slider::new(&mut self.age, 0..=120).text("age"));
            if ui.button("Increment").clicked() {
                self.age += 1;
            }
            ui.label(format!("Hello '{0}', age {1}", self.name.as_str(), self.age));

            ui.add_enabled_ui(self.total_steps > 0, |ui| {
                let progress = (self.curr_step as f32)/(self.total_steps.max(1) as f32);
                let progress_bar = egui::ProgressBar::new(progress)
                    .text(format!("{0}/{1} ({2:.2}%)", self.curr_step, self.total_steps, progress*100.0))
                    .desired_width(ui.available_width())
                    .desired_height(ui.spacing().interact_size.y);
                ui.add(progress_bar);
            });

            ui.label(format!("Total grid downloads: {0}", self.total_grid_downloads));
            ui.label(format!("Grid iter: {0}", self.grid_data_iter));
            egui::Frame::new()
                .fill(egui::Color32::WHITE)
                .stroke(egui::Stroke::new(1.0, egui::Color32::BLACK))
                .show(ui, |ui| {
                    ui.image(egui::ImageSource::Texture(egui::load::SizedTexture {
                        id: self.grid_texture_id,
                        size: egui::Vec2 { x: 512.0, y: 256.0 },
                    }));
                });
        });
    }

    pub fn handle_user_event(&mut self, event: UserEvent) -> bool {
        match event {
            UserEvent::SetProgress { curr_step, total_steps } => {
                self.curr_step = curr_step;
                self.total_steps = total_steps;
                true
            },
            UserEvent::GridDownload { data, thread_id: _thread_id, curr_iter } => {
                if curr_iter <= self.grid_data_iter {
                    return false;
                }
                let dst_view = self.queue.write_buffer_with(
                    &self.grid_buffer,
                    0,
                    NonZero::<u64>::new(self.grid_buffer.size()).unwrap(),
                );
                let Some(mut dst_view) = dst_view else {
                    log::error!("Failed to acquire grid buffer for writing: curr_iter={0}", curr_iter);
                    return false;
                };
                let src_view = data.lock().unwrap();
                log::debug!("Performing gpu mapped write: {0:?}", src_view.shape());
                let src_view = bytemuck::cast_slice(src_view.as_slice().unwrap());
                dst_view.copy_from_slice(src_view);
                drop(dst_view);
                self.queue.submit(None);
                self.grid_data_iter = curr_iter;
                self.total_grid_downloads += 1;
                true
            },
        }
    }
}
