use glow::HasContext;
use ndarray::{Array1, Array4};
use std::sync::{Arc, Mutex};
use super::app::UserEvent;

pub struct AppGui {
    name: String,
    age: usize,
    curr_step: usize,
    total_steps: usize,
    total_grid_downloads: usize,
    grid_data_iter: usize,
    grid_shape: Array1<usize>,
    grid_gpu_buffer: Option<glow::Buffer>,
}

impl AppGui {
    pub fn new() -> Self {
        Self {
            name: "Arthur".to_owned(),
            age: 42,
            curr_step: 0,
            total_steps: 0,
            total_grid_downloads: 0,
            grid_shape: Array1::from(vec![16,256,512,3]),
            grid_data_iter: 0,
            grid_gpu_buffer: None,
        }
    }

    pub fn on_gl_context(&mut self, gl: &glow::Context) {
        let total_cells: usize = self.grid_shape.iter().product();
        let buffer_size = total_cells * std::mem::size_of::<f32>();
        unsafe {
            log::info!("Creating opengl ssbo with size={0} bytes", buffer_size);
            let buffer = gl.create_buffer().unwrap();
            egui_glow::check_for_gl_error!(gl, "SSBO create");
            gl.bind_buffer(glow::SHADER_STORAGE_BUFFER, Some(buffer));
            egui_glow::check_for_gl_error!(gl, "SSBO bind");
            gl.buffer_data_size(glow::SHADER_STORAGE_BUFFER, buffer_size as i32, glow::DYNAMIC_DRAW);
            egui_glow::check_for_gl_error!(gl, "SSBO buffer size");
            gl.bind_buffer(glow::SHADER_STORAGE_BUFFER, None);
            egui_glow::check_for_gl_error!(gl, "SSBO unbind");
            self.grid_gpu_buffer = Some(buffer);
        }
    }

    pub fn render(&mut self, context: &egui::Context, gl: &glow::Context) {
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
        });
    }

    pub fn handle_user_event(&mut self, event: UserEvent, gl: &glow::Context) -> bool {
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
                // TODO: this is done inside a shader to layout(binding=index)
                // gl.bind_buffer_base(glow::SHADER_STORAGE_BUFFER, index, Some(buffer));
                // gl.bind_buffer(glow::SHADER_STORAGE_BUFFER, None);

                let data = data.lock().unwrap();
                let buffer = *self.grid_gpu_buffer.as_ref().expect("opengl ssbo should have been created already");
                assert!(self.grid_shape.as_slice().unwrap() == data.shape());
                let cpu_data = data.as_slice().unwrap();
                let cpu_data: &[u8] = bytemuck::cast_slice(cpu_data);
                unsafe {
                    log::info!("Uploading data to ssbo shape={0:?}", data.shape());
                    gl.bind_buffer(glow::SHADER_STORAGE_BUFFER, Some(buffer));
                    egui_glow::check_for_gl_error!(gl, "SSBO bind");
                    gl.buffer_sub_data_u8_slice(glow::SHADER_STORAGE_BUFFER, 0, cpu_data);
                    egui_glow::check_for_gl_error!(gl, "SSBO sub upload");
                    gl.bind_buffer(glow::SHADER_STORAGE_BUFFER, None);
                    egui_glow::check_for_gl_error!(gl, "SSBO unbind");
                }
                self.grid_data_iter = curr_iter;
                self.total_grid_downloads += 1;
                true
            },
        }
    }
}
