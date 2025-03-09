use ndarray::{Array1, Array4};
use std::sync::{Arc, Mutex};
use super::app::UserEvent;
use log::{info};

pub struct AppGui {
    name: String,
    age: usize,
    curr_step: usize,
    total_steps: usize,
    total_grid_downloads: usize,
    grid_data: Array4<f32>,
    grid_data_iter: usize,
}

impl AppGui {
    pub fn new() -> Self {
        Self {
            name: "Arthur".to_owned(),
            age: 42,
            curr_step: 0,
            total_steps: 0,
            total_grid_downloads: 0,
            grid_data: Array4::<f32>::zeros((1,1,1,1)),
            grid_data_iter: 0,
        }
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
            ui.label(format!("Grid size: {0:?}", self.grid_data.shape()));
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
                self.grid_data_iter = curr_iter;
                let buffer = data.lock().unwrap();
                if self.grid_data.shape() != buffer.shape() {
                    info!("Resizing ui grid buffer to {0:?}", buffer.shape());
                    self.grid_data = buffer.clone();
                } else {
                    self.grid_data.assign(&buffer);
                }
                self.total_grid_downloads += 1;
                true
            },
        }
    }
}
