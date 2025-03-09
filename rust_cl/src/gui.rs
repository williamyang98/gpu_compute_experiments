pub enum UserEvent {
    SetProgress { curr_step: usize, total_steps: usize },
}

pub struct AppGui {
    name: String,
    age: usize,
    curr_step: usize,
    total_steps: usize,
}

impl AppGui {
    pub fn new() -> Self {
        Self {
            name: "Arthur".to_owned(),
            age: 42,
            curr_step: 0,
            total_steps: 0,
        }
    }

    pub fn render(&mut self, ui: &mut egui::Ui) {
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
    }

    pub fn handle_user_event(&mut self, event: UserEvent) -> bool {
        match event {
            UserEvent::SetProgress { curr_step, total_steps } => {
                self.curr_step = curr_step;
                self.total_steps = total_steps;
                true
            },
        }
    }
}
