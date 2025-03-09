pub enum UserEvent {

}

pub struct AppGui {
    name: String,
    age: usize,
}

impl AppGui {
    pub fn new() -> Self {
        Self {
            name: "Arthur".to_owned(),
            age: 42,
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
    }

    pub fn handle_user_event(&mut self, event: UserEvent) {

    }
}
