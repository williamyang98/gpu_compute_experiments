use log::{info, error, debug};
use std::sync::Arc;
use super::gui::{UserEvent, AppGui};

struct WgpuWindow {
    window: Arc<winit::window::Window>,
    surface_config: wgpu::SurfaceConfiguration,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    egui_renderer: egui_wgpu::Renderer,
    egui_state: egui_winit::State,
    is_redraw_requested: bool,
}

impl WgpuWindow {
    async fn new(winit_window: winit::window::Window, context: egui::Context) -> anyhow::Result<Self> {
        let window = Arc::new(winit_window);
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::from_env().unwrap_or(wgpu::Backends::PRIMARY),
            backend_options: wgpu::BackendOptions::from_env_or_default(),
            flags: wgpu::InstanceFlags::from_build_config().with_env(),
        });
        let surface = instance.create_surface(window.clone())?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: Default::default(),
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
            })
            .await
            .ok_or(anyhow::Error::msg("Failed to find valid wgpu adapter"))?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default(), None)
            .await?;

        let initial_window_size = window.inner_size();

        let mut surface_config = surface
            .get_default_config(&adapter, initial_window_size.width, initial_window_size.height)
            .ok_or(anyhow::Error::msg("Failed to get default config for wgpu surface"))?;
        surface_config.format = wgpu::TextureFormat::Bgra8Unorm;
        surface_config.present_mode = wgpu::PresentMode::AutoVsync;
        surface.configure(&device, &surface_config);

        let egui_renderer = egui_wgpu::Renderer::new(
            &device,
            surface_config.format, None,
            1,
            false,
        );

        let viewport_id = context.viewport_id();
        let native_pixels_per_point = context.native_pixels_per_point();

        let egui_state = egui_winit::State::new(
            context,
            viewport_id,
            &window,
            native_pixels_per_point,
            window.theme(),
            None,
        );

        Ok(Self {
            window,
            surface_config,
            surface,
            device,
            queue,
            egui_renderer,
            egui_state,
            is_redraw_requested: false,
        })
    }

    fn on_resize(&mut self, width: u32, height: u32) {
        let width = width.max(100);
        let height = height.max(100);
        self.surface_config.width = width;
        self.surface_config.height = height;
        self.surface.configure(&self.device, &self.surface_config);
        self.trigger_redraw();
    }

    fn trigger_redraw(&mut self) {
        if !self.is_redraw_requested {
            self.is_redraw_requested = true;
            self.window.request_redraw();
        }
    }

    fn on_redraw_requested(&mut self, render_call: impl FnOnce(&mut egui::Ui)) {
        self.is_redraw_requested = false;

        // egui tessellation
        let raw_input = self.egui_state.take_egui_input(&self.window);
        let context = self.egui_state.egui_ctx();
        context.begin_pass(raw_input);
        egui::CentralPanel::default().show(context, render_call);
        let full_output = context.end_pass();
        let paint_jobs = context.tessellate(full_output.shapes, full_output.pixels_per_point);

        // render pass
        let frame = self.surface.get_current_texture().expect("Failed to acquire next swap chain texture");
        let texture_view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let renderer = &mut self.egui_renderer;
        let window_size = self.window.inner_size();
        let screen_descriptor = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [window_size.width, window_size.height],
            pixels_per_point: full_output.pixels_per_point,
        };

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("main_window_command_encoder"),
        });

        // egui render pass
        for (texture_id, image_delta) in full_output.textures_delta.set.iter() {
            renderer.update_texture(&self.device, &self.queue, *texture_id, image_delta);
        }
        let _command_buffers = renderer.update_buffers(&self.device, &self.queue, &mut encoder, paint_jobs.as_slice(), &screen_descriptor);
        let rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("egui_render_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &texture_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        let mut rpass = rpass.forget_lifetime();
        renderer.render(&mut rpass, paint_jobs.as_slice(), &screen_descriptor);
        // @NOTE: because of the stupid .forget_lifetime() call we need to manually drop this
        //        otherwise the renderpass will persist after its supposed to be moved into the encoder
        //        and cause a very confusing error about the renderpass not being finished
        drop(rpass);
        self.queue.submit(Some(encoder.finish()));
        frame.present();
    }
}


pub struct MainWindow {
    gui: AppGui,
    window: Option<WgpuWindow>,
}

impl MainWindow {
    pub fn new() -> Self {
        Self {
            gui: AppGui::new(),
            window: None,
        }
    }

}

impl winit::application::ApplicationHandler<UserEvent> for MainWindow {
    // Apparently windows can get destroyed/created on a whim for mobile platforms which necessitates this
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let context = egui::Context::default();
        let viewport_builder = egui::ViewportBuilder::default();
        context.set_fonts(egui::FontDefinitions::default());
        context.set_style(egui::Style::default());
        let winit_window = egui_winit::create_window(&context, event_loop, &viewport_builder).unwrap();
        let wgpu_window = pollster::block_on(WgpuWindow::new(winit_window, context)).unwrap();
        debug!("Created wgpu window");

        self.window = Some(wgpu_window);
    }

    fn window_event(&mut self, event_loop: &winit::event_loop::ActiveEventLoop, _id: winit::window::WindowId, event: winit::event::WindowEvent) {
        let window = self.window.as_mut().expect("Event loop shouldn't be able to exist without window");
        // let egui consume events first
        let response = window.egui_state.on_window_event(&window.window, &event);
        if response.repaint {
            window.trigger_redraw();
        }
        if response.consumed {
            return;
        }

        use winit::event::WindowEvent;
        match event {
            WindowEvent::CloseRequested => {
                info!("Closing winit window");
                event_loop.exit();
            },
            WindowEvent::RedrawRequested => window.on_redraw_requested(|ui| self.gui.render(ui)),
            WindowEvent::Resized(size) => window.on_resize(size.width, size.height),
            event => debug!("Unhandled window event: {0:?}", event),
        }
    }

    fn user_event(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop, event: UserEvent) {
        self.gui.handle_user_event(event);
    }
}
