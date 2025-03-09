use log::{info, error, debug};
use std::sync::Arc;
use std::num::NonZeroU32;
use super::{
    gui::AppGui,
    app::UserEvent,
};

struct WinitWindow {
    window: winit::window::Window,
    gl: Arc<glow::Context>,
    gl_context: glutin::context::PossiblyCurrentContext,
    gl_display: glutin::display::Display,
    gl_surface: glutin::surface::Surface<glutin::surface::WindowSurface>,
    egui_glow: egui_glow::EguiGlow,
    is_redraw_requested: bool,
}

impl WinitWindow {
    async fn new(event_loop: &winit::event_loop::ActiveEventLoop) -> anyhow::Result<Self> {
        // Source: https://github.com/emilk/egui/blob/0.31.1/crates/egui_glow/examples/pure_glow.rs
        use glutin::context::NotCurrentGlContext;
        use glutin::display::GetGlDisplay;
        use glutin::display::GlDisplay;
        use glutin::prelude::GlSurface;
        use winit::raw_window_handle::HasWindowHandle;

        let winit_window_builder = winit::window::WindowAttributes::default()
            .with_resizable(true)
            .with_inner_size(winit::dpi::LogicalSize {
                width: 800.0,
                height: 600.0,
            })
            .with_title("egui_glow example") // Keep hidden until we've painted something. See https://github.com/emilk/egui/pull/2279
            .with_visible(false);

        let config_template_builder = glutin::config::ConfigTemplateBuilder::new()
            .prefer_hardware_accelerated(None)
            .with_depth_size(0)
            .with_stencil_size(0)
            .with_transparency(false);

        log::debug!("trying to get gl_config");
        let (mut window, gl_config) =
            glutin_winit::DisplayBuilder::new() // let glutin-winit helper crate handle the complex parts of opengl context creation
                .with_preference(glutin_winit::ApiPreference::FallbackEgl) // https://github.com/emilk/egui/issues/2520#issuecomment-1367841150
                .with_window_attributes(Some(winit_window_builder.clone()))
                .build(
                    event_loop,
                    config_template_builder,
                    |mut config_iterator| {
                        config_iterator.next().expect(
                            "failed to find a matching configuration for creating glutin config",
                        )
                    },
                )
                .expect("failed to create gl_config");
        let gl_display = gl_config.display();
        log::debug!("found gl_config: {:?}", &gl_config);

        let raw_window_handle = window.as_ref().map(|w| {
            w.window_handle()
                .expect("failed to get window handle")
                .as_raw()
        });
        log::debug!("raw window handle: {:?}", raw_window_handle);
        let context_attributes =
            glutin::context::ContextAttributesBuilder::new().build(raw_window_handle);
        // by default, glutin will try to create a core opengl context. but, if it is not available, try to create a gl-es context using this fallback attributes
        let fallback_context_attributes = glutin::context::ContextAttributesBuilder::new()
            .with_context_api(glutin::context::ContextApi::Gles(None))
            .build(raw_window_handle);
        let not_current_gl_context = unsafe {
            gl_display
                    .create_context(&gl_config, &context_attributes)
                    .unwrap_or_else(|_| {
                        log::debug!("failed to create gl_context with attributes: {:?}. retrying with fallback context attributes: {:?}",
                            &context_attributes,
                            &fallback_context_attributes);
                        gl_config
                            .display()
                            .create_context(&gl_config, &fallback_context_attributes)
                            .expect("failed to create context even with fallback attributes")
                    })
        };

        // this is where the window is created, if it has not been created while searching for suitable gl_config
        let window = window.take().unwrap_or_else(|| {
            log::debug!("window doesn't exist yet. creating one now with finalize_window");
            glutin_winit::finalize_window(event_loop, winit_window_builder.clone(), &gl_config)
                .expect("failed to finalize glutin window")
        });
        let (width, height): (u32, u32) = window.inner_size().into();
        let width = NonZeroU32::new(width).unwrap_or(NonZeroU32::MIN);
        let height = NonZeroU32::new(height).unwrap_or(NonZeroU32::MIN);

        let surface_attributes =
            glutin::surface::SurfaceAttributesBuilder::<glutin::surface::WindowSurface>::new()
                .build(
                    window
                        .window_handle()
                        .expect("failed to get window handle")
                        .as_raw(),
                    width,
                    height,
                );
        log::debug!(
            "creating surface with attributes: {:?}",
            &surface_attributes
        );
        let gl_surface = unsafe {
            gl_display
                .create_window_surface(&gl_config, &surface_attributes)
                .unwrap()
        };
        log::debug!("surface created successfully: {gl_surface:?}.making context current");
        let gl_context = not_current_gl_context.make_current(&gl_surface).unwrap();

        gl_surface
            .set_swap_interval(
                &gl_context,
                glutin::surface::SwapInterval::Wait(NonZeroU32::MIN),
            )
            .unwrap();

        let glow_context = unsafe {
            glow::Context::from_loader_function(|s| {
                let s = std::ffi::CString::new(s).expect("failed to construct C string from string for gl proc address");
                gl_display.get_proc_address(&s)
            })
        };
        let glow_context = Arc::new(glow_context);
        let egui_glow = egui_glow::EguiGlow::new(event_loop, glow_context.clone(), None, None, true);

        window.set_visible(true);


        Ok(Self {
            window,
            gl: glow_context,
            gl_context,
            gl_display,
            gl_surface,
            egui_glow,
            is_redraw_requested: false,
        })
    }

    fn on_resize(&mut self, width: u32, height: u32) {
        let width = NonZeroU32::new(width).unwrap_or(NonZeroU32::MIN);
        let height = NonZeroU32::new(height).unwrap_or(NonZeroU32::MIN);
        use glutin::surface::GlSurface;
        self.gl_surface.resize(
            &self.gl_context,
            width,
            height,
        );
        self.trigger_redraw();
    }

    fn trigger_redraw(&mut self) {
        if !self.is_redraw_requested {
            self.is_redraw_requested = true;
            self.window.request_redraw();
        }
    }

    fn on_redraw_requested(&mut self, mut render_call: impl FnMut(&egui::Context)) {
        self.is_redraw_requested = false;
        self.egui_glow.run(&self.window, |context| render_call(context));

        use glow::HasContext as _;
        use glutin::surface::GlSurface;
        unsafe { self.gl.clear_color(1.0, 1.0, 1.0, 1.0); }
        self.egui_glow.paint(&mut self.window);
        self.gl_surface.swap_buffers(&self.gl_context).unwrap();
        self.window.set_visible(true);
    }
}

impl Drop for WinitWindow {
    fn drop(&mut self) {
        self.egui_glow.destroy();
    }
}

pub struct WinitApplication {
    gui: AppGui,
    window: Option<WinitWindow>,
}

impl WinitApplication {
    pub fn new() -> Self {
        Self {
            gui: AppGui::new(),
            window: None,
        }
    }
}

impl winit::application::ApplicationHandler<UserEvent> for WinitApplication {
    // Apparently windows can get destroyed/created on a whim for mobile platforms which necessitates this
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let wgpu_window = pollster::block_on(WinitWindow::new(event_loop)).unwrap();
        debug!("Created winit window");
        self.window = Some(wgpu_window);
    }

    fn window_event(&mut self, event_loop: &winit::event_loop::ActiveEventLoop, _id: winit::window::WindowId, event: winit::event::WindowEvent) {
        let window = self.window.as_mut().expect("Event loop shouldn't be able to exist without window");
        // let egui consume events first
        let response = window.egui_glow.on_window_event(&window.window, &event);
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
            WindowEvent::RedrawRequested => window.on_redraw_requested(|ctx| self.gui.render(ctx)),
            WindowEvent::Resized(size) => window.on_resize(size.width, size.height),
            event => debug!("Unhandled window event: {0:?}", event),
        }
    }

    fn user_event(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop, event: UserEvent) {
        let is_repaint = self.gui.handle_user_event(event);
        if is_repaint {
            let window = self.window.as_mut().expect("Event loop shouldn't be able to exist without window");
            window.trigger_redraw();
        }
    }

    fn exiting(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        self.window = None;
        debug!("Destroyed winit window");
    }
}
