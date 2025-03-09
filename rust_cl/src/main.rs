use log::{info, error};
use std::sync::{Arc, Mutex};
use opencl3::{
    device::Device, 
    error_codes::ClError, 
    platform::{get_platforms, Platform},
};
use ytdlp_server::{
    chrome_trace::Trace,
    app::App,
    gui::UserEvent,
    window::MainWindow,
};
use std::io::{BufWriter, Write};
use std::fs::File;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Platform id
    #[arg(short, long, default_value_t = 0)]
    platform_index: usize,
    /// Device id
    #[arg(short, long, default_value_t = 0)]
    device_index: usize,
    /// List platforms
    #[arg(long)]
    list_platforms: bool,
    /// List devices
    #[arg(long)]
    list_devices: bool,
    /// Total simulation steps
    #[arg(long, default_value_t = 2048)]
    total_steps: usize,
    /// Record stride
    #[arg(long)]
    record_stride: Option<usize>,
}

fn list_platforms(platforms: &[Platform]) {
    use prettytable::{Table, Row, Cell, row};
    let mut table = Table::new();
    table.set_titles(Row::new(vec![Cell::new("Platforms").style_spec("H3c")]));
    table.add_row(row!["id", "vendor", "version"]);
    for (platform_index, platform) in platforms.iter().enumerate() {
        table.add_row(row![
            platform_index,
            platform.vendor().unwrap_or("?".to_owned()),
            platform.version().unwrap_or("?".to_owned()),
        ]);
    }
    table.printstd();
}

fn list_devices(devices: &[Device]) {
    use prettytable::{Table, Row, Cell, row};
    let mut table = Table::new();
    table.set_titles(Row::new(vec![Cell::new("Devices").style_spec("H4c")]));
    table.add_row(row!["id", "name", "vendor", "version"]);
    for (device_index, device) in devices.iter().enumerate() {
        table.add_row(row![
            device_index,
            device.name().unwrap_or("?".to_owned()),
            device.vendor().unwrap_or("?".to_owned()),
            device.version().unwrap_or("?".to_owned()),
        ]);
    }
    table.printstd();
}

fn handle_args(args: &Args) -> Result<(Platform, Device), ClError> {
    let platforms = get_platforms()?;
    if args.list_platforms {
        list_platforms(platforms.as_slice());
        std::process::exit(0);
    }

    let Some(platform) = platforms.get(args.platform_index) else {
        error!("Platform index {0} is outside the range of {1} platforms", args.platform_index, platforms.len());
        list_platforms(platforms.as_slice());
        std::process::exit(1);
    };

    info!("Selected platform id={0}, name={1}, vendor={2}, version={3}",
        args.platform_index,
        platform.name().unwrap_or("?".to_owned()),
        platform.version().unwrap_or("?".to_owned()),
        platform.vendor().unwrap_or("?".to_owned()),
    );

    let devices = platform.get_devices(opencl3::device::CL_DEVICE_TYPE_ALL)?;
    let devices: Vec<Device> = devices.iter().map(|id| Device::new(*id)).collect();

    if args.list_devices {
        list_devices(devices.as_slice());
        std::process::exit(0);
    }

    let Some(device) = devices.get(args.device_index) else {
        error!("Device index {0} is outside the range of {1} devices", args.device_index, devices.len());
        list_devices(devices.as_slice());
        std::process::exit(1);
    };

    info!("Selected device id={0}, name={1}, vendor={2}, version={3}",
        args.device_index,
        device.name().unwrap_or("?".to_owned()),
        device.version().unwrap_or("?".to_owned()),
        device.vendor().unwrap_or("?".to_owned()),
    );

    Ok((*platform, *device))
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let args = Args::parse();
    let (_platform, device) = handle_args(&args)?;

    let winit_event_loop = winit::event_loop::EventLoop::<UserEvent>::with_user_event().build()?;
    let (user_events_tx, user_events_rx) = crossbeam_channel::bounded::<UserEvent>(1024);

    let device = Arc::new(device);
    let mut app = App::new(device, user_events_tx)?;

    let compute_thread = std::thread::spawn({
        move || -> anyhow::Result<()> {
            info!("Running app");
            app.init_simulation_data();
            app.upload_simulation_data()?;
            app.run(args.total_steps, args.record_stride)?;

            let app_events = app.gpu_trace.get_chrome_events();
            let chrome_trace = Trace { events: app_events };

            info!("Writing chome trace with {0} entries", chrome_trace.events.len());
            let json_string = serde_json::to_string_pretty(&chrome_trace)?;
            let file = File::create("./trace.json")?;
            let mut writer = BufWriter::new(file);
            writer.write_all(json_string.as_bytes())?;
            writer.flush()?;
            Ok(())
        }
    });

    let user_events_thread = std::thread::spawn({
        let event_loop = winit_event_loop.create_proxy();
        move || -> anyhow::Result<()> {
            for event in user_events_rx {
                if let Err(_err) = event_loop.send_event(event) {
                    break;
                }
            }
            Ok(())
        }
    });

    let mut main_window = MainWindow::new();
    winit_event_loop.set_control_flow(winit::event_loop::ControlFlow::Wait);
    winit_event_loop.run_app(&mut main_window)?;

    compute_thread.join().unwrap()?;
    user_events_thread.join().unwrap()?;

    Ok(())
}
