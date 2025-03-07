use log::info;
use std::sync::Arc;
use opencl3::{
    device::Device, 
    platform::get_platforms, 
};
use ytdlp_server::{
    chrome_trace::Trace,
    app::App,
};
use std::io::{BufWriter, Write};
use std::fs::File;

fn main() -> Result<(), String> {
    env_logger::init();

    let mut selected_platform = None;
    let mut selected_device = None;

    let platforms = get_platforms()?;
    for (platform_index, platform) in platforms.iter().enumerate() {
        selected_platform = Some(platform);
        info!("Platform {0}: {1}", platform_index, platform.name().unwrap_or("Unknown".to_owned()));
        let device_ids = platform.get_devices(opencl3::device::CL_DEVICE_TYPE_ALL)?;
        let devices: Vec<Device> = device_ids.iter().map(|id| Device::new(*id)).collect();
        for (device_index, device) in devices.iter().enumerate() {
            info!("  Device {0}: {1}", device_index, device.name().unwrap_or("Unknown".to_owned()));
            if platform_index == 0 && device_index == 0 {
                selected_device = Some(*device);
            }
        }
    }

    let _platform = selected_platform.expect("No available opencl platform");
    let device = selected_device.expect("No available opencl device");
    let device = Arc::new(device);

    info!("Running app");
    let mut app = App::new(device).map_err(|err| err.to_string())?;
    app.init_simulation_data();
    app.upload_simulation_data().map_err(|err| err.to_string())?;
    app.run()?;

    let app_events = app.gpu_trace.get_chrome_events();
    let chrome_trace = Trace { events: app_events };

    info!("Writing chome trace with {0} entries", chrome_trace.events.len());
    let json_string = serde_json::to_string_pretty(&chrome_trace).unwrap();
    let file = File::create("./trace.json").unwrap();
    let mut writer = BufWriter::new(file);
    writer.write_all(json_string.as_bytes()).unwrap();
    writer.flush().unwrap();
    Ok(())
}

