use serde::Serialize;
use std::time::Duration;
use opencl3::{
    event::Event,
    error_codes::ClError,
};

#[derive(Default, Clone, Copy)]
pub struct TraceSpan {
    pub start: Duration,
    pub end: Duration,
}

impl TraceSpan {
    pub fn elapsed(&self) -> Duration {
        self.end - self.start
    }

    pub fn from_event(ev: &Event, ns_per_tick: u64) -> Result<Self, ClError> {
        let ticks_start: u64 = ev.profiling_command_start()?;
        let ticks_end: u64 = ev.profiling_command_end()?;
        let ns_start = ticks_start*ns_per_tick;
        let ns_end = ticks_end*ns_per_tick;
        let start = Duration::from_nanos(ns_start);
        let end = Duration::from_nanos(ns_end);
        Ok(Self{ start, end })
    }
}

#[derive(Default, Serialize)]
pub struct Trace {
    #[serde(rename="traceEvents")]
    pub events: Vec<TraceEvent>,
}

#[derive(Clone, Serialize)]
pub struct TraceEvent {
    #[serde(rename="pid")]
    pub process_id: u64,
    #[serde(rename="tid")]
    pub thread_id: u64,
    #[serde(rename="ts")]
    pub us_start: u128,
    #[serde(rename="dur")]
    pub us_duration: u128,
    #[serde(rename="name")]
    pub name: String,
    #[serde(rename="ph")]
    pub category: String,
}

