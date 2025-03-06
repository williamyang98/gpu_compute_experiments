use serde::Serialize;

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

