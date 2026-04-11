use std::sync::{Arc, Mutex};
use std::time::Duration;
use that_bass::v2::{
    instrumentation::{Event, ScheduleBuild, Sink},
    Configuration, Store,
};

pub fn run() {
    let recording_sink = Arc::new(RecordingSink::default());
    let store = Store::with_instrumentation(Configuration::default(), recording_sink.clone());

    store
        .instrumentation_sink()
        .record(Event::ScheduleBuildCompleted(ScheduleBuild {
            scheduled_function_count: 3,
            happens_before_edge_count: 2,
            elapsed: Duration::from_micros(42),
        }));

    let recorded_events = recording_sink.recorded_events();

    println!("Instrumentation");
    println!(
        "  store target chunk bytes: {}",
        store.configuration().target_chunk_byte_count().get()
    );
    println!("  recorded events: {}", recorded_events.len());
    println!("  first event: {:?}", recorded_events.first());
}

#[derive(Default)]
struct RecordingSink {
    recorded_events: Mutex<Vec<Event>>,
}

impl RecordingSink {
    fn recorded_events(&self) -> Vec<Event> {
        self.recorded_events
            .lock()
            .expect("recording sink mutex should not be poisoned")
            .clone()
    }
}

impl Sink for RecordingSink {
    fn record(&self, event: Event) {
        self.recorded_events
            .lock()
            .expect("recording sink mutex should not be poisoned")
            .push(event);
    }
}
