use std::sync::{Condvar, Mutex};

pub enum PoolChannel<T> {
    Pending(T),
    Closed,
    Empty,
}

pub struct PoolEntry<T, U> {
    pub value: Mutex<T>,
    pub state: Mutex<PoolChannel<U>>,
    signal_state: Condvar,
}

impl<T,U> PoolEntry<T,U> {
    pub fn new(value: T) -> Self {
        Self {
            value: Mutex::new(value),
            state: Mutex::new(PoolChannel::Empty),
            signal_state: Condvar::new(),
        }
    }

    pub fn wait_pending_or_close(&self) {
        let mut state = self.state.lock().unwrap();
        while matches!(*state, PoolChannel::Empty) {
            state = self.signal_state.wait(state).unwrap();
        }
    }

    pub fn wait_empty(&self) {
        let mut state = self.state.lock().unwrap();
        while matches!(*state, PoolChannel::Pending(..)) {
            state = self.signal_state.wait(state).unwrap();
        }
    }

    pub fn signal_free(&self) {
        let mut state = self.state.lock().unwrap();
        *state = PoolChannel::Empty;
        self.signal_state.notify_one();
    }
 
    pub fn signal_pending(&self, data: U) {
        let mut state = self.state.lock().unwrap();
        *state = PoolChannel::Pending(data);
        self.signal_state.notify_one();
    }

    pub fn signal_close(&self) {
        let mut state = self.state.lock().unwrap();
        *state = PoolChannel::Closed;
        self.signal_state.notify_one();
    }
}
