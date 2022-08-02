use lazy_static::lazy_static;
use log::{info, log, Level};
use std::fmt::Display;
use std::sync::RwLock;
use std::time::{Duration, Instant};

/// periodic logging
///
/// useful for when the process will go into a loop, and wants to inform the user
/// that something is still happening, and maybe even show it's progress.
///
/// fn factorial(n: u128) -> u128 {
///     let mut result = 1;
///     let mut periodic = PeriodicLogger::new("calculating factorial", Level::Info);
///     for i in 2..=n {
///         result *= i;
///         periodic.log(format!("{} / {}", i, n));
///     }
///     result
/// }
pub struct PeriodicLogger {
    last_logged: Instant,
    interval: Duration,
    level: Level,
}

impl PeriodicLogger {
    /// creates a new PeriodicLogger, with the default interval which is one second by default
    ///
    /// in order to set the default interval, use [`set_default_interval`]
    ///
    /// panics if the underlying lock holding the default interval is poisoned
    pub fn new(start_message: &str, level: Level) -> Self {
        Self::new_with_interval(start_message, *DEFAULT_INTERVAL.read().unwrap(), level)
    }

    pub fn new_with_interval(start_message: &str, interval: Duration, level: Level) -> Self {
        info!("{}", start_message);
        Self {
            last_logged: Instant::now(),
            interval,
            level,
        }
    }

    /// logs the message at the self.level if self.interval has passed, and updates
    /// self.last_logged so that the next log will not occur until self.interval passes again
    pub fn log<D: Display>(&mut self, message: D) {
        let now = Instant::now();
        if now - self.last_logged >= self.interval {
            self.last_logged = now;
            log!(self.level, "\t{}", message);
        }
    }
}

lazy_static! {
    static ref DEFAULT_INTERVAL: RwLock<Duration> = RwLock::new(Duration::from_secs(1));
}

/// sets the default interval
/// panics if the underlying lock holding the default interval is poisoned
#[allow(unused)]
pub fn set_default_interval(interval: Duration) {
    *DEFAULT_INTERVAL.write().unwrap() = interval;
}
