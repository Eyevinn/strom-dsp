//! Open/close state machine with hysteresis and hold, as used by a noise gate
//! (textbook gate description, Zölzer, *DAFX*, dynamics chapter).

/// Two-threshold (hysteresis) switch with a hold counter.
///
/// Levels are in whatever unit the caller uses consistently (the gate uses
/// linear peak amplitude). A closed machine opens when `level > open`. An open
/// machine re-arms its hold counter every frame where `level ≥ close`; while
/// below `close` the counter runs down one per frame, and the machine closes on
/// the first frame below `close` after the counter reached 0. With `hold = 0`
/// it closes on the first frame below `close`. `close > open` is treated as
/// `close = open`.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct GateStateMachine {
    open_level: f64,
    close_level: f64,
    hold_frames: u64,
    open: bool,
    hold_left: u64,
}

impl GateStateMachine {
    /// Set the thresholds. `close` is limited to `open`.
    pub fn set_thresholds(&mut self, open: f64, close: f64) {
        self.open_level = open;
        self.close_level = close.min(open);
    }

    /// Set the hold time in frames; a running hold is shortened if needed.
    pub fn set_hold_frames(&mut self, frames: u64) {
        self.hold_frames = frames;
        self.hold_left = self.hold_left.min(frames);
    }

    /// Whether the machine is open (including during hold).
    #[inline]
    pub fn is_open(&self) -> bool {
        self.open
    }

    /// Return to the closed state.
    pub fn reset(&mut self) {
        self.open = false;
        self.hold_left = 0;
    }

    /// Feed one frame's detector level and return whether the machine is open.
    #[inline]
    pub fn step(&mut self, level: f64) -> bool {
        if self.open {
            if level >= self.close_level {
                self.hold_left = self.hold_frames;
            } else if self.hold_left > 0 {
                self.hold_left -= 1;
            } else {
                self.open = false;
            }
        } else if level > self.open_level {
            self.open = true;
            self.hold_left = self.hold_frames;
        }
        self.open
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn machine(hold: u64) -> GateStateMachine {
        let mut m = GateStateMachine::default();
        m.set_thresholds(0.5, 0.2);
        m.set_hold_frames(hold);
        m
    }

    #[test]
    fn hysteresis_no_chatter_between_thresholds() {
        let mut m = machine(0);
        assert!(!m.step(0.4), "between thresholds while closed stays closed");
        assert!(m.step(0.6));
        for _ in 0..100 {
            assert!(m.step(0.3), "between thresholds while open stays open");
        }
        assert!(!m.step(0.1));
    }

    #[test]
    fn hold_counts_frames_below_close() {
        let mut m = machine(10);
        m.step(1.0);
        for i in 0..10 {
            assert!(m.step(0.0), "frame {i} still held");
        }
        assert!(!m.step(0.0));
        // Re-arm: a frame at/above close restarts the hold.
        m.step(1.0);
        for _ in 0..5 {
            m.step(0.0);
        }
        m.step(0.2);
        for _ in 0..10 {
            assert!(m.step(0.0));
        }
        assert!(!m.step(0.0));
    }

    #[test]
    fn close_above_open_acts_as_open() {
        let mut m = GateStateMachine::default();
        m.set_thresholds(0.3, 0.9);
        m.step(0.31);
        assert!(m.step(0.3), "level == close (== open) keeps it open");
        assert!(!m.step(0.29));
    }
}
