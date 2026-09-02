//! Cooperative cancellation between the async caller and the blocking run.
//!
//! Inference happens on a blocking thread, which the runtime cannot interrupt:
//! dropping the future that is waiting on it leaves the thread transcribing
//! audio nobody will read, holding the engine slot against requests that still
//! have a client. A [`CancelFlag`] is the way to tell it to stop.
//!
//! Two halves, because engines abort differently. Every engine can poll
//! [`CancelFlag::is_cancelled`] between units of work it controls; an engine
//! whose inference call blocks for seconds at a time instead installs an abort
//! action for the duration of that call ([`CancelFlag::on_cancel`]), which
//! [`CancelFlag::cancel`] invokes from whichever thread noticed the client is
//! gone.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

type AbortAction = Arc<dyn Fn() + Send + Sync>;

#[derive(Default)]
struct Inner {
    cancelled: AtomicBool,
    /// Installed for the duration of one engine call, so a cancel arriving
    /// mid-inference reaches the library that is running it.
    abort: Mutex<Option<AbortAction>>,
}

/// A cancellation handle shared by the async caller and the engine call it
/// spawned. Clones share one flag.
#[derive(Clone, Default)]
pub struct CancelFlag(Arc<Inner>);

impl CancelFlag {
    pub fn new() -> Self {
        CancelFlag::default()
    }

    /// Ask the in-flight run to stop. Idempotent, callable from any thread,
    /// and safe to call when nothing is running.
    pub fn cancel(&self) {
        self.0.cancelled.store(true, Ordering::SeqCst);
        // Cloned out of the lock: the action reaches into the engine, and
        // holding our mutex across that invites a lock order we do not
        // control.
        let action = self.action();
        if let Some(abort) = action {
            abort();
        }
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.cancelled.load(Ordering::SeqCst)
    }

    /// Install `abort` for as long as the returned guard lives.
    ///
    /// A cancel that lands between the caller spawning the run and this
    /// installing the action reaches the flag but not the action, so it would
    /// be lost if the engine went straight into its inference. Callers must
    /// therefore check [`CancelFlag::is_cancelled`] after installing and
    /// before starting work.
    #[must_use]
    pub fn on_cancel(&self, abort: AbortAction) -> AbortGuard<'_> {
        *self.lock() = Some(abort);
        AbortGuard { flag: self }
    }

    fn action(&self) -> Option<AbortAction> {
        self.lock().clone()
    }

    /// A poisoned lock means a previous holder panicked while swapping the
    /// action; the value itself is a plain Option and stays usable.
    fn lock(&self) -> std::sync::MutexGuard<'_, Option<AbortAction>> {
        self.0.abort.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl std::fmt::Debug for CancelFlag {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CancelFlag")
            .field("cancelled", &self.is_cancelled())
            .finish_non_exhaustive()
    }
}

/// Removes the abort action when the engine call it belongs to returns, so a
/// later cancel cannot reach into a run that has already finished.
pub struct AbortGuard<'a> {
    flag: &'a CancelFlag,
}

impl Drop for AbortGuard<'_> {
    fn drop(&mut self) {
        *self.flag.lock() = None;
    }
}

/// Cancels the flag unless disarmed: the async side holds one while it waits
/// for a run, so a dropped future (the client hung up) stops the run instead
/// of leaving it to finish into nothing.
pub struct CancelOnDrop {
    flag: CancelFlag,
    armed: bool,
}

impl CancelOnDrop {
    pub fn new(flag: CancelFlag) -> Self {
        CancelOnDrop { flag, armed: true }
    }

    /// The run finished on its own; leave the flag alone.
    pub fn disarm(mut self) {
        self.armed = false;
    }
}

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        if self.armed {
            self.flag.cancel();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_fresh_flag_is_not_cancelled() {
        let flag = CancelFlag::new();
        assert!(!flag.is_cancelled());
    }

    #[test]
    fn cancel_runs_the_installed_action_once_per_call() {
        let flag = CancelFlag::new();
        let calls = Arc::new(AtomicBool::new(false));
        let seen = Arc::clone(&calls);
        let _guard = flag.on_cancel(Arc::new(move || seen.store(true, Ordering::SeqCst)));
        flag.cancel();
        assert!(flag.is_cancelled());
        assert!(calls.load(Ordering::SeqCst), "abort action must run");
    }

    #[test]
    fn clones_share_one_flag() {
        let flag = CancelFlag::new();
        let clone = flag.clone();
        clone.cancel();
        assert!(flag.is_cancelled());
    }

    #[test]
    fn cancelling_without_an_action_is_fine() {
        let flag = CancelFlag::new();
        flag.cancel();
        assert!(flag.is_cancelled());
    }

    /// The guard is what keeps a cancel arriving after the run from reaching
    /// into the library that already returned.
    #[test]
    fn the_action_is_gone_once_the_guard_drops() {
        let flag = CancelFlag::new();
        let fired = Arc::new(AtomicBool::new(false));
        let seen = Arc::clone(&fired);
        {
            let _guard = flag.on_cancel(Arc::new(move || seen.store(true, Ordering::SeqCst)));
        }
        flag.cancel();
        assert!(flag.is_cancelled());
        assert!(!fired.load(Ordering::SeqCst), "stale action must not run");
    }

    /// A cancel that lands before the engine installs its action must still be
    /// visible, or the run starts with nobody waiting for it.
    #[test]
    fn a_cancel_before_install_is_still_observed() {
        let flag = CancelFlag::new();
        flag.cancel();
        let _guard = flag.on_cancel(Arc::new(|| {}));
        assert!(flag.is_cancelled());
    }

    #[test]
    fn cancel_on_drop_cancels_unless_disarmed() {
        let flag = CancelFlag::new();
        drop(CancelOnDrop::new(flag.clone()));
        assert!(flag.is_cancelled());

        let flag = CancelFlag::new();
        CancelOnDrop::new(flag.clone()).disarm();
        assert!(!flag.is_cancelled());
    }
}
