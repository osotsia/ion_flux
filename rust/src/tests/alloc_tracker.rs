use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

thread_local! {
    static IS_TRACKING: Cell<bool> = Cell::new(false);
    static THREAD_ALLOCS: Cell<usize> = Cell::new(0);
    static THREAD_DEALLOCS: Cell<usize> = Cell::new(0);
    static THREAD_BYTES_ALLOCATED: Cell<usize> = Cell::new(0);
}

pub struct TrackingAllocator;

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        IS_TRACKING.with(|tracking| {
            if tracking.get() {
                tracking.set(false);
                THREAD_ALLOCS.with(|c| c.set(c.get() + 1));
                THREAD_BYTES_ALLOCATED.with(|b| b.set(b.get() + layout.size()));
                tracking.set(true);
            }
        });
        System.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        IS_TRACKING.with(|tracking| {
            if tracking.get() {
                tracking.set(false);
                THREAD_DEALLOCS.with(|c| c.set(c.get() + 1));
                tracking.set(true);
            }
        });
        System.dealloc(ptr, layout)
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        IS_TRACKING.with(|tracking| {
            if tracking.get() {
                tracking.set(false);
                THREAD_ALLOCS.with(|c| c.set(c.get() + 1));
                if new_size > layout.size() {
                    THREAD_BYTES_ALLOCATED.with(|b| b.set(b.get() + (new_size - layout.size())));
                }
                tracking.set(true);
            }
        });
        System.realloc(ptr, layout, new_size)
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        IS_TRACKING.with(|tracking| {
            if tracking.get() {
                tracking.set(false);
                THREAD_ALLOCS.with(|c| c.set(c.get() + 1));
                THREAD_BYTES_ALLOCATED.with(|b| b.set(b.get() + layout.size()));
                tracking.set(true);
            }
        });
        System.alloc_zeroed(layout)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AllocStats {
    pub alloc_count: usize,
    pub dealloc_count: usize,
    pub bytes_allocated: usize,
}

pub fn track_allocations<F, R>(f: F) -> (R, AllocStats)
where
    F: FnOnce() -> R,
{
    IS_TRACKING.with(|t| t.set(true));
    THREAD_ALLOCS.with(|c| c.set(0));
    THREAD_DEALLOCS.with(|c| c.set(0));
    THREAD_BYTES_ALLOCATED.with(|b| b.set(0));

    struct Guard;
    impl Drop for Guard {
        fn drop(&mut self) {
            IS_TRACKING.with(|t| t.set(false));
        }
    }
    let _guard = Guard;

    let result = f();

    let stats = AllocStats {
        alloc_count: THREAD_ALLOCS.with(|c| c.get()),
        dealloc_count: THREAD_DEALLOCS.with(|c| c.get()),
        bytes_allocated: THREAD_BYTES_ALLOCATED.with(|b| b.get()),
    };

    (result, stats)
}
