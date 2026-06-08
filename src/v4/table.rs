use crate::v4::{
    depend::Access,
    error::Error,
    meta::Meta,
    slice::Slice,
    utility::{self, IteratorExtension, allocate, deallocate, defer, is_unique},
};
use arc_swap::{ArcSwapAny, AsRaw};
use core::{
    alloc::Layout,
    any::{Any, TypeId},
    cell::RefCell,
    cmp::{self, Reverse},
    fmt::{self, Debug, Formatter},
    iter::{FusedIterator, empty},
    ops::Range,
    ptr::NonNull,
    slice::{from_raw_parts, from_raw_parts_mut},
    sync::atomic::{AtomicU32, Ordering},
};
use parking_lot::{Condvar, Mutex, RwLock, lock_api::RawRwLock};
use triomphe::{Arc, ThinArc};

#[derive(Debug, Clone)]
pub struct Table(ThinArc<Header, Column>);

#[derive(Debug, Clone)]
pub struct Tables(Arc<ArcSwapAny<ThinArc<(), Table>>>);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Lock {
    Rows,
    Column(u32, Access),
}

pub struct Row<'a> {
    row: u32,
    table: &'a Table,
    remove: &'a RefCell<Vec<u32>>,
}

#[derive(Debug)]
pub(crate) struct State {
    lock: u32,
    count: u32,
    capacity: u32,
    remove: Vec<u32>,
}

#[derive(Debug)]
struct Header {
    index: u32,
    count: AtomicU32,
    unlocked: Condvar,
    resolved: Condvar,
    state: Mutex<State>,
}

#[test]
fn boba() {
    dbg!(size_of::<Table>());
    dbg!(size_of::<Header>());
    dbg!(size_of::<State>());
}

impl Debug for Row<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("Row")
            .field("row", &self.row)
            .field("table", &self.table.index())
            .finish_non_exhaustive()
    }
}

#[derive(Debug)]
pub struct Rows<'a> {
    rows: Range<u32>,
    table: &'a Table,
    remove: &'a RefCell<Vec<u32>>,
}

#[derive(Debug)]
pub struct Column {
    meta: Meta,
    data: RwLock<NonNull<u8>>,
}

unsafe impl Send for Column {}
unsafe impl Sync for Column {}

impl Tables {
    pub(crate) fn new() -> Self {
        Self(Arc::new(ArcSwapAny::new(ThinArc::from_header_and_iter(
            (),
            empty(),
        ))))
    }

    pub fn len(&self) -> usize {
        self.0.load().slice.len()
    }

    pub fn get(&self, index: u32) -> Option<Table> {
        self.map(index, Table::clone)
    }

    pub(crate) fn map<T>(&self, index: u32, map: impl FnOnce(&Table) -> T) -> Option<T> {
        self.0.load().slice.get(index as usize).map(map)
    }

    pub(crate) fn find_or_add(
        &self,
        metas: impl IntoIterator<Item = Meta>,
    ) -> Result<Table, Error> {
        let metas = sort(metas).ok_or(Error::DuplicateMeta)?;
        let mut old = self.0.load();
        loop {
            match old
                .slice
                .iter()
                .find(|table| table.is(metas.iter().copied()))
            {
                Some(table) => break Ok(table.clone()),
                None => {
                    let index = old.slice.len().try_into().map_err(Error::TablesOverflow)?;
                    let table = Table::new(index, &metas);
                    let tables = ThinArc::from_header_and_iter(
                        (),
                        old.slice.iter().cloned().and(table.clone()),
                    );
                    let new = self.0.compare_and_swap(&*old, tables);
                    if old.as_raw() == new.as_raw() {
                        break Ok(table);
                    } else {
                        old = new;
                    }
                }
            }
        }
    }
}

impl Default for Tables {
    fn default() -> Self {
        Self::new()
    }
}

impl Column {
    #[inline]
    pub(crate) const fn new(meta: Meta) -> Self {
        Self {
            meta,
            data: RwLock::new(NonNull::dangling()),
        }
    }

    #[inline]
    pub const fn meta(&self) -> Meta {
        self.meta
    }

    #[inline]
    pub(crate) unsafe fn lock(&self, access: Access) -> bool {
        if self.meta.size() == 0 {
            false
        } else {
            match access {
                Access::Read => unsafe { self.data.raw().lock_shared() },
                Access::Write => unsafe { self.data.raw().lock_exclusive() },
            }
            true
        }
    }

    #[inline]
    pub(crate) unsafe fn unlock(&self, access: Access) -> bool {
        if self.meta.size() == 0 {
            false
        } else {
            match access {
                Access::Read => unsafe { self.data.raw().unlock_shared() },
                Access::Write => unsafe { self.data.raw().unlock_exclusive() },
            }
            true
        }
    }

    #[inline]
    pub(crate) unsafe fn get<T: 'static>(&self, count: u32) -> &[T] {
        debug_assert_eq!(self.meta.identifier(), TypeId::of::<T>());
        unsafe { from_raw_parts(self.data().cast::<T>().as_ptr(), count as usize) }
    }

    #[inline]
    pub(crate) unsafe fn get_mut<T: 'static>(&self, count: u32) -> &mut [T] {
        debug_assert_eq!(self.meta.identifier(), TypeId::of::<T>());
        unsafe { from_raw_parts_mut(self.data().cast::<T>().as_ptr(), count as usize) }
    }

    #[inline]
    pub(crate) unsafe fn get_in(&self, slice: &mut Slice, count: u32) {
        debug_assert_eq!(self.meta, slice.meta());
        unsafe { slice.set_parts(self.data(), count as _) };
    }

    #[inline]
    pub(crate) unsafe fn set_at<T: 'static>(&self, item: T, row: u32) {
        debug_assert_eq!(self.meta.identifier(), TypeId::of::<T>());
        unsafe { self.data().cast::<T>().add(row as usize).write(item) };
    }

    #[inline]
    pub(crate) unsafe fn set_at_with(&self, item: Box<dyn Any>, row: u32) {
        debug_assert_eq!(self.meta.identifier(), item.type_id());
        unsafe { self.meta.set_at(self.data(), item, row) };
    }

    #[inline]
    pub(crate) unsafe fn copy_at(&self, source: u32, target: u32, count: u32) -> bool {
        let data = unsafe { self.data() };
        unsafe { self.meta.copy_at((data, source), (data, target), count) }
    }

    #[inline]
    pub(crate) unsafe fn drop_at(&self, row: u32, count: u32) -> bool {
        unsafe { self.meta.drop_at(self.data(), row, count) }
    }

    unsafe fn data(&self) -> NonNull<u8> {
        unsafe { *self.data.data_ptr() }
    }
}

impl<'a> Row<'a> {
    pub const fn row(&self) -> u32 {
        self.row
    }

    pub fn table(&self) -> u32 {
        self.table.index()
    }

    pub fn remove(&mut self) {
        self.remove.borrow_mut().push(self.row)
    }
}

impl PartialEq for Row<'_> {
    fn eq(&self, other: &Self) -> bool {
        (self.row, self.table) == (other.row, other.table)
    }
}

impl Eq for Row<'_> {}

impl PartialOrd for Row<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        (self.table, self.row).partial_cmp(&(other.table, other.row))
    }
}

impl Ord for Row<'_> {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        (self.table, self.row).cmp(&(other.table, other.row))
    }
}

impl<'a> Rows<'a> {
    pub fn remove(&mut self) {
        self.remove.borrow_mut().extend(self.rows.clone());
    }

    fn row(&mut self, row: u32) -> Row<'a> {
        Row {
            row,
            table: self.table,
            remove: self.remove,
        }
    }
}

impl<'a> Iterator for Rows<'a> {
    type Item = Row<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        self.rows.next().map(|row| self.row(row))
    }
}

impl ExactSizeIterator for Rows<'_> {
    fn len(&self) -> usize {
        self.rows.len()
    }
}

impl DoubleEndedIterator for Rows<'_> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.rows.next_back().map(|row| self.row(row))
    }
}

impl FusedIterator for Rows<'_> {}

impl Table {
    pub(super) fn new(index: u32, metas: &[Meta]) -> Self {
        Self(ThinArc::from_header_and_iter(
            Header {
                index,
                count: AtomicU32::new(0),
                unlocked: Condvar::new(),
                resolved: Condvar::new(),
                state: Mutex::new(State {
                    lock: 0,
                    count: 0,
                    capacity: 0,
                    remove: Vec::new(),
                }),
            },
            metas.iter().copied().map(Column::new),
        ))
    }

    pub(crate) fn address(&self) -> usize {
        self.0.as_ptr().addr()
    }

    pub(crate) fn column(&self, identifier: TypeId) -> Option<u32> {
        utility::find(&self.0.slice, identifier, |column| column.meta.identifier())?
            .try_into()
            .ok()
    }

    pub fn columns(&self) -> &[Column] {
        &self.0.slice
    }

    pub fn index(&self) -> u32 {
        self.header().index
    }

    pub fn count(&self) -> u32 {
        self.header().count.load(Ordering::Acquire)
    }

    pub(crate) unsafe fn rows<'a>(&'a self, count: u32, remove: &'a RefCell<Vec<u32>>) -> Rows<'a> {
        Rows {
            rows: 0..count,
            table: self,
            remove,
        }
    }

    pub(crate) fn is(&self, metas: impl IntoIterator<Item = Meta>) -> bool {
        self.columns().iter().map(|column| column.meta()).eq(metas)
    }

    pub(crate) unsafe fn lock(&self, locks: impl IntoIterator<Item = Lock>) -> Option<u32> {
        let header = self.header();
        let columns = self.columns();
        let mut count = None;
        for lock in locks {
            match lock {
                Lock::Rows => {
                    let mut guard = header.state.lock();
                    header
                        .resolved
                        .wait_while(&mut guard, |state| state.remove.len() > 0);
                    guard.lock += 1;
                    count = Some(guard.count);
                }
                Lock::Column(column, access) => {
                    if let Some(column) = columns.get(column as usize) {
                        unsafe { column.lock(access) };
                    }
                }
            }
        }
        count
    }

    pub(crate) unsafe fn unlock(
        &self,
        locks: impl IntoIterator<Item = Lock>,
        remove: &mut Vec<u32>,
    ) -> bool {
        let header = self.header();
        let columns = self.columns();
        let mut rows = false;
        for lock in locks {
            match lock {
                // `Rows` unlocking must be deferred otherwise it may cause a deadlock condition
                // with `Self::insert` if this thread holds a lock on column 'A' and tries to take
                // the state lock while another thread in `Self::insert` holds the state lock and
                // tries to lock column 'A' (in `Self::ensure`).
                Lock::Rows => rows = true,
                Lock::Column(column, access) => {
                    if let Some(column) = columns.get(column as usize) {
                        unsafe { column.unlock(access) };
                    }
                }
            }
        }
        if rows {
            let mut guard = header.state.lock();
            guard.lock -= 1;
            guard.remove.append(remove);
            let lock = guard.lock;
            let resolve = guard.remove.len() > 0;
            drop(guard);
            if lock == 0 {
                header.unlocked.notify_all();
            }
            resolve
        } else {
            false
        }
    }

    pub(crate) fn insert<F: FnOnce(u32)>(&self, count: u32, apply: F) -> Result<(), Error> {
        if count == 0 {
            return Ok(());
        }

        let header = self.header();
        let columns = self.columns();
        let mut state = header.state.lock();
        let start = state.count;
        let end = start.checked_add(count).ok_or(Error::TableOverflow)?;
        Self::ensure(&mut state, columns, end)?;
        apply(start);
        state.count = end;
        header.count.store(state.count, Ordering::Release);
        Ok(())
    }

    pub(crate) fn resolve(&self) -> Result<(), Error> {
        let header = self.header();
        let columns = self.columns();
        let mut state = header.state.lock();
        state.remove.sort_unstable_by_key(|&row| Reverse(row));
        header
            .unlocked
            .wait_while(&mut state, |state| state.lock > 0);
        if state.remove.is_empty() {
            return Ok(());
        }

        for column in columns {
            unsafe { column.lock(Access::Write) };
        }
        let guard = defer(|| {
            for column in columns {
                unsafe { column.unlock(Access::Write) };
            }
        });
        {
            let state = &mut *state;
            for chunk in state
                .remove
                .chunk_by(|&left, &right| left.saturating_sub(right) <= 1)
            {
                if let Some(&end) = chunk.first()
                    && let Some(&start) = chunk.last()
                {
                    let count = (end - start).checked_add(1).ok_or(Error::TableUnderflow)?;
                    debug_assert!(start <= end);
                    debug_assert!(start < state.count);
                    debug_assert!(end < state.count);
                    debug_assert!(count <= state.count);
                    state.count = state
                        .count
                        .checked_sub(count)
                        .ok_or(Error::TableUnderflow)?;
                    for column in columns {
                        unsafe { column.drop_at(start, count) };
                        if start < state.count {
                            unsafe { column.copy_at(state.count, start, count) };
                        }
                    }
                }
            }
        }
        drop(guard);
        state.remove.clear();
        header.count.store(state.count, Ordering::Release);
        drop(state);
        header.resolved.notify_all();
        Ok(())
    }

    fn ensure(state: &mut State, columns: &[Column], count: u32) -> Result<(), Error> {
        if count <= state.capacity {
            return Ok(());
        }

        let capacity = count
            .checked_next_power_of_two()
            .ok_or(Error::TableOverflow)?;
        let new_layout = columns
            .iter()
            .try_fold(Layout::new::<()>(), |layout, column| {
                Ok(column.meta.extend(layout, capacity)?.0)
            })?;
        // TODO: Restore a valid state if an error occurs after allocation and/or while
        // some columns have been updated.
        let new_data = unsafe { allocate(new_layout.pad_to_align())? };
        let mut old_layout = Layout::new::<()>();
        let mut new_layout = Layout::new::<()>();
        let mut old_data = NonNull::dangling();
        for column in columns {
            let old_pair = column.meta.extend(old_layout, state.capacity)?;
            let new_pair = column.meta.extend(new_layout, capacity)?;
            let target = unsafe { new_data.add(new_pair.1) };
            let mut source = column.data.write();
            unsafe { column.meta.copy(*source, target, state.count) };
            *source = target;
            old_data = unsafe { source.sub(old_pair.1) };
            old_layout = old_pair.0;
            new_layout = new_pair.0;
        }
        unsafe { deallocate(old_data, old_layout.pad_to_align()) };
        state.capacity = capacity;
        Ok(())
    }

    fn header(&self) -> &Header {
        &self.0.header.header
    }
}

impl PartialEq for Table {
    fn eq(&self, other: &Self) -> bool {
        self.address().eq(&other.address())
    }
}

impl Eq for Table {}

impl PartialOrd for Table {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        self.address().partial_cmp(&other.address())
    }
}

impl Ord for Table {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        self.address().cmp(&other.address())
    }
}

impl Drop for Table {
    fn drop(&mut self) {
        let _ = self.0.with_arc_mut(|table| {
            if let Some(table) = Arc::get_mut(table) {
                let header = table.header_mut();
                let state = header.state.get_mut();
                let count = state.count;
                let capacity = state.capacity;
                let mut old_data = NonNull::dangling();
                let mut old_layout = Layout::new::<()>();
                for column in table.slice_mut() {
                    let pair = column.meta.extend(old_layout, capacity)?;
                    let data = *column.data.get_mut();
                    unsafe { column.meta.drop(data, count) };
                    old_data = unsafe { data.sub(pair.1) };
                    old_layout = pair.0;
                }
                unsafe { deallocate(old_data, old_layout.pad_to_align()) };
                Ok::<_, Error>(true)
            } else {
                Ok::<_, Error>(false)
            }
        });
    }
}

fn sort<T: Ord>(items: impl IntoIterator<Item = T>) -> Option<Vec<T>> {
    let mut items = items.into_iter().collect::<Vec<_>>();
    items.sort_unstable();
    if is_unique(&items) { Some(items) } else { None }
}
