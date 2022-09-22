use parking_lot::{Mutex, RwLock};
use std::{
    any::{Any, TypeId},
    cell::{RefCell, UnsafeCell},
    collections::HashMap,
    hash::Hash,
    rc::Rc,
    sync::Arc,
    thread::{self, ThreadId},
};

pub(crate) struct Resources {
    lock: Mutex<()>,
    locals: Locals,
    globals: Globals,
}

pub struct Local<T>(Rc<RefCell<T>>);
pub struct Global<T>(Arc<RwLock<T>>);

struct Locals<K = (ThreadId, TypeId)>(UnsafeCell<HashMap<K, Rc<dyn Any>>>);
struct Globals<K = TypeId>(UnsafeCell<HashMap<K, Arc<dyn Any + Sync + Send>>>);

impl Resources {
    pub fn new() -> Self {
        Self {
            lock: Mutex::new(()),
            locals: Locals(HashMap::new().into()),
            globals: Globals(HashMap::new().into()),
        }
    }

    pub fn global<T: Send + Sync + 'static>(&self, default: impl FnOnce() -> T) -> Global<T> {
        let key = TypeId::of::<T>();
        let lock = self.lock.lock();
        let resources = unsafe { &mut *self.globals.0.get() };
        let resource = resources
            .entry(key)
            .or_insert_with(|| Arc::new(RwLock::new(default())))
            .clone();
        drop(lock);
        Global(resource.downcast().expect("Expected proper type."))
    }

    pub fn global_with<K: Eq + Hash + Send + Sync + 'static, T: Send + Sync + 'static>(
        &self,
        key: K,
        default: impl FnOnce() -> T,
    ) -> Global<T> {
        self.global(|| Globals::<K>(HashMap::new().into()))
            .write(|resources| {
                let resources = resources.0.get_mut();
                let resource = resources
                    .entry(key)
                    .or_insert_with(|| Arc::new(RwLock::new(default())))
                    .clone();
                Global(resource.downcast().expect("Expected valid type."))
            })
    }

    pub fn local<T: 'static>(&self, default: impl FnOnce() -> T) -> Local<T> {
        let key = (thread::current().id(), TypeId::of::<T>());
        let lock = self.lock.lock();
        let resources = unsafe { &mut *self.locals.0.get() };
        let resource = resources
            .entry(key)
            .or_insert_with(|| Rc::new(RefCell::new(default())))
            .clone();
        drop(lock);
        Local(resource.downcast().expect("Expected proper type."))
    }

    pub fn local_with<K: Eq + Hash + 'static, T: 'static>(
        &self,
        key: K,
        default: impl FnOnce() -> T,
    ) -> Local<T> {
        self.local(|| Locals::<K>(HashMap::new().into()))
            .write(|resources| {
                let resources = resources.0.get_mut();
                let resource = resources
                    .entry(key)
                    .or_insert_with(|| Rc::new(RefCell::new(default())))
                    .clone();
                Local(resource.downcast().expect("Expected valid type."))
            })
    }
}

impl<T> Local<T> {
    #[inline]
    pub fn read<U>(&self, read: impl FnOnce(&T) -> U) -> U {
        read(&mut self.0.borrow())
    }

    #[inline]
    pub fn write<U>(&self, write: impl FnOnce(&mut T) -> U) -> U {
        write(&mut self.0.borrow_mut())
    }
}

impl<T> Clone for Local<T> {
    #[inline]
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<T> Global<T> {
    #[inline]
    pub fn read<U>(&self, read: impl FnOnce(&T) -> U) -> U {
        read(&mut self.0.read())
    }

    #[inline]
    pub fn write<U>(&self, write: impl FnOnce(&mut T) -> U) -> U {
        write(&mut self.0.write())
    }
}

impl<T> Clone for Global<T> {
    #[inline]
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

unsafe impl<K: Send> Send for Locals<K> {}
unsafe impl<K: Sync> Sync for Locals<K> {}
unsafe impl<K: Send> Send for Globals<K> {}
unsafe impl<K: Sync> Sync for Globals<K> {}
