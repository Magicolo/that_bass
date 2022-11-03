use parking_lot::{Mutex, RwLock};
use std::{
    any::{Any, TypeId},
    cell::RefCell,
    collections::HashMap,
    hash::Hash,
    ops::Deref,
    rc::Rc,
    sync::Arc,
    thread::{self, ThreadId},
};

use crate::Error;

pub struct Resources {
    locals: Mutex<Locals>,
    globals: Mutex<Globals>,
}

pub struct Local<T>(Rc<RefCell<T>>);
pub struct Global<T>(Arc<RwLock<T>>);

struct Locals<K = (ThreadId, TypeId)>(HashMap<K, Result<Rc<dyn Any>, Error>>);
struct Globals<K = TypeId>(HashMap<K, Result<Arc<dyn Any + Sync + Send>, Error>>);

impl<K: Eq + Hash> Globals<K> {
    pub fn new() -> Self {
        Self(HashMap::new())
    }

    pub fn try_get<T: Send + Sync + 'static>(
        &mut self,
        key: K,
        default: impl FnOnce() -> Result<T, Error>,
    ) -> Result<Global<T>, Error> {
        Ok(Global(
            self.0
                .entry(key)
                .or_insert_with(|| Ok(Arc::new(RwLock::new(default()?))))
                .clone()?
                .downcast()
                .map_err(|_| Error::InvalidType)?,
        ))
    }

    pub fn get<T: Send + Sync + 'static>(
        &mut self,
        key: K,
        default: impl FnOnce() -> T,
    ) -> Global<T> {
        match self.0.get_mut(&key) {
            Some(result) => {
                if let Ok(resource) = result {
                    if let Ok(resource) = resource.clone().downcast() {
                        return Global(resource);
                    }
                }
                let resource = Arc::new(RwLock::new(default()));
                *result = Ok(resource.clone());
                Global(resource)
            }
            None => {
                let resource = Arc::new(RwLock::new(default()));
                self.0.insert(key, Ok(resource.clone()));
                Global(resource)
            }
        }
    }
}

impl<K: Eq + Hash> Locals<K> {
    pub fn new() -> Self {
        Self(HashMap::new())
    }

    pub fn try_get<T: 'static>(
        &mut self,
        key: K,
        default: impl FnOnce() -> Result<T, Error>,
    ) -> Result<Local<T>, Error> {
        Ok(Local(
            self.0
                .entry(key)
                .or_insert_with(|| Ok(Rc::new(RefCell::new(default()?))))
                .clone()?
                .downcast()
                .map_err(|_| Error::InvalidType)?,
        ))
    }

    pub fn get<T: 'static>(&mut self, key: K, default: impl FnOnce() -> T) -> Local<T> {
        match self.0.get_mut(&key) {
            Some(result) => {
                if let Ok(resource) = result {
                    if let Ok(resource) = resource.clone().downcast() {
                        return Local(resource);
                    }
                }
                let resource = Rc::new(RefCell::new(default()));
                *result = Ok(resource.clone());
                Local(resource)
            }
            None => {
                let resource = Rc::new(RefCell::new(default()));
                self.0.insert(key, Ok(resource.clone()));
                Local(resource)
            }
        }
    }
}

impl Resources {
    pub fn new() -> Self {
        Self {
            locals: Mutex::new(Locals::new()),
            globals: Mutex::new(Globals::new()),
        }
    }

    pub fn global<T: Send + Sync + 'static>(&self, default: impl FnOnce() -> T) -> Global<T> {
        self.globals.lock().get(TypeId::of::<T>(), default)
    }

    pub fn try_global<T: Send + Sync + 'static>(
        &self,
        default: impl FnOnce() -> Result<T, Error>,
    ) -> Result<Global<T>, Error> {
        self.globals.lock().try_get(TypeId::of::<T>(), default)
    }

    pub fn global_with<K: Eq + Hash + Send + Sync + 'static, T: Send + Sync + 'static>(
        &self,
        key: K,
        default: impl FnOnce() -> T,
    ) -> Global<T> {
        self.global(Globals::new)
            .write()
            .get((key, TypeId::of::<T>()), default)
    }

    pub fn try_global_with<K: Eq + Hash + Send + Sync + 'static, T: Send + Sync + 'static>(
        &self,
        key: K,
        default: impl FnOnce() -> Result<T, Error>,
    ) -> Result<Global<T>, Error> {
        self.global(Globals::new)
            .write()
            .try_get((key, TypeId::of::<T>()), default)
    }

    pub fn local<T: 'static>(&self, default: impl FnOnce() -> T) -> Local<T> {
        self.locals
            .lock()
            .get((thread::current().id(), TypeId::of::<T>()), default)
    }

    pub fn try_local<T: 'static>(
        &self,
        default: impl FnOnce() -> Result<T, Error>,
    ) -> Result<Local<T>, Error> {
        self.locals
            .lock()
            .try_get((thread::current().id(), TypeId::of::<T>()), default)
    }

    pub fn local_with<K: Eq + Hash + 'static, T: 'static>(
        &self,
        key: K,
        default: impl FnOnce() -> T,
    ) -> Local<T> {
        self.local(Locals::new)
            .borrow_mut()
            .get((key, TypeId::of::<T>()), default)
    }

    pub fn try_local_with<K: Eq + Hash + 'static, T: 'static>(
        &self,
        key: K,
        default: impl FnOnce() -> Result<T, Error>,
    ) -> Result<Local<T>, Error> {
        self.local(Locals::new)
            .borrow_mut()
            .try_get((key, TypeId::of::<T>()), default)
    }
}

impl<T> Deref for Local<T> {
    type Target = RefCell<T>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> Clone for Local<T> {
    #[inline]
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<T> Deref for Global<T> {
    type Target = RwLock<T>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> Clone for Global<T> {
    #[inline]
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

unsafe impl<K: Send> Send for Locals<K> {}
