use crate::{Database, Error};
use parking_lot::Mutex;
use std::{
    any::{Any, TypeId},
    cell::RefCell,
    collections::HashMap,
    hash::Hash,
    rc::Rc,
    sync::Arc,
    thread::{self, ThreadId},
};

pub struct Resources {
    locals: Mutex<Locals>,
    globals: Mutex<Globals>,
}

struct Locals<K = (ThreadId, TypeId)>(HashMap<K, Result<Rc<dyn Any>, Error>>);
struct Globals<K = TypeId>(HashMap<K, Result<Arc<dyn Any + Sync + Send>, Error>>);

impl Database {
    #[inline]
    pub const fn resources(&self) -> &Resources {
        &self.resources
    }
}

impl<K: Eq + Hash> Globals<K> {
    pub fn new() -> Self {
        Self(HashMap::new())
    }

    pub fn try_get<T: Send + Sync + 'static>(
        &mut self,
        key: K,
        default: impl FnOnce() -> Result<T, Error>,
    ) -> Result<Arc<T>, Error> {
        self.0
            .entry(key)
            .or_insert_with(|| Ok(Arc::new(default()?)))
            .clone()?
            .downcast()
            .map_err(|_| Error::InvalidType(TypeId::of::<T>()))
    }

    pub fn get<T: Send + Sync + 'static>(&mut self, key: K, default: impl FnOnce() -> T) -> Arc<T> {
        match self.0.get_mut(&key) {
            Some(result) => {
                if let Ok(resource) = result {
                    if let Ok(resource) = resource.clone().downcast() {
                        return resource;
                    }
                }
                let resource = Arc::new(default());
                *result = Ok(resource.clone());
                resource
            }
            None => {
                let resource = Arc::new(default());
                self.0.insert(key, Ok(resource.clone()));
                resource
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
    ) -> Result<Rc<T>, Error> {
        self.0
            .entry(key)
            .or_insert_with(|| Ok(Rc::new(default()?)))
            .clone()?
            .downcast()
            .map_err(|_| Error::InvalidType(TypeId::of::<T>()))
    }

    pub fn get<T: 'static>(&mut self, key: K, default: impl FnOnce() -> T) -> Rc<T> {
        match self.0.get_mut(&key) {
            Some(result) => {
                if let Ok(resource) = result {
                    if let Ok(resource) = resource.clone().downcast() {
                        return resource;
                    }
                }
                let resource = Rc::new(default());
                *result = Ok(resource.clone());
                resource
            }
            None => {
                let resource = Rc::new(default());
                self.0.insert(key, Ok(resource.clone()));
                resource
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

    pub fn global<T: Send + Sync + 'static>(&self, default: impl FnOnce() -> T) -> Arc<T> {
        self.globals.lock().get(TypeId::of::<T>(), default)
    }

    pub fn try_global<T: Send + Sync + 'static>(
        &self,
        default: impl FnOnce() -> Result<T, Error>,
    ) -> Result<Arc<T>, Error> {
        self.globals.lock().try_get(TypeId::of::<T>(), default)
    }

    pub fn global_with<K: Eq + Hash + Send + Sync + 'static, T: Send + Sync + 'static>(
        &self,
        key: K,
        default: impl FnOnce() -> T,
    ) -> Arc<T> {
        self.global(|| Mutex::new(Globals::new()))
            .lock()
            .get((key, TypeId::of::<T>()), default)
    }

    pub fn try_global_with<K: Eq + Hash + Send + Sync + 'static, T: Send + Sync + 'static>(
        &self,
        key: K,
        default: impl FnOnce() -> Result<T, Error>,
    ) -> Result<Arc<T>, Error> {
        self.global(|| Mutex::new(Globals::new()))
            .lock()
            .try_get((key, TypeId::of::<T>()), default)
    }

    pub fn local<T: 'static>(&self, default: impl FnOnce() -> T) -> Rc<T> {
        self.locals
            .lock()
            .get((thread::current().id(), TypeId::of::<T>()), default)
    }

    pub fn try_local<T: 'static>(
        &self,
        default: impl FnOnce() -> Result<T, Error>,
    ) -> Result<Rc<T>, Error> {
        self.locals
            .lock()
            .try_get((thread::current().id(), TypeId::of::<T>()), default)
    }

    pub fn local_with<K: Eq + Hash + 'static, T: 'static>(
        &self,
        key: K,
        default: impl FnOnce() -> T,
    ) -> Rc<T> {
        self.local(|| RefCell::new(Locals::new()))
            .borrow_mut()
            .get((key, TypeId::of::<T>()), default)
    }

    pub fn try_local_with<K: Eq + Hash + 'static, T: 'static>(
        &self,
        key: K,
        default: impl FnOnce() -> Result<T, Error>,
    ) -> Result<Rc<T>, Error> {
        self.local(|| RefCell::new(Locals::new()))
            .borrow_mut()
            .try_get((key, TypeId::of::<T>()), default)
    }
}

impl Default for Resources {
    fn default() -> Self {
        Self::new()
    }
}

unsafe impl<K: Send> Send for Locals<K> {}
