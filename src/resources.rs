use std::{
    any::{Any, TypeId},
    cell::UnsafeCell,
    collections::HashMap,
    sync::Arc,
};

use parking_lot::{RwLock, RwLockUpgradableReadGuard};

pub(crate) struct Resources {
    indices: RwLock<HashMap<TypeId, usize>>,
    resources: UnsafeCell<Vec<Arc<dyn Any + Sync + Send>>>,
}

impl Resources {
    pub fn new() -> Self {
        Self {
            indices: RwLock::new(HashMap::new()),
            resources: Vec::new().into(),
        }
    }

    pub fn get_or_add<T: Sync + Send + 'static>(&self, default: impl FnOnce() -> T) -> Arc<T> {
        let key = TypeId::of::<T>();
        let indices_upgrade = self.indices.upgradable_read();
        match indices_upgrade.get(&key) {
            Some(&index) => {
                let resources = unsafe { &mut *self.resources.get() };
                let resource = unsafe { resources.get_unchecked(index) }.clone();
                drop(indices_upgrade);
                unsafe { resource.downcast_unchecked() }
            }
            None => {
                let resources = unsafe { &mut *self.resources.get() };
                let resource = Arc::new(default());
                let mut indices_write = RwLockUpgradableReadGuard::upgrade(indices_upgrade);
                indices_write.insert(key, resources.len());
                resources.push(resource.clone());
                drop(indices_write);
                resource
            }
        }
    }
}

unsafe impl Send for Resources {}
unsafe impl Sync for Resources {}
