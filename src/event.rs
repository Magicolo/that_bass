use crate::{core::tuples, key::Key, table::Table, Database};

/// Allows to listen to database events. These events are guaranteed to be coherent (ex. `create` always happens before
/// `destroy` for a given key).
///
/// **All listen methods should be considered as time critical** since they are called while holding table locks and may add
/// contention on many other database operations. If these events need to be processed in some way, it is recommended to queue
/// the events and defer the processing.
pub trait Listen {
    fn created(&self, keys: &[Key], table: &Table);
    fn destroyed(&self, keys: &[Key], table: &Table);
    fn added(&self, keys: &[Key], source: &Table, target: &Table);
    fn removed(&self, keys: &[Key], source: &Table, target: &Table);
}

impl<L> Database<L> {
    pub fn listen<M: Listen>(self, listen: M) -> Database<(L, M)> {
        Database {
            inner: self.inner,
            listen: (self.listen, listen),
        }
    }
}

macro_rules! tuple {
    ($n:ident, $c:expr $(, $p:ident, $t:ident, $i:tt)*) => {
        impl<$($t: Listen,)*> Listen for ($($t,)*) {
            #[inline]
            fn created(&self, _keys: &[Key], _table: &Table) {
                $(self.$i.created(_keys, _table);)*
            }
            #[inline]
            fn destroyed(&self, _keys: &[Key], _table: &Table) {
                $(self.$i.destroyed(_keys, _table);)*
            }
            #[inline]
            fn added(&self, _keys: &[Key], _source: &Table, _target: &Table) {
                $(self.$i.added(_keys, _source, _target);)*
            }
            #[inline]
            fn removed(&self, _keys: &[Key], _source: &Table, _target: &Table) {
                $(self.$i.removed(_keys, _source, _target);)*
            }
        }
    };
}
tuples!(tuple);
