use crate::v4::{Meta, Table, Vector};
use core::{
    any::{Any, TypeId},
    marker::PhantomData,
    ptr::NonNull,
};

pub trait Template {
    type Item;
    type State;

    fn declare(&self) -> impl Iterator<Item = Meta>;
    fn initialize(&self, table: &mut Table) -> Option<Self::State>;
    fn defer(&self, state: &mut Self::State, item: Self::Item) -> bool;
    unsafe fn resolve(&self, state: &mut Self::State, table: &Table) -> bool;
}

pub struct Key;
pub struct Column<T: ?Sized>(PhantomData<T>);
pub struct ColumnWith(Meta);

pub const fn key() -> Key {
    Key
}

pub const fn column<T>() -> Column<T> {
    Column(PhantomData)
}

pub const fn column_with(meta: Meta) -> ColumnWith {
    ColumnWith(meta)
}

impl Template for Key {
    type Item = ();
    type State = ();

    fn declare(&self) -> impl Iterator<Item = Meta> {
        [].into_iter()
    }

    fn initialize(&self, table: &mut Table) -> Option<Self::State> {
        None
    }

    fn defer(&self, state: &mut Self::State, item: Self::Item) -> bool {
        true
    }

    unsafe fn resolve(&self, state: &mut Self::State, table: &Table) -> bool {
        true
    }
}

impl Template for ColumnWith {
    type Item = Box<dyn Any>;
    type State = (Vector, u32);

    fn declare(&self) -> impl Iterator<Item = Meta> {
        [self.0.clone()].into_iter()
    }

    fn initialize(&self, table: &mut Table) -> Option<Self::State> {
        Some((
            Vector::new(self.0.clone()),
            table.column(self.0.identifier)?.index(),
        ))
    }

    fn defer(&self, state: &mut Self::State, item: Self::Item) -> bool {
        state.0.push(item).is_ok()
    }

    unsafe fn resolve(&self, state: &mut Self::State, table: &Table) -> bool {
        let count = table.count();
        let column = unsafe { table.columns().get_unchecked(state.1 as usize) };
        debug_assert_eq!(self.0.identifier, column.meta.identifier);
        unsafe { state.0.move_at(column.data, count) }
    }
}

impl<T: 'static> Template for Column<T> {
    type Item = T;
    type State = (Vec<Self::Item>, u32);

    fn declare(&self) -> impl Iterator<Item = Meta> {
        [Meta::of::<T>()].into_iter()
    }

    fn initialize(&self, table: &mut Table) -> Option<Self::State> {
        Some((
            Vec::new(),
            table.column(TypeId::of::<T>())?.index().try_into().ok()?,
        ))
    }

    fn defer(&self, state: &mut Self::State, item: Self::Item) -> bool {
        state.0.push(item);
        true
    }

    unsafe fn resolve(&self, state: &mut Self::State, table: &Table) -> bool {
        if let Some(source) = NonNull::new(state.0.as_mut_ptr()) {
            if let Ok(count) = state.0.len().try_into() {
                let index = table.count();
                let column = unsafe { table.columns().get_unchecked(state.1 as usize) };
                if unsafe { column.copy(source, index, count) } {
                    unsafe { state.0.set_len(0) };
                    return true;
                }
            }
        }
        false
    }
}

macro_rules! tuple {
    ($($index:tt: $type:ident),*) => {
        impl<$($type: Template),*> Template for ($($type,)*) {
            type Item = ($($type::Item,)*);
            type State = ($($type::State,)*);

            fn declare(&self) -> impl Iterator<Item = Meta> {
                let metas = [].into_iter();
                $(let metas = metas.chain(self.$index.declare());)*
                metas
            }

            fn initialize(&self, _table: &mut Table) -> Option<Self::State> {
                Some(($(self.$index.initialize(_table)?,)*))
            }

            fn defer(&self, _state: &mut Self::State, _item: Self::Item) -> bool {
                $((self.$index.defer(&mut _state.$index, _item.$index)) &)* true
            }

            unsafe fn resolve(&self, _state: &mut Self::State, _table: &Table) -> bool {
                $((unsafe { self.$index.resolve(&mut _state.$index, _table) }) |)* true
            }
        }
    };
}

tuple!();
tuple!(0: T0);
tuple!(0: T0, 1: T1);
tuple!(0: T0, 1: T1, 2: T2);
tuple!(0: T0, 1: T1, 2: T2, 3: T3);
tuple!(0: T0, 1: T1, 2: T2, 3: T3, 4: T4);
tuple!(0: T0, 1: T1, 2: T2, 3: T3, 4: T4, 5: T5);
tuple!(0: T0, 1: T1, 2: T2, 3: T3, 4: T4, 5: T5, 6: T6);
tuple!(0: T0, 1: T1, 2: T2, 3: T3, 4: T4, 5: T5, 6: T6, 7: T7);
