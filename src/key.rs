use crate::{
    query::{At, Item},
    table::Table,
};
use std::slice::from_raw_parts;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Key {
    index: u32,
    generation: u32,
}

pub struct State;

impl Key {
    pub const NULL: Self = Self {
        index: u32::MAX,
        generation: u32::MAX,
    };

    #[inline]
    pub(crate) const fn new(index: u32) -> Self {
        Self {
            index: index,
            generation: 0,
        }
    }

    #[inline]
    pub const fn index(&self) -> u32 {
        self.index
    }

    #[inline]
    pub const fn generation(&self) -> u32 {
        self.generation
    }

    #[inline]
    pub(crate) fn increment(&mut self) {
        self.generation = self.generation.saturating_add(1);
    }
}

impl Item for Key {
    type State = State;

    fn initialize(_: &Table) -> Option<Self::State> {
        Some(State)
    }
}

impl<'a> At<'a> for State {
    type State = (*const Key, usize);
    type Chunk = &'a [Key];
    type Item = Key;

    #[inline]
    fn try_get(&self, table: &Table) -> Option<Self::State> {
        Some(self.get(table))
    }
    #[inline]
    fn get(&self, table: &Table) -> Self::State {
        let keys = table.keys();
        (keys.as_ptr(), keys.len())
    }
    #[inline]
    unsafe fn chunk(state: &mut Self::State) -> Self::Chunk {
        from_raw_parts(state.0, state.1)
    }
    #[inline]
    unsafe fn item(state: &mut Self::State, index: usize) -> Self::Item {
        *Self::chunk(state).get_unchecked(index)
    }
}
