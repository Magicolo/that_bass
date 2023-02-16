use crate::{
    event::{Collect, DeclareContext, Event, ProcessContext, Raw},
    key::Key,
};
use std::{collections::HashMap, ops::ControlFlow};

pub struct Trace {}

struct Buffer(HashMap<Key, Vec<u32>>);
struct Pair(Key, u32);

impl Event for Pair {
    #[inline]
    fn declare(mut context: DeclareContext) {
        context.create(true);
        context.modify(true);
    }

    #[inline]
    fn process<C: Collect<Self>>(collect: &mut C, context: ProcessContext) -> ControlFlow<()> {
        for &event in context.events() {
            match event {
                Raw::Create { keys, table } => {
                    collect.all(context.keys(keys).iter().map(|&key| Pair(key, table)))?
                }
                Raw::Modify { keys, tables } => collect.all(
                    context
                        .keys(keys)
                        .iter()
                        .map(|&key| Pair(key, tables.target)),
                )?,
                _ => {}
            }
        }
        ControlFlow::Continue(())
    }
}

impl Collect<Pair> for Buffer {
    #[inline]
    fn one(&mut self, item: Pair) -> ControlFlow<()> {
        self.0
            .entry(item.0)
            .or_insert_with(Default::default)
            .push(item.1);
        ControlFlow::Continue(())
    }

    #[inline]
    fn all<I: IntoIterator<Item = Pair>>(&mut self, items: I) -> ControlFlow<()> {
        for item in items {
            self.one(item)?;
        }
        ControlFlow::Continue(())
    }

    fn next(&mut self) -> Option<Pair> {
        todo!()
    }
}
