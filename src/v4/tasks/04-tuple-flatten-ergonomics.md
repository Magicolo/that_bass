# Task 04: Flatten Query/Module Items via `Tuple::normalize`

## Goal

Improve the ergonomics of destructuring query results and module items by
converting from the nested `Push`-produced form `(A, (B, (C, ())))` to the
flat form `(A, B, C)`.

## Current State

### How items are produced

`Push` builds nested right-associated tuples:
```
().push(a).push(b).push(c)  →  (a, (b, (c, ())))
```

This is how `query::Module` builds its access chain:
```rust
Query::build().read::<char>().write::<String>().read::<u8>()
// Internal: (Read<char>, (Write<String>, (Read<u8>, ())))
```

And the composed `Module::Item` type for this chain is:
```rust
type Item<'a> = (&'a [char], (&'a mut [String], (&'a [u8], ())));
```

### The ergonomic problem

Currently, to access all items from a `State` chain, you write:
```rust
let (chars, rest) = state.next();
let (strings, rest) = rest.next();
let (u8s, rest) = rest.next();
let (_, _) = rest.next();
```

Or with an iterator:
```rust
for (chars, (strings, (u8s, ()))) in query.iter() {
    // ...
}
```

The nested destructuring is noisy.  You want:
```rust
for (chars, strings, u8s) in query.iter() {
    // Works
}
```

### The `Tuple` trait (already macro-generated)

```rust
pub trait Tuple {
    type Normal;
    fn normalize(self) -> Self::Normal;
    fn flatten(tuple: Self::Normal) -> Self;
}
```

- `normalize`: flat form → nested form.  `(A, B, C).normalize()` → `(A, (B, (C, ())))`
- `flatten`: nested form → flat form.  `(A, (B, (C, ()))).flatten()` → `(A, B, C)`

The trait is already implemented for tuples up to 32 elements via the
`tuple!` macro in `utility.rs:269-301`.

### `Normal` form examples (generated)

| Flat tuple | `normalize()` → `Normal` type |
|------------|-------------------------------|
| `(A,)` | `(A, ())` |
| `(A, B)` | `(A, (B, ()))` |
| `(A, B, C)` | `(A, (B, (C, ())))` |
| `(A, B, C, D)` | `(A, (B, (C, (D, ()))))` |

## Where Flattening Should Apply

### 1. `Query::iter()` — flatten the `Item` type

Currently `Iterator<A::Item>` yields nested tuples.  The iterator should
produce flat tuples:

```rust
impl<'a, A: Access> iter::Iterator for Iterator<'a, A>
where
    A::Item<'a>: Tuple,  // <--- new bound
{
    type Item = <A::Item<'a> as Tuple>::Normal;

    fn next(&mut self) -> Option<Self::Item> {
        let item = /* existing logic */?;
        Some(item.flatten())
    }
}
```

Wait — `flatten` takes `Self::Normal` and returns `Self`.  Let's re-read:

```rust
impl<T0, T1> Tuple for (T0, T1) {
    type Normal = (T0, (T1, ()));  // flat = (T0, T1), Normal = nested right-assoc
    fn normalize(self) -> Self::Normal { /* (T0, T1) → (T0, (T1, ())) */ }
    fn flatten(tuple: Self::Normal) -> Self { /* (T0, (T1, ())) → (T0, T1) */ }
}
```

So:
- `normalize`: `(T0, T1)` → `(T0, (T1, ()))`  (flat → nested)
- `flatten`: `(T0, (T1, ()))` → `(T0, T1)`  (nested → flat)

The `Push` chain produces nested form.  The iterator's `Item` type is nested.
We want to convert nested → flat before yielding:

```rust
impl<'a, A: Access> iter::Iterator for Iterator<'a, A>
where
    A::Item<'a>: Tuple,
{
    type Item = <A::Item<'a> as Tuple>::Normal;  // the flat form

    fn next(&mut self) -> Option<Self::Item> {
        let (table, state) = self.states.next()?;
        let table = unsafe { self.tables.get_unchecked(*table as usize) };
        let item = self.query.get(state, table);
        Some(item.normalize())  // nested → flat
    }
}
```

Wait — `normalize` goes flat → nested.  `flatten` goes nested → flat.
The iterator receives nested from `Access::get`.  So it should call
`flatten`.  Let's check the signatures:

`normalize(self) -> Self::Normal` — takes `Self` (the flat form), returns `Normal` (nested).
So if `Self = (T0, T1)` (flat), `Self::Normal = (T0, (T1, ()))` (nested).

`flatten(tuple: Self::Normal) -> Self` — takes `Normal` (nested), returns `Self` (flat).

Since the iterator gets nested form, and we want flat form, we need
`flatten`.  But `flatten` takes `Self::Normal`, not `Self`.  This means we
need to call it on the correct type.

The issue: the nested form IS the `Self` for some nesting level.

For `(Read<char>, (Write<String>, (Read<u8>, ())))`:
- This is `(T0, (T1, (T2, ())))` = the nested form
- It IS the `Normal` form of `(T0, T1, T2)`.

So we can't call `flatten` on it directly — we'd need to go through the
triple tuple impl.  The trait is structured so that:

```
(A, (B, (C, ()))).normalize() // doesn't make sense — it IS the Normal form
```

Actually wait, `Tuple` is implemented for each flat tuple AND its nested
equivalent is a different type.  For `(A, (B, (C, ())))`:

Looking at the macro again:
```rust
impl<T0, T1, T2> Tuple for (T0, T1, T2) {
    type Normal = (T0, (T1, (T2, ())));
    fn normalize(self) -> Self::Normal { /* produces (T0, (T1, (T2, ()))) */ }
    fn flatten(tuple: Self::Normal) -> Self { /* pattern-matches (T0, (T1, (T2, ()))) → (T0, T1, T2) */ }
}
```

So `flatten` is a static method on the *flat* type that takes the *nested*
form and returns the flat form.  Usage:

```rust
let nested: (T0, (T1, (T2, ()))) = /* from Access::get */;
let flat: (T0, T1, T2) = <(T0, T1, T2) as Tuple>::flatten(nested);
```

This is a bit awkward to call generically.  An extension trait or method
would help:

```rust
pub trait Flatten<T> {
    fn flatten_into(self) -> T;
}

impl<T: Tuple> Flatten<T> for T::Normal {
    fn flatten_into(self) -> T { T::flatten(self) }
}
```

Then: `item.flatten_into::<the-flat-type>()`.

But we don't want to specify the flat type explicitly.  We need the
flattening to be automatic.  Two approaches:

**Approach A**: Add a method on the nested type that knows its flat form
through an associated type.

**Approach B**: Change the `Tuple` trait so `flatten` is an instance method
on `Normal`, not a static method on the flat type.

Approach B requires re-macro-ing but is cleaner:

```rust
// Reverse: make flatten an instance method
pub trait IntoFlat {
    type Flat;
    fn flatten(self) -> Self::Flat;
}
```

But this loses the symmetry with `normalize`.  Let's step back.

**Approach C** (simplest): Add a free function or helper trait that takes
the nested type and infers the flat type:

```rust
pub trait Tuple {
    type Normal;
    fn normalize(self) -> Self::Normal;
    fn flatten(normal: Self::Normal) -> Self;
}

pub trait IntoFlat: Sized {
    type Flat;
    fn into_flat(self) -> Self::Flat;
}

// Implement for the Normal of every Tuple
impl<T: Tuple> IntoFlat for T::Normal {
    type Flat = T;
    fn into_flat(self) -> T { T::flatten(self) }
}
```

Then: `item.into_flat()` — and Rust infers `T` from the return type
or from context.

But wait, there may be multiple `Tuple` impls with the same `Normal`.  For
example, `(A, (B, ()))` could be the Normal of both `(A, B)` and of itself
if Tuple is reflexive.  Looking at the macro output...

Actually no, the macro only generates for flat tuples up to 32.  `(A, (B,
()))` is not covered by a flat impl unless it's at nesting depth 2.  Let me
check:

For `(T0, T1)`: `Normal = (T0, (T1, ()))`.  This is the nesting form.
For `(T0, (T1, ()))`: No impl generated (it's not a flat tuple, it's a pair
with a nested tail).

So `(T0, (T1, ()))` is the Normal of `(T0, T1)` and ONLY of `(T0, T1)`.
The `IntoFlat` impl is unambiguous.

### 2. `State::next()` / `Rest::next()` — flatten the yielded item

Currently `State::next()` yields `(H::Item<'b>, Rest<'b, T>)`.  The
`H::Item<'b>` may itself be a nested tuple (if H is a composed module).  We
could allow the caller to flatten it.

However, the `next()` API is primarily for sequential access where the user
wants one module group at a time.  Flattening is less pressing here since
the destructuring is linear.

The primary beneficiary is `Query::iter()`.

## Proposed Plan

1. **Add an `IntoFlat` trait** in `utility.rs` that provides `.into_flat()`
   on any `T::Normal`:

   ```rust
   pub trait IntoFlat {
       type Flat;
       fn into_flat(self) -> Self::Flat;
   }

   impl<T: Tuple> IntoFlat for T::Normal {
       type Flat = T;
       fn into_flat(self) -> T { T::flatten(self) }
   }
   ```

2. **Update `Iterator<'a, A>`** to flatten the yielded item when the item
   type supports it:

   ```rust
   impl<'a, A: Access> iter::Iterator for Iterator<'a, A>
   where
       A::Item<'a>: IntoFlat,
   {
       type Item = <A::Item<'a> as IntoFlat>::Flat;

       fn next(&mut self) -> Option<Self::Item> {
           let (table, state) = self.states.next()?;
           let table = unsafe { self.tables.get_unchecked(*table as usize) };
           Some(self.query.get(state, table).into_flat())
       }
   }
   ```

3. **Preserve the existing impl** for non-flattenable items (like `()`,
   single-element accessors).  This may require a separate impl block or
   specialization (nightly), or we can use a blanket impl marker:

   ```rust
   // Helper: identity "flatten" for non-tuple types
   impl IntoFlat for () {
       type Flat = ();
       fn into_flat(self) -> Self { self }
   }
   ```

4. **Handle the unit type properly**.  `Access for ()` yields `()`.  `()`
   as an item should remain `()`.  The `IntoFlat for ()` impl handles this.

5. **Update the `access` test** to use flat destructuring.

## Challenges

1. **Iterator coherence**.  We need two `Iterator` impls: one for
   flattenable items, one for non-flattenable.  In stable Rust, this
   requires either:
   - A marker trait (`Flattenable`) that distinguishes the two cases.
   - A unified impl that uses a blanket identity `IntoFlat` for non-tuple
     types.
   
   The blanket identity approach is cleaner: implement `IntoFlat` for all
   non-tuple types with `type Flat = Self; fn into_flat(self) -> Self { self }`.
   Then there's one `Iterator` impl.

   But we can't do a blanket `impl<T> IntoFlat for T` because of coherence
   with `impl<T: Tuple> IntoFlat for T::Normal`.  We'd need to be selective.

   Pragmatic approach: manually implement `IntoFlat` for `()`, single-column
   types, `Rows`, `&Table`, and `&Column`.  This covers all current Access
   impls.

2. **32-element limit**.  The current `tuple!` macro goes up to 31 type
   params.  If someone pushes 32+ accessors, flattening won't work.  This is
   an acceptable limit — the v1 `Row` trait has similar macro limits.

3. **Performance impact**.  `into_flat()` is a zero-cost operation (the
   nested and flat forms have the same layout in memory — just different
   type-level structure).  The compiler should optimize it to a no-op.

4. **Readability of `Item` type**.  When users hover over the item type in
   their IDE, they should see the flat form, not the nested form.  This
   works if the `Iterator::Item` type is the flat form.

## Relation to Other Tasks

- Independent of Task 01 and 02.  Can be done in parallel.
- Task 03 doesn't affect the flattening story directly.
- The `Query` builder (`Module<A>::read()`, etc.) is unaffected — it still
  produces nested Access tuples internally.

## Success Criteria

- [ ] `IntoFlat` trait added to `utility.rs` with blanket impl for
      `T::Normal where T: Tuple`.
- [ ] `IntoFlat` manually implemented for all non-tuple `Access::Item` types
      (`()`, `&[T]`, `&mut [T]`, `Rows`, `&Table`, `&Column`).
- [ ] `Iterator<'a, A>` yields flat tuples instead of nested.
- [ ] Single-column queries (e.g. `Query::build().read::<u32>()`) still
      compile and work.
- [ ] The `access` test uses flat destructuring in its usage example.
- [ ] Build passes with no new warnings.
