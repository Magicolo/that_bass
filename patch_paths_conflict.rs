<<<<<<< SEARCH
/// Checks if two dependency paths conflict.
///
/// Two paths conflict if they share the exact same resources up to the length of the shorter path,
/// and at any matched level (including the prefix end), the accesses conflict.
pub fn paths_conflict(left: &[Dependency], right: &[Dependency]) -> bool {
    let min_len = left.len().min(right.len());

    for i in 0..min_len {
        let l = &left[i];
        let r = &right[i];

        if l.resource != r.resource {
            return false;
        }

        if l.access.conflicts_with(r.access) {
            return true;
        }
    }

    // If we reached here, the paths matched exactly up to the length of the shorter one.
    // If they matched, but no explicit conflict was found at each step? Wait.
    // E.g. left = [Read(Store)], right = [Write(Store), Write(Table(1))].
    // At index 0, Read vs Write conflicts. Handled above.
    // E.g. left = [Read(Store), Read(Table(1))], right = [Read(Store), Write(Table(1))].
    // At index 0, Read vs Read. At index 1, Read vs Write -> conflict.
    // E.g. left = [Read(Store)], right = [Read(Store), Write(Table(1))].
    // At index 0, Read vs Read. No conflict. Then loop ends. But wait!
    // Does `Read(Store)` conflict with `Write(Table)`? YES. Because writing to a table inside the store mutates the store!
    // But `Read(Store)` vs `Write(Table(1))` -> if someone locks the whole Store for reading, you can't write to a Table.
    // So if the prefix matched, does a shorter path implicitly conflict with a longer path's write?
    // Let's refine: If the shorter path requested Read, does it conflict with a later Write in the longer path?
    // YES. `Read(Store)` means "I am reading the whole store". `Write(Table)` requires `Read(Store)` + `Write(Table)`? Wait.
    // If the writer requested `[Read(Store), Write(Table)]`, and reader requested `[Read(Store)]`:
    // The shorter one (`Read(Store)`) doesn't conflict at `Store` level with `Read(Store)`.
    // But conceptually, reading the whole store means reading all tables. So writing one table conflicts with reading the whole store.
    // If so, the writer path should probably request `Write(Store)` if it modifies structural store things, OR
    // if a reader requests `Read(Store)`, it means read *everything*.
    // However, in our model, `Write(Table)` jobs declare `[Read(Store), Write(Table)]`.
    // If someone wants to read everything, they declare `[Read(Store)]`?
    // If `[Read(Store)]` conflicts with `[Read(Store), Write(Table)]`, then ANY write to a table conflicts with ANY read of the store.
    // Let's just compare them and if the shorter path has ANY access, it overlaps with the longer path.
    // Wait, if left=[Read(Store), Read(Table)] and right=[Read(Store), Write(Table), Write(Column)].
    // At Store: Read vs Read.
    // At Table: Read vs Write -> conflict! True.
    // If left=[Read(Store)] and right=[Read(Store), Write(Table)]:
    // At Store: Read vs Read. Loop ends. Return false? But left wants to read the whole Store!
    false // For now, we rely on the loop finding an explicit conflict (e.g. `Write(Store)` vs `Read(Store)`).
}
=======
/// Checks if two sets of dependencies conflict.
pub fn conflicts(left: &[Dependency], right: &[Dependency]) -> bool {
    for l in left {
        for r in right {
            if l.conflicts_with(r) {
                return true;
            }
        }
    }
    false
}
>>>>>>> REPLACE
