<<<<<<< SEARCH
use that_bass::v2::{
    query,
    schedule::{paths_conflict, Dependency, Resource},
    schema::{ChunkIndex, TableIndex},
    store::Store,
};

#[test]
fn schedule_no_conflict_between_pure_reads() {
    let _store = Store::new();
    let left = vec![
        Dependency {
            resource: Resource::store(),
            access: query::Access::Read,
        },
        Dependency {
            resource: Resource::table(TableIndex::new(0)),
            access: query::Access::Read,
        },
    ];
    let right = vec![
        Dependency {
            resource: Resource::store(),
            access: query::Access::Read,
        },
        Dependency {
            resource: Resource::table(TableIndex::new(0)),
            access: query::Access::Read,
        },
    ];
    assert!(!paths_conflict(&left, &right));
}

#[test]
fn schedule_broad_store_writer_conflicts_correctly() {
    let left = vec![Dependency {
        resource: Resource::store(),
        access: query::Access::Write,
    }];
    let right = vec![
        Dependency {
            resource: Resource::store(),
            access: query::Access::Read,
        },
        Dependency {
            resource: Resource::table(TableIndex::new(0)),
            access: query::Access::Read,
        },
    ];
    assert!(paths_conflict(&left, &right));
}

#[test]
fn schedule_chunk_scoped_ordering_between_same_writer_families() {
    let left = vec![
        Dependency {
            resource: Resource::store(),
            access: query::Access::Read,
        },
        Dependency {
            resource: Resource::table(TableIndex::new(0)),
            access: query::Access::Read,
        },
        Dependency {
            resource: Resource::chunk(TableIndex::new(0), ChunkIndex::new(5)),
            access: query::Access::Write,
        },
    ];
    let right = vec![
        Dependency {
            resource: Resource::store(),
            access: query::Access::Read,
        },
        Dependency {
            resource: Resource::table(TableIndex::new(0)),
            access: query::Access::Read,
        },
        Dependency {
            resource: Resource::chunk(TableIndex::new(0), ChunkIndex::new(5)),
            access: query::Access::Write,
        },
    ];
    let right_different_chunk = vec![
        Dependency {
            resource: Resource::store(),
            access: query::Access::Read,
        },
        Dependency {
            resource: Resource::table(TableIndex::new(0)),
            access: query::Access::Read,
        },
        Dependency {
            resource: Resource::chunk(TableIndex::new(0), ChunkIndex::new(6)),
            access: query::Access::Write,
        },
    ];
    assert!(paths_conflict(&left, &right));
    assert!(!paths_conflict(&left, &right_different_chunk));
}
=======
use that_bass::v2::{
    query,
    schedule::{conflicts, Dependency, Resource},
    schema::{ChunkIndex, TableIndex},
};

#[test]
fn schedule_no_conflict_between_pure_reads() {
    let left = vec![
        Dependency { resource: Resource::Store, access: query::Access::Read },
        Dependency { resource: Resource::Table(TableIndex::new(0)), access: query::Access::Read },
    ];
    let right = vec![
        Dependency { resource: Resource::Store, access: query::Access::Read },
        Dependency { resource: Resource::Table(TableIndex::new(0)), access: query::Access::Read },
    ];
    assert!(!conflicts(&left, &right));
}

#[test]
fn schedule_broad_store_writer_conflicts_correctly() {
    let left = vec![
        Dependency { resource: Resource::Store, access: query::Access::Write },
    ];
    let right = vec![
        Dependency { resource: Resource::Store, access: query::Access::Read },
        Dependency { resource: Resource::Table(TableIndex::new(0)), access: query::Access::Read },
    ];
    assert!(conflicts(&left, &right));
}

#[test]
fn schedule_chunk_scoped_ordering_between_same_writer_families() {
    let left = vec![
        Dependency { resource: Resource::Store, access: query::Access::Read },
        Dependency { resource: Resource::Table(TableIndex::new(0)), access: query::Access::Read },
        Dependency { resource: Resource::Chunk(TableIndex::new(0), ChunkIndex::new(5)), access: query::Access::Write },
    ];
    let right = vec![
        Dependency { resource: Resource::Store, access: query::Access::Read },
        Dependency { resource: Resource::Table(TableIndex::new(0)), access: query::Access::Read },
        Dependency { resource: Resource::Chunk(TableIndex::new(0), ChunkIndex::new(5)), access: query::Access::Write },
    ];
    let right_different_chunk = vec![
        Dependency { resource: Resource::Store, access: query::Access::Read },
        Dependency { resource: Resource::Table(TableIndex::new(0)), access: query::Access::Read },
        Dependency { resource: Resource::Chunk(TableIndex::new(0), ChunkIndex::new(6)), access: query::Access::Write },
    ];
    assert!(conflicts(&left, &right));
    assert!(!conflicts(&left, &right_different_chunk));
}
>>>>>>> REPLACE
