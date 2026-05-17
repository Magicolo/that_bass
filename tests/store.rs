use that_bass::store::{Store, query, template::Column};

#[test]
fn store_new_and_default() {
    let _ = Store::new();
    let _: Store = Default::default();
}

#[test]
fn insert_single_column() {
    let mut store = Store::new();
    let mut insert = store.insert(Column::<u32>::default()).unwrap();
    let row = insert.one(42).unwrap();
    assert_eq!(row.table(), 0);
    assert_eq!(row.row(), 0);
}

#[test]
fn insert_multi_column_via_tuple() {
    let mut store = Store::new();
    let template = (Column::<u32>::default(), Column::<f64>::default());
    let mut insert = store.insert(template).unwrap();
    let row = insert.one((42, 1.5_f64)).unwrap();
    assert_eq!(row.table(), 0);
    assert_eq!(row.row(), 0);

    let row2 = insert.one((100, 2.5_f64)).unwrap();
    assert_eq!(row2.table(), 0);
    assert_eq!(row2.row(), 1);
}

#[test]
fn insert_creates_tables_by_schema() {
    let mut store = Store::new();

    let (row1, row2, row3) = {
        let mut t1 = store.insert(Column::<u32>::default()).unwrap();
        let row1 = t1.one(1).unwrap();
        std::mem::drop(t1);

        let mut t2 = store.insert(Column::<f64>::default()).unwrap();
        let row2 = t2.one(2.0).unwrap();
        std::mem::drop(t2);

        let mut t3 = store.insert(Column::<u32>::default()).unwrap();
        let row3 = t3.one(3).unwrap();
        std::mem::drop(t3);

        (row1, row2, row3)
    };

    assert_eq!(row1.table(), 0);
    assert_eq!(row2.table(), 1);
    assert_eq!(row3.table(), 0);
}

#[test]
fn query_empty_store_returns_no_items() {
    let store = Store::new();
    let query = store.query(query::Read::<u32>::new());
    assert!(query.iter().next().is_none());
}

#[test]
fn query_after_insert_without_resolve_returns_no_items() {
    let mut store = Store::new();
    let mut insert = store.insert(Column::<u32>::default()).unwrap();
    let _row = insert.one(42).unwrap();
    std::mem::drop(insert);

    let query = store.query(query::Read::<u32>::new());
    for slice in query.iter() {
        assert!(slice.is_empty());
    }
}

#[test]
fn query_with_row_type() {
    let store = Store::new();
    let query = store.query(query::Row);
    assert!(query.iter().next().is_none());
}

#[test]
fn query_with_table_type() {
    let store = Store::new();
    let query = store.query(query::Table);
    assert!(query.iter().next().is_none());
}

#[test]
fn query_with_column_type() {
    let mut store = Store::new();
    let _insert = store.insert(Column::<u32>::default()).unwrap();

    let meta = that_bass::store::Meta::of::<u32>();
    let query = store.query(query::Column::new(meta));
    for column in query.iter() {
        assert_eq!(column.meta().identifier, std::any::TypeId::of::<u32>());
    }
}

#[test]
fn query_matches_tables_by_schema() {
    let mut store = Store::new();
    let mut insert = store
        .insert((Column::<u32>::default(), Column::<String>::default()))
        .unwrap();
    insert.one((42, "hello".to_string())).unwrap();
    std::mem::drop(insert);

    let query_u32 = store.query(query::Read::<u32>::new());
    assert_eq!(query_u32.iter().count(), 1);

    let query_string = store.query(query::Read::<String>::new());
    assert_eq!(query_string.iter().count(), 1);

    let query_f64 = store.query(query::Read::<f64>::new());
    assert_eq!(query_f64.iter().count(), 0);
}

#[test]
fn row_equality() {
    let mut store = Store::new();
    let mut insert = store.insert(Column::<u32>::default()).unwrap();
    let a = insert.one(1).unwrap();
    let b = insert.one(2).unwrap();
    let c = insert.one(3).unwrap();

    assert_eq!(a, a);
    assert_ne!(a, b);
    assert_ne!(a, c);
    assert_eq!(b, b);
    assert_ne!(b, c);
}

#[test]
fn row_ordering() {
    let mut store = Store::new();
    let mut insert = store.insert(Column::<u32>::default()).unwrap();
    let a = insert.one(1).unwrap();
    let b = insert.one(2).unwrap();
    let c = insert.one(3).unwrap();

    assert!(a < b);
    assert!(b < c);
    assert!(a < c);
}

#[test]
fn multiple_tables_different_schemas() {
    let mut store = Store::new();

    let insert1 = store.insert((Column::<u32>::default(), Column::<String>::default()));
    std::mem::drop(insert1);
    let insert2 = store.insert((Column::<f64>::default(), Column::<bool>::default()));
    std::mem::drop(insert2);
    let insert3 = store.insert(Column::<u32>::default());
    std::mem::drop(insert3);
    let insert4 = store.insert((Column::<u32>::default(), Column::<String>::default()));
    std::mem::drop(insert4);

    let query_all = store.query(query::Table);
    assert_eq!(query_all.iter().count(), 3);
}

#[test]
fn query_iter_uses_commit_count() {
    let mut store = Store::new();
    let mut insert = store.insert(Column::<u32>::default()).unwrap();
    insert.one(1).unwrap();
    insert.one(2).unwrap();
    insert.one(3).unwrap();
    std::mem::drop(insert);

    let mut insert2 = store.insert(Column::<f64>::default()).unwrap();
    insert2.one(1.0).unwrap();
    std::mem::drop(insert2);

    let query = store.query(query::Table);
    assert_eq!(query.iter().count(), 2);

    let query_u32 = store.query(query::Read::<u32>::new());
    for slice in query_u32.iter() {
        assert_eq!(slice.len(), 0);
    }
}

#[test]
fn template_six_tuple() {
    let mut store = Store::new();
    let t = (
        Column::<u8>::default(),
        Column::<u16>::default(),
        Column::<u32>::default(),
        Column::<u64>::default(),
        Column::<bool>::default(),
        Column::<String>::default(),
    );
    let mut insert = store.insert(t).unwrap();
    let row = insert
        .one((1u8, 2u16, 3u32, 4u64, true, "test".to_string()))
        .unwrap();
    assert_eq!(row.row(), 0);
}

#[test]
fn template_seven_tuple() {
    let mut store = Store::new();
    let t = (
        Column::<u8>::default(),
        Column::<u16>::default(),
        Column::<u32>::default(),
        Column::<u64>::default(),
        Column::<bool>::default(),
        Column::<String>::default(),
        Column::<i8>::default(),
    );
    let mut insert = store.insert(t).unwrap();
    let row = insert
        .one((1u8, 2u16, 3u32, 4u64, true, "test".to_string(), -1i8))
        .unwrap();
    assert_eq!(row.row(), 0);
}

#[test]
fn template_eight_tuple() {
    let mut store = Store::new();
    let t = (
        Column::<u8>::default(),
        Column::<u16>::default(),
        Column::<u32>::default(),
        Column::<u64>::default(),
        Column::<bool>::default(),
        Column::<String>::default(),
        Column::<i8>::default(),
        Column::<i16>::default(),
    );
    let mut insert = store.insert(t).unwrap();
    let row = insert
        .one((1u8, 2u16, 3u32, 4u64, true, "test".to_string(), -1i8, -2i16))
        .unwrap();
    assert_eq!(row.row(), 0);
}

#[test]
fn rows_iterator_empty_table() {
    let store = Store::new();
    let query = store.query(query::Row);
    for rows in query.iter() {
        let count = rows.into_iter().count();
        assert_eq!(count, 0);
    }
}

#[test]
fn column_iterator_empty_table() {
    let mut store = Store::new();
    let _insert = store.insert(Column::<u32>::default()).unwrap();
    let query = store.query(query::Column::new(that_bass::store::Meta::of::<u32>()));
    for column in query.iter() {
        assert_eq!(column.meta().identifier, std::any::TypeId::of::<u32>());
    }
}

#[test]
fn table_reference_via_query() {
    let mut store = Store::new();
    let _insert = store.insert(Column::<u32>::default()).unwrap();
    let query = store.query(query::Table);
    for table in query.iter() {
        assert_eq!(table.count(), 0);
    }
}
