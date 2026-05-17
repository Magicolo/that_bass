use that_bass::store::{Store, query, template};

#[test]
fn store_new_and_default() {
    let _ = Store::new();
    let _: Store = Default::default();
}

#[test]
fn insert_single_column() -> anyhow::Result<()> {
    let mut store = Store::new();
    let mut insert = store.insert(template::column::<u32>())?;
    let row = insert.one(42)?;
    assert_eq!(row.table(), 0);
    assert_eq!(row.row(), 0);
    Ok(())
}

#[test]
fn insert_multi_column_via_tuple() -> anyhow::Result<()> {
    let mut store = Store::new();
    let template = (template::column::<u32>(), template::column::<f64>());
    let mut insert = store.insert(template)?;
    let row = insert.one((42, 1.5_f64))?;
    assert_eq!(row.table(), 0);
    assert_eq!(row.row(), 0);

    let row2 = insert.one((100, 2.5_f64))?;
    assert_eq!(row2.table(), 0);
    assert_eq!(row2.row(), 1);
    Ok(())
}

#[test]
fn insert_creates_tables_by_schema() -> anyhow::Result<()> {
    let mut store = Store::new();

    let (r1_t, r1_r, r2_t, r2_r, r3_t, r3_r) = {
        let mut t = store.insert(template::column::<u32>())?;
        let row = t.one(1)?;
        (row.table(), row.row(), 0, 0, 0, 0)
    };

    let (_, _, r2_t, r2_r, r3_t, r3_r) = {
        let mut t = store.insert(template::column::<f64>())?;
        let row = t.one(2.0)?;
        (0, 0, row.table(), row.row(), 0, 0)
    };

    let (_, _, _, _, r3_t, r3_r) = {
        let mut t = store.insert(template::column::<u32>())?;
        let row = t.one(3)?;
        (0, 0, 0, 0, row.table(), row.row())
    };

    assert_eq!(r1_t, 0);
    assert_eq!(r2_t, 1);
    assert_eq!(r3_t, 0);
    assert_eq!(r1_r, 0);
    assert_eq!(r2_r, 0);
    assert_eq!(r3_r, 1);
    Ok(())
}

#[test]
fn query_empty_store_returns_no_items() {
    let store = Store::new();
    let query = store.query(query::read::<u32>());
    assert!(query.iter().next().is_none());
}

#[test]
fn query_after_insert_without_resolve_returns_no_items() -> anyhow::Result<()> {
    let mut store = Store::new();
    let mut insert = store.insert(template::column::<u32>())?;
    let _row = insert.one(42)?;
    std::mem::drop(insert);

    let query = store.query(query::read::<u32>());
    for slice in query.iter() {
        assert!(slice.is_empty());
    }
    Ok(())
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
fn query_with_column_type() -> anyhow::Result<()> {
    let mut store = Store::new();
    let _insert = store.insert(template::column::<u32>())?;

    let meta = that_bass::store::Meta::of::<u32>();
    let query = store.query(query::read_with(meta));
    for column in query.iter() {
        assert_eq!(column.meta().identifier, std::any::TypeId::of::<u32>());
    }
    Ok(())
}

#[test]
fn query_matches_tables_by_schema() -> anyhow::Result<()> {
    let mut store = Store::new();
    let mut insert = store.insert((template::column::<u32>(), template::column::<String>()))?;
    insert.one((42, "hello".to_string()))?;
    std::mem::drop(insert);

    let query_u32 = store.query(query::read::<u32>());
    assert_eq!(query_u32.iter().count(), 1);

    let query_string = store.query(query::read::<String>());
    assert_eq!(query_string.iter().count(), 1);

    let query_f64 = store.query(query::read::<f64>());
    assert_eq!(query_f64.iter().count(), 0);
    Ok(())
}

#[test]
fn row_equality() -> anyhow::Result<()> {
    let mut store = Store::new();
    let mut insert = store.insert(template::column::<u32>())?;
    let a = insert.one(1)?;
    let b = insert.one(2)?;
    let c = insert.one(3)?;

    assert_eq!(a, a);
    assert_ne!(a, b);
    assert_ne!(a, c);
    assert_eq!(b, b);
    assert_ne!(b, c);
    Ok(())
}

#[test]
fn row_ordering() -> anyhow::Result<()> {
    let mut store = Store::new();
    let mut insert = store.insert(template::column::<u32>())?;
    let a = insert.one(1)?;
    let b = insert.one(2)?;
    let c = insert.one(3)?;

    assert!(a < b);
    assert!(b < c);
    assert!(a < c);
    Ok(())
}

#[test]
fn multiple_tables_different_schemas() -> anyhow::Result<()> {
    let mut store = Store::new();

    let insert1 = store.insert((template::column::<u32>(), template::column::<String>()))?;
    std::mem::drop(insert1);
    let insert2 = store.insert((template::column::<f64>(), template::column::<bool>()))?;
    std::mem::drop(insert2);
    let insert3 = store.insert(template::column::<u32>())?;
    std::mem::drop(insert3);
    let insert4 = store.insert((template::column::<u32>(), template::column::<String>()))?;
    std::mem::drop(insert4);

    let query_all = store.query(query::Table);
    assert_eq!(query_all.iter().count(), 3);
    Ok(())
}

#[test]
fn query_iter_uses_commit_count() -> anyhow::Result<()> {
    let mut store = Store::new();
    let mut insert = store.insert(template::column::<u32>())?;
    insert.one(1)?;
    insert.one(2)?;
    insert.one(3)?;
    std::mem::drop(insert);

    let mut insert2 = store.insert(template::column::<f64>())?;
    insert2.one(1.0)?;
    std::mem::drop(insert2);

    let query = store.query(query::Table);
    assert_eq!(query.iter().count(), 2);

    let query_u32 = store.query(query::read::<u32>());
    for slice in query_u32.iter() {
        assert_eq!(slice.len(), 0);
    }
    Ok(())
}

#[test]
fn template_six_tuple() -> anyhow::Result<()> {
    let mut store = Store::new();
    let t = (
        template::column::<u8>(),
        template::column::<u16>(),
        template::column::<u32>(),
        template::column::<u64>(),
        template::column::<bool>(),
        template::column::<String>(),
    );
    let mut insert = store.insert(t)?;
    let row = insert.one((1u8, 2u16, 3u32, 4u64, true, "test".to_string()))?;
    assert_eq!(row.row(), 0);
    Ok(())
}

#[test]
fn template_seven_tuple() -> anyhow::Result<()> {
    let mut store = Store::new();
    let t = (
        template::column::<u8>(),
        template::column::<u16>(),
        template::column::<u32>(),
        template::column::<u64>(),
        template::column::<bool>(),
        template::column::<String>(),
        template::column::<i8>(),
    );
    let mut insert = store.insert(t)?;
    let row = insert.one((1u8, 2u16, 3u32, 4u64, true, "test".to_string(), -1i8))?;
    assert_eq!(row.row(), 0);
    Ok(())
}

#[test]
fn template_eight_tuple() -> anyhow::Result<()> {
    let mut store = Store::new();
    let t = (
        template::column::<u8>(),
        template::column::<u16>(),
        template::column::<u32>(),
        template::column::<u64>(),
        template::column::<bool>(),
        template::column::<String>(),
        template::column::<i8>(),
        template::column::<i16>(),
    );
    let mut insert = store.insert(t)?;
    let row = insert.one((1u8, 2u16, 3u32, 4u64, true, "test".to_string(), -1i8, -2i16))?;
    assert_eq!(row.row(), 0);
    Ok(())
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
fn column_iterator_empty_table() -> anyhow::Result<()> {
    let mut store = Store::new();
    let _insert = store.insert(template::column::<u32>())?;
    let query = store.query(query::read_with(that_bass::store::Meta::of::<u32>()));
    for column in query.iter() {
        assert_eq!(column.meta().identifier, std::any::TypeId::of::<u32>());
    }
    Ok(())
}

#[test]
fn table_reference_via_query() -> anyhow::Result<()> {
    let mut store = Store::new();
    let _insert = store.insert(template::column::<u32>())?;
    let query = store.query(query::Table);
    for table in query.iter() {
        assert_eq!(table.count(), 0);
    }
    Ok(())
}
