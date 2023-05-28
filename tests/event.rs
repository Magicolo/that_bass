pub mod common;
use common::*;

#[test]
fn broadcast_on_add() -> Result<(), Error> {
    #[derive(Default, Debug)]
    struct A;
    impl Datum for A {}

    let database = Database::new();
    let mut create = database.create::<()>()?;
    let mut destroy = database.destroy_all();
    let mut on_add1 = database.events().on_add().with_key().with_type::<A>();
    let mut on_add2 = database.events().on_add().with_key().with_type::<A>();
    let mut on_add3 = database.events().on_add().with_key().with_type::<A>();
    let mut keys2 = Vec::new();
    let mut keys3 = Vec::new();

    for i in 0..COUNT {
        assert!(on_add1.next().is_none());
        let keys = create.defaults(i).to_vec();
        keys2.extend(keys.iter().copied());
        keys3.extend(keys.iter().copied());
        let on_add4 = database.events().on_add().with_key().with_type::<A>();
        assert_eq!(create.resolve(), i);
        assert!(on_add1.next().is_none());
        let mut add = database.add::<A>()?;
        add.all(keys.iter().copied());
        assert_eq!(add.resolve(), i);
        assert!((&mut on_add1).map(|e| e.key).eq(keys.iter().copied()));
        assert!(on_add4.map(|e| e.key).eq(keys.iter().copied()));
        assert!(database
            .events()
            .on_add()
            .with_key()
            .with_type::<A>()
            .next()
            .is_none());

        if i % 13 == 0 {
            on_add3.clear();
            keys3.clear();
        }
        if i % 11 == 0 {
            assert!((&mut on_add3).map(|e| e.key).eq(keys3.drain(..)));
        }
        if i % 7 == 0 {
            on_add2.clear();
            keys2.clear();
        }
        if i % 3 == 0 {
            assert!((&mut on_add2).map(|e| e.key).eq(keys2.drain(..)));
        }

        assert_eq!(destroy.resolve(), i);
    }
    Ok(())
}
