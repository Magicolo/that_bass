pub mod common;
use common::*;

pub mod filter {
    use super::*;

    #[derive(Filter, Default)]
    pub struct UnitStruct;
    #[derive(Filter, Default)]
    pub struct EmptyTupleStruct();
    #[derive(Filter, Default)]
    pub struct EmptyMapStruct {}
    #[derive(Filter, Default)]
    pub struct GenericTupleStruct<T: Filter, U>(T, PhantomData<U>);
    #[derive(Filter, Default)]
    pub struct TupleStruct(
        UnitStruct,
        EmptyTupleStruct,
        EmptyMapStruct,
        Has<A>,
        (Is<B>, Not<Has<C>>),
        (),
        GenericTupleStruct<UnitStruct, [bool; 32]>,
    );
    #[derive(Filter, Default)]
    pub struct MapStruct {
        a: UnitStruct,
        b: EmptyTupleStruct,
        c: TupleStruct,
        d: EmptyMapStruct,
        e: Has<A>,
        f: (Is<B>, Not<Has<C>>),
        h: (),
        i: GenericTupleStruct<(), usize>,
    }
    #[derive(Filter)]
    pub enum EmptyEnum {}
    #[derive(Filter, Default)]
    pub enum GenericEnum<T: Filter, U> {
        A(T),
        B(PhantomData<U>),
        #[default]
        C,
    }
    #[derive(Filter, Default)]
    pub enum Enum {
        #[default]
        A,
        B(),
        C(Has<A>),
        D(
            Is<B>,
            Not<Has<C>>,
            TupleStruct,
            MapStruct,
            GenericEnum<MapStruct, (char,)>,
        ),
        E {},
        F {
            a: UnitStruct,
            b: EmptyTupleStruct,
            c: TupleStruct,
            d: MapStruct,
            e: GenericTupleStruct<EmptyTupleStruct, i32>,
        },
    }

    #[test]
    fn compiles() -> Result<(), Error> {
        let database = Database::new();
        database.destroy().filter::<UnitStruct>();
        database.destroy().filter::<EmptyTupleStruct>();
        database.destroy().filter::<Any<(EmptyMapStruct,)>>();
        database.destroy().filter::<(Enum, Enum)>();
        database.destroy().filter::<Any<(TupleStruct, MapStruct)>>();
        database
            .destroy()
            .filter::<GenericTupleStruct<UnitStruct, bool>>();
        database.destroy().filter::<GenericEnum<Any<()>, bool>>();
        Ok(())
    }
}

pub mod template {
    use super::*;

    #[derive(Template, Default)]
    pub struct UnitStruct;
    #[derive(Template, Default)]
    pub struct EmptyTupleStruct();
    #[derive(Template, Default)]
    pub struct EmptyMapStruct {}
    #[derive(Template, Default)]
    pub struct GenericTupleStruct<T: Template, U: 'static>(T, PhantomData<U>);
    #[derive(Template, Default)]
    pub struct TupleStruct(
        UnitStruct,
        EmptyTupleStruct,
        EmptyMapStruct,
        A,
        (B, C),
        (),
        GenericTupleStruct<UnitStruct, [bool; 32]>,
    );
    #[derive(Template, Default)]
    pub struct MapStruct {
        a: UnitStruct,
        b: EmptyTupleStruct,
        c: EmptyMapStruct,
        d: A,
        e: (B, C),
        f: (),
        g: GenericTupleStruct<(), usize>,
    }

    #[test]
    fn compiles() -> Result<(), Error> {
        let database = Database::new();
        database.create::<UnitStruct>()?.defaults(1);
        database.create::<EmptyTupleStruct>()?.defaults(1);
        database.create::<EmptyMapStruct>()?.defaults(1);
        database
            .create::<GenericTupleStruct<A, bool>>()?
            .defaults(1);
        database.create::<TupleStruct>()?.defaults(1);
        database.create::<MapStruct>()?.defaults(1);
        Ok(())
    }
}

// pub mod row {
//     use super::*;

//     struct Boba<'a>(&'a ());
//     trait Fett {}
//     impl<'a> Fett for Boba<'static> {}

//     #[derive(Row)]
//     pub struct UnitStruct;
//     #[derive(Row)]
//     pub struct EmptyTupleStruct();
//     #[derive(Row)]
//     pub struct EmptyMapStruct {}
//     #[derive(Row)]
//     pub struct GenericTupleStruct<T: Row, U: 'static>(T, PhantomData<U>);
//     #[derive(Row)]
//     pub struct TupleStruct<'a>(
//         UnitStruct,
//         EmptyTupleStruct,
//         EmptyMapStruct,
//         &'a A,
//         (&'a mut B, PhantomData<&'a C>),
//         (),
//         GenericTupleStruct<UnitStruct, [bool; 32]>,
//     );
//     #[derive(Row)]
//     pub struct MapStruct<'a> {
//         _a: UnitStruct,
//         _b: EmptyTupleStruct,
//         _c: TupleStruct<'a>,
//         _d: EmptyMapStruct,
//         _e: &'a A,
//         _f: (&'a B, &'a mut C),
//         _h: (),
//         _i: GenericTupleStruct<(), usize>,
//     }

//     #[test]
//     fn compiles() -> Result<(), Error> {
//         let database = Database::new();
//         database.query::<UnitStruct>()?.each(|_item| {});
//         database.query::<EmptyTupleStruct>()?.each(|_item| {});
//         database.query::<EmptyMapStruct>()?.each(|_item| {});
//         database
//             .query::<GenericTupleStruct<&A, bool>>()?
//             .each(|_item| {});
//         database.query::<TupleStruct>()?.each(|_item| {});
//         database.query::<MapStruct>()?.each(|_item| {});
//         Ok(())
//     }
// }
