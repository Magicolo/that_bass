use criterion::Criterion;
use that_bass::v1::{Database, Datum};

#[derive(Datum, Clone, Copy, Default, Debug)]
#[allow(dead_code)]
struct A(u128);

pub fn benchmark(criterion: &mut Criterion) {
    const COUNT: usize = 1000;
    let mut group = criterion.benchmark_group("create");
    group.bench_function("all_n", |bencher| {
        let database = Database::new();
        let mut create = database.create::<A>().unwrap();
        bencher.iter(|| {
            create.all_n([A(1); COUNT]);
            create.resolve();
        })
    });

    group.bench_function("all", |bencher| {
        let database = Database::new();
        let mut create = database.create::<A>().unwrap();
        bencher.iter(|| {
            create.all([A(1); COUNT]);
            create.resolve();
        })
    });

    group.bench_function("one", |bencher| {
        let database = Database::new();
        let mut create = database.create::<A>().unwrap();
        bencher.iter(|| {
            for _ in 0..COUNT {
                create.one(A(1));
            }
            create.resolve();
        })
    });

    group.bench_function("with", |bencher| {
        let database = Database::new();
        let mut create = database.create::<A>().unwrap();
        bencher.iter(|| {
            create.with(COUNT, || A(1));
            create.resolve();
        })
    });

    group.bench_function("with_n", |bencher| {
        let database = Database::new();
        let mut create = database.create::<A>().unwrap();
        bencher.iter(|| {
            create.with_n::<COUNT>(|| A(1));
            create.resolve();
        })
    });

    group.bench_function("clones", |bencher| {
        let database = Database::new();
        let mut create = database.create::<A>().unwrap();
        bencher.iter(|| {
            create.clones(COUNT, &A(1));
            create.resolve();
        })
    });

    group.bench_function("clones_n", |bencher| {
        let database = Database::new();
        let mut create = database.create::<A>().unwrap();
        bencher.iter(|| {
            create.clones_n::<COUNT>(&A(1));
            create.resolve();
        })
    });

    group.bench_function("defaults", |bencher| {
        let database = Database::new();
        let mut create = database.create::<A>().unwrap();
        bencher.iter(|| {
            create.defaults(COUNT);
            create.resolve();
        })
    });

    group.bench_function("defaults_n", |bencher| {
        let database = Database::new();
        let mut create = database.create::<A>().unwrap();
        bencher.iter(|| {
            create.defaults_n::<COUNT>();
            create.resolve();
        })
    });

    group.finish();
}
