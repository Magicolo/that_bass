pub mod bits;
pub mod iterate;
pub mod utility;

macro_rules! tuples {
    ($m:ident) => {
        $m!(Tuples0, 0);
        $m!(Tuples1, 1, p0, T0, 0);
        $m!(Tuples2, 2, p0, T0, 0, p1, T1, 1);
        $m!(Tuples3, 3, p0, T0, 0, p1, T1, 1, p2, T2, 2);
        $m!(Tuples4, 4, p0, T0, 0, p1, T1, 1, p2, T2, 2, p3, T3, 3);
        $m!(Tuples5, 5, p0, T0, 0, p1, T1, 1, p2, T2, 2, p3, T3, 3, p4, T4, 4);
        $m!(Tuples6, 6, p0, T0, 0, p1, T1, 1, p2, T2, 2, p3, T3, 3, p4, T4, 4, p5, T5, 5);
        $m!(
            Tuples7, 7, p0, T0, 0, p1, T1, 1, p2, T2, 2, p3, T3, 3, p4, T4, 4, p5, T5, 5, p6, T6, 6
        );
        $m!(
            Tuples8, 8, p0, T0, 0, p1, T1, 1, p2, T2, 2, p3, T3, 3, p4, T4, 4, p5, T5, 5, p6, T6,
            6, p7, T7, 7
        );
        $m!(
            Tuples9, 9, p0, T0, 0, p1, T1, 1, p2, T2, 2, p3, T3, 3, p4, T4, 4, p5, T5, 5, p6, T6,
            6, p7, T7, 7, p8, T8, 8
        );
        $m!(
            Tuples10, 10, p0, T0, 0, p1, T1, 1, p2, T2, 2, p3, T3, 3, p4, T4, 4, p5, T5, 5, p6, T6,
            6, p7, T7, 7, p8, T8, 8, p9, T9, 9
        );
        $m!(
            Tuples11, 11, p0, T0, 0, p1, T1, 1, p2, T2, 2, p3, T3, 3, p4, T4, 4, p5, T5, 5, p6, T6,
            6, p7, T7, 7, p8, T8, 8, p9, T9, 9, p10, T10, 10
        );
        $m!(
            Tuples12, 12, p0, T0, 0, p1, T1, 1, p2, T2, 2, p3, T3, 3, p4, T4, 4, p5, T5, 5, p6, T6,
            6, p7, T7, 7, p8, T8, 8, p9, T9, 9, p10, T10, 10, p11, T11, 11
        );
        $m!(
            Tuples13, 13, p0, T0, 0, p1, T1, 1, p2, T2, 2, p3, T3, 3, p4, T4, 4, p5, T5, 5, p6, T6,
            6, p7, T7, 7, p8, T8, 8, p9, T9, 9, p10, T10, 10, p11, T11, 11, p12, T12, 12
        );
        $m!(
            Tuples14, 14, p0, T0, 0, p1, T1, 1, p2, T2, 2, p3, T3, 3, p4, T4, 4, p5, T5, 5, p6, T6,
            6, p7, T7, 7, p8, T8, 8, p9, T9, 9, p10, T10, 10, p11, T11, 11, p12, T12, 12, p13, T13,
            13
        );
        $m!(
            Tuples15, 15, p0, T0, 0, p1, T1, 1, p2, T2, 2, p3, T3, 3, p4, T4, 4, p5, T5, 5, p6, T6,
            6, p7, T7, 7, p8, T8, 8, p9, T9, 9, p10, T10, 10, p11, T11, 11, p12, T12, 12, p13, T13,
            13, p14, T14, 14
        );
        $m!(
            Tuples16, 16, p0, T0, 0, p1, T1, 1, p2, T2, 2, p3, T3, 3, p4, T4, 4, p5, T5, 5, p6, T6,
            6, p7, T7, 7, p8, T8, 8, p9, T9, 9, p10, T10, 10, p11, T11, 11, p12, T12, 12, p13, T13,
            13, p14, T14, 14, p15, T15, 15
        );
    };
}
pub(crate) use tuples;
