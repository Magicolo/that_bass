use that_bass::v2::{
    command::{Kind, Strategy},
    key::Key,
    query::Access,
    schedule::Ordering,
};

pub fn run() {
    let position_access = Access::Write;
    let velocity_access = Access::Read;
    let remove_command_kind = Kind::Remove;
    let resolve_strategy = Strategy::FunctionLevelBatch;
    let default_ordering = Ordering::ImplicitDeclarationOrder;
    let managed_key = Key::new(12, 3);

    println!("Vocabulary snapshot");
    println!("  query access: position={position_access:?}, velocity={velocity_access:?}");
    println!("  command kind: {remove_command_kind:?}");
    println!("  resolve strategy: {resolve_strategy:?}");
    println!("  default ordering: {default_ordering:?}");
    println!(
        "  managed key example: slot_index={}, generation={}",
        managed_key.slot_index(),
        managed_key.generation()
    );
}
