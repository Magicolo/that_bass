use that_bass::v2::{
    command::{Kind, Remove, Strategy},
    key::Key,
    query::{self, Access},
    schedule::Ordering,
};

pub fn run() {
    let position_access = Access::Write;
    let velocity_access = Access::Read;
    let rows_request = query::rows();
    let remove_command_kind = Kind::Remove;
    let resolve_strategy = Strategy::FunctionLevelBatch;
    let default_ordering = Ordering::ImplicitDeclarationOrder;
    let remove_buffer = Remove::new();
    let key = Key::new(12, 3);

    println!("Vocabulary snapshot");
    println!("  query access: position={position_access:?}, velocity={velocity_access:?}");
    println!("  query rows request: {rows_request:?}");
    println!("  command kind: {remove_command_kind:?}");
    println!("  empty remove buffer length: {}", remove_buffer.len());
    println!("  resolve strategy: {resolve_strategy:?}");
    println!("  default ordering: {default_ordering:?}");
    println!(
        "  key datum example: slot_index={}, generation={}",
        key.slot_index(),
        key.generation()
    );
}
