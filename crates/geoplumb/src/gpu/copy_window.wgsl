struct Params {
    source_cols: u32,
    source_rows: u32,
    destination_cols: u32,
    destination_rows: u32,
    channels: u32,
    source_col0: u32,
    source_row0: u32,
    destination_col0: u32,
    destination_row0: u32,
    cols: u32,
    rows: u32,
    padding: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// u32 rather than f32 so nan payloads copy bit for bit
@group(0) @binding(1) var<storage, read> source_values: array<u32>;
@group(0) @binding(2) var<storage, read_write> destination_values: array<u32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.cols || id.y >= params.rows || id.z >= params.channels) {
        return;
    }
    let source_index = id.z * params.source_cols * params.source_rows
        + (params.source_row0 + id.y) * params.source_cols
        + params.source_col0 + id.x;
    let destination_index = id.z * params.destination_cols * params.destination_rows
        + (params.destination_row0 + id.y) * params.destination_cols
        + params.destination_col0 + id.x;
    destination_values[destination_index] = source_values[source_index];
}
