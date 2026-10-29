struct Params {
    input_cols: u32,
    input_rows: u32,
    channels: u32,
    col0: i32,
    row0: i32,
    output_cols: u32,
    output_rows: u32,
    padding: u32,
    kernel: array<vec4<f32>, 3>,
}

const NAN_BITS: u32 = 0x7fc00000u;

@group(0) @binding(0) var<uniform> params: Params;
// values move as bits, wgsl has no nan test and may fold `v != v` away
@group(0) @binding(1) var<storage, read> input_values: array<u32>;
@group(0) @binding(2) var<storage, read_write> output_values: array<u32>;

fn is_nan(bits: u32) -> bool {
    return (bits & 0x7fffffffu) > 0x7f800000u;
}

fn tap(plane: u32, row: i32, col: i32) -> u32 {
    if (row < 0 || col < 0 || row >= i32(params.input_rows) || col >= i32(params.input_cols)) {
        return NAN_BITS;
    }
    return input_values[plane + u32(row) * params.input_cols + u32(col)];
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.output_cols || id.y >= params.output_rows || id.z >= params.channels) {
        return;
    }
    let plane = id.z * params.input_cols * params.input_rows;
    let row = params.row0 + i32(id.y);
    let col = params.col0 + i32(id.x);
    let output_index = id.z * params.output_cols * params.output_rows
        + id.y * params.output_cols
        + id.x;
    if (is_nan(tap(plane, row, col))) {
        output_values[output_index] = NAN_BITS;
        return;
    }
    var sum = 0.0;
    for (var ky = 0; ky < 3; ky++) {
        for (var kx = 0; kx < 3; kx++) {
            let bits = tap(plane, row + ky - 1, col + kx - 1);
            if (!is_nan(bits)) {
                sum += bitcast<f32>(bits) * params.kernel[ky][kx];
            }
        }
    }
    output_values[output_index] = bitcast<u32>(sum);
}
