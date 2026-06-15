const std = @import("std");
const tensor = @import("tensor.zig");

const VEC_SIZE = 32;

pub fn matmulVectorized(a_row: []const f16, b_col: []const f16) f16 {
    const vec_size = VEC_SIZE;
    var sum_vec: @Vector(vec_size, f32) = @splat(0.0);
    var i: usize = 0;
    while (i < a_row.len) : (i += vec_size) {
        const remaining = a_row.len - i;
        if (remaining >= vec_size) {
            const va: @Vector(vec_size, f16) = @bitCast(a_row[i .. i + vec_size].*);
            const vb: @Vector(vec_size, f16) = @bitCast(b_col[i .. i + vec_size].*);
            const va_f32: @Vector(vec_size, f32) = @floatCast(va);
            const vb_f32: @Vector(vec_size, f32) = @floatCast(vb);
            sum_vec += va_f32 * vb_f32;
        } else {
            var sum_scalar: f32 = 0;
            for (a_row[i..], b_col[i..]) |a_val, b_val| {
                sum_scalar += @f16tof32(a_val) * @f16tof32(b_val);
            }
            sum_vec += @splat(sum_scalar);
        }
    }
    return @f32tof16(@reduce(.Add, sum_vec));
}

pub fn forward(a: tensor.Tensor, b: tensor.Tensor) !tensor.Tensor {
    const m = a.shape[0];
    const n = b.shape[1];
    const k = a.shape[1];
    if (k != b.shape[0]) return TensorError.MatrixDimensionMismatch;

    var c = try tensor.Tensor.init(a.allocator, &[2]usize{ m, n }, tensor.Dtype.f16, a.device);
    const BLOCK_SIZE = 64;
    var i: usize = 0;
    while (i < m) : (i += BLOCK_SIZE) {
        var j: usize = 0;
        while (j < n) : (j += BLOCK_SIZE) {
            var p: usize = 0;
            while (p < k) : (p += BLOCK_SIZE) {
                var ii: usize = i;
                while (ii < m and ii < i + BLOCK_SIZE) : (ii += 1) {
                    var jj: usize = j;
                    while (jj < n and jj < j + BLOCK_SIZE) : (jj += 1) {
                        var sum: f32 = 0;
                        var pp: usize = p;
                        while (pp < k and pp < p + BLOCK_SIZE) : (pp += 1) {
                            sum += @f16tof32(a.data[ii * k + pp]) * @f16tof32(b.data[pp * n + jj]);
                        }
                        c.data[ii * n + jj] = @f32tof16(sum);
                    }
                }
            }
        }
    }
    return c;
}

pub fn forwardSharded(a: tensor.Tensor, b_shards: []tensor.Tensor, num_shards: usize, arena: *std.heap.ArenaAllocator) !tensor.Tensor {
    const m = a.shape[0];
    const n = b_shards[0].shape[1];
    const partials = try arena.alloc([]f16, num_shards);
    var s: usize = 0;
    while (s < num_shards) : (s += 1) {
        const partial = try forward(a, b_shards[s]);
        partials[s] = partial.data;
    }
    var c = try tensor.Tensor.init(a.allocator, &[2]usize{ m, n }, tensor.Dtype.f16, a.device);
    for (c.data, 0..) |_, idx| {
        var sum: f32 = 0;
        var shard_idx: usize = 0;
        while (shard_idx < num_shards) : (shard_idx += 1) {
            sum += @f16tof32(partials[shard_idx][idx]);
        }
        c.data[idx] = @f32tof16(sum);
    }
    return c;
}
