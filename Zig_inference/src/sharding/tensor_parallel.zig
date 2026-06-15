const std = @import("std");
const tensor = @import("../tensor/tensor.zig");
const matmul = @import("../tensor/matmul.zig");

pub const TensorParallelLayer = struct {
    weight_shards: []tensor.Tensor,
    num_shards: usize,
    execution_arena: *std.heap.ArenaAllocator,

    pub fn init(allocator: std.mem.Allocator, weight: tensor.Tensor, num_shards: usize, arena: *std.heap.ArenaAllocator) !TensorParallelLayer {
        var shards = try allocator.alloc(tensor.Tensor, num_shards);
        for (shards, 0..) |_, i| {
            const config = TensorParallelConfig{ .num_shards = num_shards, .shard_rank = i, .world_size = num_shards };
            shards[i] = try shardWeights(weight, config);
        }
        return TensorParallelLayer{ .weight_shards = shards, .num_shards = num_shards, .execution_arena = arena };
    }

    pub fn deinit(self: *TensorParallelLayer) void {
        for (self.weight_shards) |shard| shard.deinit();
        std.heap.page_allocator.free(self.weight_shards);
    }

    pub fn forward(self: TensorParallelLayer, x: tensor.Tensor) !tensor.Tensor {
        const partials = try self.execution_arena.alloc([]f16, self.num_shards);
        for (self.weight_shards, 0..) |shard, i| {
            const partial = try matmul.forward(x, shard);
            partials[i] = partial.data;
        }
        var c = try tensor.Tensor.init(x.allocator, &[2]usize{ x.shape[0], self.weight_shards[0].shape[1] }, x.dtype, x.device);
        for (c.data, 0..) |_, idx| {
            var sum: f32 = 0;
            var s: usize = 0;
            while (s < self.num_shards) : (s += 1) { sum += @f16tof32(partials[s][idx]); }
            c.data[idx] = @f32tof16(sum);
        }
        return c;
    }
};

pub const TensorParallelConfig = struct { num_shards: usize, shard_rank: usize, world_size: usize; };
pub fn shardWeights(weight: tensor.Tensor, config: TensorParallelConfig) !tensor.Tensor {
    const out_dim = weight.shape[0];
    const in_dim = weight.shape[1];
    const out_dim_local = out_dim / config.num_shards;
    const start_idx = config.shard_rank * out_dim_local;
    const local_size = out_dim_local * in_dim;
    var shard = try tensor.Tensor.init(weight.allocator, &[2]usize{ out_dim_local, in_dim }, weight.dtype, weight.device);
    std.mem.copyForwards(f16, shard.data, weight.data[start_idx * in_dim .. start_idx * in_dim + local_size]);
    return shard;
}
