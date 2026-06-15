pub const tensor_parallel = @import("tensor_parallel.zig");
pub const pipeline_parallel = @import("pipeline_parallel.zig");
pub const rpc = @import("rpc.zig");

pub const TensorParallelConfig = struct {
    num_shards: usize,
    shard_rank: usize,
    world_size: usize,
};

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
pub const tensor = @import("../tensor/tensor.zig");
