const std = @import("std");
const tensor = @import("../tensor/tensor.zig");

pub const TransformerConfig = struct {
    vocab_size: usize,
    hidden_dim: usize,
    num_layers: usize,
    num_heads: usize,
    head_dim: usize,
    intermediate_dim: usize,
    max_seq_len: usize,
    num_shards: usize,
};

pub const TransformerModel = struct {
    config: TransformerConfig,
    token_embedding: tensor.Tensor,
    layers: []tensor.Tensor,
    output_weight: tensor.Tensor,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, config: TransformerConfig, weights: []tensor.Tensor, arena: *std.heap.ArenaAllocator) !TransformerModel {
        var layers = try allocator.alloc(tensor.Tensor, config.num_layers);
        for (layers, 0..) |_, i| { layers[i] = weights[i]; }
        return TransformerModel{
            .config = config,
            .token_embedding = weights[config.num_layers],
            .layers = layers,
            .output_weight = weights[config.num_layers + 1],
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *TransformerModel) void {
        self.allocator.free(self.layers);
        self.token_embedding.deinit();
        self.output_weight.deinit();
    }

    pub fn forward(self: TransformerModel, input_ids: []u32) !tensor.Tensor {
        return try self.token_embedding;
    }
};
