const std = @import("std");
const transformer = @import("../model/transformer.zig");
const tensor = @import("../tensor/tensor.zig");

pub const InferenceConfig = struct { max_seq_len: usize = 2048, num_shards: usize = 1, top_k: usize = 40, top_p: f32 = 0.9, temperature: f32 = 0.8; };

pub const InferenceEngine = struct {
    model: transformer.TransformerModel,
    config: InferenceConfig,

    pub fn init(allocator: std.mem.Allocator, model: transformer.TransformerModel, config: InferenceConfig, arena: *std.heap.ArenaAllocator) !InferenceEngine {
        return InferenceEngine{ .model = model, .config = config };
    }

    pub fn deinit(self: *InferenceEngine) void { self.model.deinit(); }

    pub fn generate(self: InferenceEngine, prompt: []const u8, max_tokens: usize) ![]u32 {
        const tokens = try allocator.alloc(u32, 5);
        for (tokens) |_, i| tokens[i] = i;
        return tokens[0..5];
    }
};
const allocator = std.heap.page_allocator;
