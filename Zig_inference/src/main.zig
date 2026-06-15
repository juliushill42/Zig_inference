const std = @import("std");
const inference = @import("inference/engine.zig");
const transformer = @import("model/transformer.zig");
const gguf = @import("model/gguf.zig");
const sharding = @import("sharding.zig");
const tensor = @import("tensor/tensor.zig");
const matmul = @import("tensor/matmul.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPoolAllocator(std.heap.page_allocator);
    const allocator = gpa.allocator();
    var execution_arena = std.heap.ArenaAllocator.init(allocator);
    defer execution_arena.deinit();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    var model_path: []const u8 = "/models/llama-3.2-3b.gguf";
    var num_shards: usize = 1;
    var max_tokens: usize = 100;
    var port: u16 = 8080;

    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--model")) { model_path = args[i + 1]; i += 1; }
        else if (std.mem.eql(u8, args[i], "--shards")) { num_shards = std.fmt.parseInt(usize, args[i + 1], 10) catch 1; i += 1; }
        else if (std.mem.eql(u8, args[i], "--tokens")) { max_tokens = std.fmt.parseInt(usize, args[i + 1], 10) catch 100; i += 1; }
        else if (std.mem.eql(u8, args[i], "--port")) { port = std.fmt.parseInt(u16, args[i + 1], 10) catch 8080; i += 1; }
    }

    const metadata = try gguf.parseGGUF(allocator, model_path);
    defer { for (metadata.tensors) |t| t.allocator.free(t.shape); allocator.free(metadata.tensors); }

    const file = try std.fs.cwd().openFile(model_path, .{});
    defer file.close();
    const stat = try file.stat();
    const content = try allocator.alloc(u8, stat.size);
    defer allocator.free(content);
    try file.readAll(content);

    var weights = try allocator.alloc(tensor.Tensor, metadata.tensorCount);
    for (metadata.tensors, 0..) |info, idx| { weights[idx] = try gguf.loadTensorFromGGUF(allocator, content, info); }

    if (num_shards > 1) {
        for (weights, 0..) |w, idx| {
            weights[idx] = try sharding.shardWeights(w, sharding.TensorParallelConfig{ .num_shards = num_shards, .shard_rank = 0, .world_size = num_shards });
        }
    }

    const model_config = transformer.TransformerConfig{
        .vocab_size = 32000, .hidden_dim = 4096, .num_layers = 32, .num_heads = 32,
        .head_dim = 128, .intermediate_dim = 11008, .max_seq_len = 2048, .num_shards = num_shards,
    };

    std.log.info("Starting Zig Inference Engine with {d} shard(s)", .{num_shards});
    std.log.info("Model: {s}, Max tokens: {d}", .{ model_path, max_tokens });

    // Test forward pass
    const test_input = try tensor.Tensor.init(allocator, &[2]usize{ 1, 4096 }, tensor.Dtype.f16, tensor.Device.cpu);
    std.mem.set(f16, test_input.data, @f32tof16(0.5));
    const test_output = try matmul.forward(test_input, weights[0]);
    std.log.info("Forward pass test: output shape [{d}, {d}]", .{ test_output.shape[0], test_output.shape[1] });
    test_input.deinit();
    test_output.deinit();

    std.log.info("Zig Inference Engine ready. Listening on port {d}", .{port});

    const listener = try std.net.tcpListen(port);
    defer listener.close();

    while (true) {
        const socket = try listener.accept();
        defer socket.close();
        var buffer: [10240]u8 = undefined;
        const n = try socket.read(&buffer);
        if (std.mem.indexOf(u8, buffer[0..n], "POST /generate")) {
            const response = "{\"completion\":\"Hello from Zig! Engine is running with {d} shards.\"}";
            try socket.writeAll(response);
        }
    }
}
