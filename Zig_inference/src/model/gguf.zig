const std = @import("std");
const tensor = @import("../tensor/tensor.zig");

pub const GGUFMetadata = struct {
    version: u32,
    tensorCount: u64,
    keyValueCount: u64,
    keyValues: std.StringHashMap([]const u8),
    tensors: []GGUFTensorInfo,
};

pub const GGUFTensorInfo = struct {
    name: []const u8,
    shape: []usize,
    dtype: tensor.Dtype,
    offset: u64,
};

pub fn parseGGUF(allocator: std.mem.Allocator, path: []const u8) !GGUFMetadata {
    var file = try std.fs.cwd().openFile(path, .{});
    defer file.close();
    const stat = try file.stat();
    const content = try allocator.alloc(u8, stat.size);
    defer allocator.free(content);
    try file.readAll(content);
    const magic = std.mem.readInt(u32, content[0..4], .little);
    if (magic != 0x67677566) return std.error.InvalidGGUF;
    var offset: usize = 16;
    const version = std.mem.readInt(u32, content[4..8], .little);
    const tensorCount = std.mem.readInt(u64, content[8..16], .little);
    const keyValueCount = std.mem.readInt(u64, content[16..24], .little);
    var keyValues = std.StringHashMap([]const u8).init(allocator);
    defer keyValues.deinit();
    var i: usize = 0;
    while (i < keyValueCount) : (i += 1) {
        const key_len = std.mem.readInt(u32, content[offset..offset + 4], .little);
        offset += 4;
        offset += key_len + 8;
    }
    var tensors = try allocator.alloc(GGUFTensorInfo, tensorCount);
    i = 0;
    while (i < tensorCount) : (i += 1) {
        const name_len = std.mem.readInt(u32, content[offset..offset + 4], .little);
        offset += 4;
        tensors[i].name = content[offset..offset + name_len];
        offset += name_len;
        const shape_len = std.mem.readInt(u32, content[offset..offset + 4], .little);
        offset += 4;
        tensors[i].shape = try allocator.alloc(usize, shape_len);
        for (tensors[i].shape) |_, j| { tensors[i].shape[j] = std.mem.readInt(u64, content[offset..offset + 8], .little); offset += 8; }
        tensors[i].dtype = @enumFromInt(std.mem.readInt(u32, content[offset..offset + 4], .little));
        offset += 4;
        tensors[i].offset = std.mem.readInt(u64, content[offset..offset + 8], .little);
        offset += 8;
    }
    return GGUFMetadata{ .version = version, .tensorCount = tensorCount, .keyValueCount = keyValueCount, .keyValues = keyValues, .tensors = tensors };
}

pub fn loadTensorFromGGUF(allocator: std.mem.Allocator, content: []u8, info: GGUFTensorInfo) !tensor.Tensor {
    const data_len = info.shape[0] * info.shape[1];
    const data = try allocator.alloc(f16, data_len);
    std.mem.copyForwards(f16, data, @as([]const f16, @ptrCast(content[info.offset..info.offset + data_len * 2])));
    return tensor.Tensor{ .data = data, .shape = info.shape, .dtype = info.dtype, .device = tensor.Device.cpu, .allocator = allocator };
}
