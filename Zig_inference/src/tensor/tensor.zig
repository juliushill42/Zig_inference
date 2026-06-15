const std = @import("std");
const matmul = @import("matmul.zig");

pub const Dtype = enum(u8) {
    f32 = 0,
    f16 = 1,
    q8_0 = 2,
    q4_0 = 3,
    q4_k = 4,
};

pub const TensorError = error{
    InvalidShape,
    AllocatorFailed,
    MatrixDimensionMismatch,
};

pub const Tensor = struct {
    data: []f16,
    shape: []const usize,
    dtype: Dtype,
    device: Device,
    allocator: std.mem.Allocator,

    pub const Device = enum { cpu, cuda, vulkan };

    pub fn init(allocator: std.mem.Allocator, shape: []const usize, dtype: Dtype, device: Device) !Tensor {
        var total_elements: usize = 1;
        for (shape) |dim| total_elements *= dim;

        const data = try allocator.alloc(f16, total_elements);
        const shape_copy = try allocator.alloc(usize, shape.len);
        std.mem.copyForwards(usize, shape_copy, shape);

        return Tensor{
            .data = data,
            .shape = shape_copy,
            .dtype = dtype,
            .device = device,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Tensor) void {
        self.allocator.free(self.data);
        self.allocator.free(self.shape);
    }

    pub fn get(self: Tensor, indices: []const usize) f16 {
        var offset: usize = 0;
        var stride: usize = 1;
        var i: usize = self.shape.len;
        while (i > 1) {
            i -= 1;
            stride *= self.shape[i];
        }
        i = self.shape.len;
        while (i > 0) {
            i -= 1;
            offset += indices[i] * stride;
            if (i > 0) stride /= self.shape[i - 1];
        }
        return self.data[offset];
    }

    pub fn row(self: Tensor, row_idx: usize) []f16 {
        const row_size = self.shape[self.shape.len - 1];
        const offset = row_idx * row_size;
        return self.data[offset .. offset + row_size];
    }

    pub fn reshape(self: *Tensor, new_shape: []const usize) !void {
        var total: usize = 1;
        for (new_shape) |dim| total *= dim;
        if (total != self.data.len) return TensorError.InvalidShape;
        self.allocator.free(self.shape);
        const allocated_shape = try self.allocator.alloc(usize, new_shape.len);
        std.mem.copyForwards(usize, allocated_shape, new_shape);
        self.shape = allocated_shape;
    }

    pub fn zeros(allocator: std.mem.Allocator, shape: []const usize) !Tensor {
        var t = try Tensor.init(allocator, shape, Dtype.f16, Device.cpu);
        std.mem.set(f16, t.data, 0);
        return t;
    }
};
