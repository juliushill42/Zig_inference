const std = @import("std");
const tensor = @import("../tensor/tensor.zig");

pub const CpuBackend = struct {
    pub fn init() CpuBackend { return CpuBackend{}; }
    pub fn allocate(self: CpuBackend, size: usize) ![]f16 { return try std.heap.page_allocator.alloc(f16, size); }
};
