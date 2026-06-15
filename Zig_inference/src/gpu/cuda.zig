const std = @import("std");
const tensor = @import("../tensor/tensor.zig");

pub const CUdeviceptr = u64;
pub const nclComm_t = ?*anyopaque;
pub const cudaStream_t = ?*anyopaque;

pub const CudaBackend = struct {
    device_id: c_int,
    stream: cudaStream_t,
    nccl_comm: nclComm_t,

    pub fn init(device_id: c_int, world_size: c_int, rank: c_int) !CudaBackend {
        var nccl_comm: nclComm_t = null;
        return CudaBackend{ .device_id = device_id, .stream = null, .nccl_comm = nccl_comm };
    }

    pub fn allocateDeviceMemory(self: CudaBackend, size: usize) !CUdeviceptr { return 0; }
    pub fn freeDeviceMemory(self: CudaBackend, dev_ptr: CUdeviceptr) void { _ = self; }
    pub fn hostToDevice(self: CudaBackend, dev_dst: CUdeviceptr, host_src: []const f16) !void { _ = self; _ = dev_dst; _ = host_src; }
    pub fn deviceToHost(self: CudaBackend, host_dst: []f16, dev_src: CUdeviceptr) !void { _ = self; _ = host_dst; _ = dev_src; }
    pub fn executeAllReduce(self: CudaBackend, dev_buffer: CUdeviceptr, count: usize) !void { _ = self; _ = dev_buffer; _ = count; }
};
