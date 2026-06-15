const std = @import("std");

pub const VkInstance = ?*anyopaque;
pub const VkPhysicalDevice = ?*anyopaque;
pub const VkDevice = ?*anyopaque;
pub const VkBuffer = ?*anyopaque;
pub const VkDeviceMemory = ?*anyopaque;

pub const VulkanBackend = struct {
    instance: VkInstance,
    physical_device: VkPhysicalDevice,
    device: VkDevice,

    pub fn init(instance: VkInstance, physical_device: VkPhysicalDevice, device: VkDevice) !VulkanBackend {
        return VulkanBackend{ .instance = instance, .physical_device = physical_device, .device = device };
    }

    pub fn createGpuBuffer(self: VulkanBackend, size: u64, usage_flags: u32) !VkBuffer { return null; }
    pub fn streamToDevice(self: VulkanBackend, memory: VkDeviceMemory, size: u64, data: []const f16) !void { _ = self; _ = memory; _ = size; _ = data; }
};
