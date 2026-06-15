const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{
        .default_target = .{
            .cpu_arch = .x86_64,
            .os_tag = .linux,
            .abi = .gnu,
        },
    });
    const optimize = b.standardOptimizeOption(.{});

    const exe = b.addExecutable(.{
        .name = "zig-inference",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });

    if (std.File.exists("/usr/local/cuda/lib64/libcudart.so")) {
        exe.linkSystemLibrary("cuda");
        exe.linkSystemLibrary("nccl");
        exe.addIncludePath("/usr/local/cuda/include");
    }

    if (std.File.exists("/usr/lib/x86_64-linux-gnu/libvulkan.so")) {
        exe.linkSystemLibrary("vulkan");
        exe.addIncludePath("/usr/include/vulkan");
    }

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the application");
    run_step.dependOn(&run_cmd.step);

    const tests = b.addTest(.{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    const run_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_tests.step);
}
