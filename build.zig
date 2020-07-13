const std = @import("std");
const Builder = std.build.Builder;
const vk_gen = @import("deps/vulkan-zig/generator/index.zig");

pub fn build(b: *Builder) void {
    const target = b.standardTargetOptions(.{});
    const mode = b.standardReleaseOptions();

    const exe = b.addExecutable("oxiana", "src/main.zig");
    exe.setTarget(target);
    exe.setBuildMode(mode);
    exe.install();
    exe.linkSystemLibrary("c");
    exe.linkSystemLibrary("glfw");

    const vk_sdk_path = std.os.getenv("VULKAN_SDK").?;
    const vk_xml_path = std.fs.path.join(
        b.allocator,
        &[_][]const u8{vk_sdk_path, "share/vulkan/registry/vk.xml"},
    ) catch unreachable;
    const vk_gen_step = vk_gen.VkGenerateStep.init(b, vk_xml_path, "vk.zig");
    exe.step.dependOn(&vk_gen_step.step);
    exe.addPackagePath("vulkan", vk_gen_step.full_out_path);

    const run_cmd = exe.run();
    run_cmd.step.dependOn(b.getInstallStep());

    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);
}
