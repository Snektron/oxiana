const std = @import("std");
const vk_gen = @import("deps/vulkan-zig/generator/index.zig");
const Builder = std.build.Builder;
const Step = std.build.Step;
const path = std.fs.path;

pub const ResourceGenStep = struct {
    step: Step,
    shader_step: *vk_gen.ShaderCompileStep,
    builder: *Builder,
    full_out_path: []const u8,
    resources: std.ArrayList(u8),

    pub fn init(builder: *Builder, out: []const u8) *ResourceGenStep {
        const self = builder.allocator.create(ResourceGenStep) catch unreachable;
        self.* = .{
            .step = Step.init(.Custom, "resources", builder.allocator, make),
            .shader_step = vk_gen.ShaderCompileStep.init(builder, &[_][]const u8{"glslc", "--target-env=vulkan1.2"}),
            .builder = builder,
            .full_out_path = path.join(builder.allocator, &[_][]const u8{
                self.builder.build_root,
                builder.cache_root,
                out,
            }) catch unreachable,
            .resources = std.ArrayList(u8).init(builder.allocator),
        };

        self.step.dependOn(&self.shader_step.step);
        return self;
    }

    pub fn addShader(self: *ResourceGenStep, name: []const u8, source: []const u8) void {
        self.resources.writer().print(
            "pub const {} = @embedFile(\"{}\");\n",
            .{name, self.shader_step.add(source)}
        ) catch unreachable;
    }

    fn make(step: *Step) !void {
        const self = @fieldParentPtr(ResourceGenStep, "step", step);
        const cwd = std.fs.cwd();

        const dir = path.dirname(self.full_out_path).?;
        try cwd.makePath(dir);
        try cwd.writeFile(self.full_out_path, self.resources.items);
    }
};

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

    const res = ResourceGenStep.init(b, "resources.zig");
    exe.step.dependOn(&res.step);
    exe.addPackagePath("resources", res.full_out_path);

    const run_cmd = exe.run();
    run_cmd.step.dependOn(b.getInstallStep());

    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);
}
