const std = @import("std");
const vk = @import("vulkan");
const c = @import("c.zig");
const gfx = @import("graphics.zig");
const Swapchain = @import("swapchain.zig").Swapchain;
const Renderer = @import("renderer.zig").Renderer;

const app_name = "Oxiana";
const required_device_extensions = &[_][*:0]const u8{
    vk.extension_info.khr_swapchain.name,
};

const app_info = .{
    .p_application_name = app_name,
    .application_version = vk.makeVersion(0, 0, 0),
    .p_engine_name = app_name,
    .engine_version = vk.makeVersion(0, 0, 0),
    .api_version = vk.API_VERSION_1_2,
};

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    if (c.glfwInit() != c.GLFW_TRUE) return error.GlfwInitFailed;
    defer c.glfwTerminate();

    var extent = vk.Extent2D{.width = 800, .height = 600};

    c.glfwWindowHint(c.GLFW_CLIENT_API, c.GLFW_NO_API);
    const window = c.glfwCreateWindow(
        @intCast(c_int, extent.width),
        @intCast(c_int, extent.height),
        app_name,
        null,
        null
    ) orelse return error.WindowInitFailed;
    defer c.glfwDestroyWindow(window);

    const glfw_exts = blk: {
        var count: u32 = 0;
        const exts = c.glfwGetRequiredInstanceExtensions(&count);
        break :blk @ptrCast([*]const [*:0]const u8, exts)[0 .. count];
    };

    const instance = try gfx.Instance.init(c.glfwGetInstanceProcAddress, glfw_exts, app_info);
    defer instance.deinit();

    const surface = try createSurface(instance, window);
    defer instance.vki.destroySurfaceKHR(instance.handle, surface, null);

    const device = try instance.findAndCreateDevice(allocator, surface, required_device_extensions);
    defer device.deinit();

    std.log.info(.main, "Using device '{}'", .{device.pdev.name()});

    const swapchain_options = .{
        .surface = surface,
        .swap_image_usage = .{.transfer_dst_bit = true},
    };

    var swapchain = try Swapchain.init(&instance, &device, allocator, extent, swapchain_options);
    defer swapchain.deinit();

    var renderer = try Renderer.init(allocator, &device, &swapchain);
    defer renderer.deinit();

    while (c.glfwWindowShouldClose(window) == c.GLFW_FALSE) {
        const fence = try swapchain.acquireFrameFence();
        const image_acquired = swapchain.currentImageAcquiredSem();
        const render_finished = swapchain.currentRenderFinishedSem();

        const cmd_buf = try renderer.render(swapchain.image_index);

        const wait_stage = [_]vk.PipelineStageFlags{.{.bottom_of_pipe_bit = true}};
        try device.vkd.queueSubmit(device.graphics_queue.handle, 1, &[_]vk.SubmitInfo{.{
            .wait_semaphore_count = 1,
            .p_wait_semaphores = @ptrCast([*]const vk.Semaphore, &image_acquired),
            .p_wait_dst_stage_mask = &wait_stage,
            .command_buffer_count = 1,
            .p_command_buffers = @ptrCast([*]const vk.CommandBuffer, &cmd_buf),
            .signal_semaphore_count = 1,
            .p_signal_semaphores = @ptrCast([*]const vk.Semaphore, &render_finished),
        }}, fence);

        const state = swapchain.swapBuffers() catch |err| switch (err) {
            error.OutOfDateKHR => Swapchain.PresentState.suboptimal,
            else => |narrow| return narrow,
        };

        if (state == .suboptimal) {
            var w: c_int = undefined;
            var h: c_int = undefined;
            c.glfwGetWindowSize(window, &w, &h);
            extent.width = @intCast(u32, w);
            extent.height = @intCast(u32, h);
            // try swapchain.waitForAllFrames();
            try swapchain.recreate(extent, swapchain_options);

            // TODO: Optimize
            renderer.deinit();
            renderer = try Renderer.init(allocator, &device, &swapchain);
        }

        c.glfwSwapBuffers(window);
        c.glfwPollEvents();
    }
    try swapchain.waitForAllFrames();
}

fn createSurface(instance: gfx.Instance, window: *c.GLFWwindow) !vk.SurfaceKHR {
    var surface: vk.SurfaceKHR = undefined;
    if (c.glfwCreateWindowSurface(instance.handle, window, null, &surface) != .success) {
        return error.SurfaceInitFailed;
    }

    return surface;
}
