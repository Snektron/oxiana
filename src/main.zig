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

    var swapchain = try Swapchain.init(&instance, &device, allocator, extent, .{
        .surface = surface,
        .swap_image_usage = .{.transfer_dst_bit = true},
    });
    defer swapchain.deinit();

    var renderer = try Renderer.init(allocator, &device, &swapchain);
    defer renderer.deinit();

    while (c.glfwWindowShouldClose(window) == c.GLFW_FALSE) {
        c.glfwSwapBuffers(window);
        c.glfwPollEvents();
    }
}

fn createSurface(instance: gfx.Instance, window: *c.GLFWwindow) !vk.SurfaceKHR {
    var surface: vk.SurfaceKHR = undefined;
    if (c.glfwCreateWindowSurface(instance.handle, window, null, &surface) != .success) {
        return error.SurfaceInitFailed;
    }

    return surface;
}
