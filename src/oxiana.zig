const std = @import("std");
const vk = @import("vulkan");
const c = @import("c.zig");
const gfx = @import("graphics.zig");
const math = @import("math.zig");
const Swapchain = @import("swapchain.zig").Swapchain;
const Renderer = @import("renderer.zig").Renderer;
const asManyPtr = @import("util.zig").asManyPtr;
const Allocator = std.mem.Allocator;

const initial_extent = vk.Extent2D{
    .width = 800,
    .height = 600
};

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

pub const Oxiana = struct {
    allocator: *Allocator,
    window: *c.GLFWwindow,
    instance: gfx.Instance,
    surface: vk.SurfaceKHR,
    device: gfx.Device,
    swapchain: Swapchain,
    renderer: Renderer,
    camera: math.Camera,
    mouse_captured: bool,

    pub fn init(allocator: *Allocator) !*Oxiana {
        // Oxiana is heap-allocated as it shouldn't move
        var self = try allocator.create(Oxiana);
        errdefer allocator.destroy(self);

        self.allocator = allocator;

        c.glfwWindowHint(c.GLFW_CLIENT_API, c.GLFW_NO_API);
        self.window = c.glfwCreateWindow(
            @intCast(c_int, initial_extent.width),
            @intCast(c_int, initial_extent.height),
            app_name,
            null,
            null
        ) orelse return error.WindowInitFailed;
        errdefer c.glfwDestroyWindow(self.window);
        c.glfwSetWindowUserPointer(self.window, @ptrCast(*c_void, self));
        _ = c.glfwSetKeyCallback(self.window, keyCallback);
        _ = c.glfwSetCursorPosCallback(self.window, cursorCallback);
        _ = c.glfwSetMouseButtonCallback(self.window, mouseButtonCallback);

        const glfw_exts = blk: {
            var count: u32 = 0;
            const exts = c.glfwGetRequiredInstanceExtensions(&count);
            break :blk @ptrCast([*]const [*:0]const u8, exts)[0 .. count];
        };

        self.instance = try gfx.Instance.init(c.glfwGetInstanceProcAddress, glfw_exts, app_info);
        errdefer self.instance.deinit();

        self.surface = try createSurface(self.instance, self.window);
        errdefer self.instance.vki.destroySurfaceKHR(self.instance.handle, self.surface, null);

        self.device = try self.instance.findAndCreateDevice(allocator, self.surface, required_device_extensions);
        errdefer self.device.deinit();

        std.log.info(.oxiana, "Using device '{}'", .{ self.device.pdev.name() });

        self.swapchain = try Swapchain.init(&self.instance, &self.device, allocator, initial_extent, .{
            .surface = self.surface,
            .swap_image_usage = .{.transfer_dst_bit = true},
            .vsync = false,
        });
        errdefer self.swapchain.deinit();

        self.renderer = try Renderer.init(&self.device, self.swapchain.extent);
        errdefer self.renderer.deinit();

        self.camera = .{
            .rotation = math.Quaternion(f32).identity,
            .translation = math.Vec(f32, 3).zero,
        };

        self.mouse_captured = false;

        return self;
    }

    pub fn deinit(self: *const Oxiana) void {
        self.renderer.waitForAllFrames() catch {};
        self.renderer.deinit();
        self.swapchain.deinit();

        self.device.deinit();
        self.instance.vki.destroySurfaceKHR(self.instance.handle, self.surface, null);
        self.instance.deinit();
        c.glfwDestroyWindow(self.window);
        self.allocator.destroy(self);
    }

    pub fn keyCallback(window: ?*c.GLFWwindow, key: c_int, scancode: c_int, action: c_int, mods: c_int) callconv(.C) void {
        const self = @ptrCast(*Oxiana, @alignCast(@alignOf(Oxiana), c.glfwGetWindowUserPointer(window).?));

        switch (key) {
            c.GLFW_KEY_ESCAPE => {
                self.mouse_captured = false;
                c.glfwSetInputMode(window, c.GLFW_CURSOR, c.GLFW_CURSOR_NORMAL);
            },
            else => {},
        }
    }

    pub fn cursorCallback(window: ?*c.GLFWwindow, x: f64, y: f64) callconv(.C) void {
        const self = @ptrCast(*Oxiana, @alignCast(@alignOf(Oxiana), c.glfwGetWindowUserPointer(window).?));
    }

    pub fn mouseButtonCallback(window: ?*c.GLFWwindow, button: c_int, action: c_int, mods: c_int) callconv(.C) void {
        const self = @ptrCast(*Oxiana, @alignCast(@alignOf(Oxiana), c.glfwGetWindowUserPointer(window).?));

        if (button == c.GLFW_MOUSE_BUTTON_LEFT and action == c.GLFW_PRESS and !self.mouse_captured) {
            self.mouse_captured = true;
            c.glfwSetInputMode(window, c.GLFW_CURSOR, c.GLFW_CURSOR_DISABLED);
        }
    }

    pub fn loop(self: *Oxiana) !void {
        var frame: usize = 0;
        var timer = try std.time.Timer.start();

        while (c.glfwWindowShouldClose(self.window) == c.GLFW_FALSE) {
            const swap_image = try self.swapchain.acquireNextSwapImage();
            const frame_data = try self.renderer.render(self.swapchain.extent, swap_image.image, self.camera);

            const wait_stage = [_]vk.PipelineStageFlags{.{.bottom_of_pipe_bit = true}};
            try self.device.vkd.queueSubmit(self.device.compute_queue.handle, 1, &[_]vk.SubmitInfo{.{
                .wait_semaphore_count = 1,
                .p_wait_semaphores = asManyPtr(&swap_image.image_acquired),
                .p_wait_dst_stage_mask = &wait_stage,
                .command_buffer_count = 1,
                .p_command_buffers = asManyPtr(&frame_data.cmd_buf),
                .signal_semaphore_count = 1,
                .p_signal_semaphores = asManyPtr(&swap_image.render_finished),
            }}, frame_data.frame_fence);

            const state = self.swapchain.swapBuffers() catch |err| switch (err) {
                error.OutOfDateKHR => Swapchain.PresentState.suboptimal,
                else => |narrow| return narrow,
            };

            if (state == .suboptimal) {
                var w: c_int = undefined;
                var h: c_int = undefined;
                c.glfwGetWindowSize(self.window, &w, &h);
                try self.renderer.waitForAllFrames();
                try self.swapchain.recreate(.{
                    .width = @intCast(u32, w),
                    .height = @intCast(u32, h),
                }, .{
                    .surface = self.surface,
                    .swap_image_usage = .{.transfer_dst_bit = true},
                    .vsync = false,
                });

                try self.renderer.resize(self.swapchain.extent);
            }

            c.glfwPollEvents();

            frame += 1;
            if (timer.read() > std.time.ns_per_s) {
                std.debug.print("{} fps\n", .{ frame });
                frame = 0;
                timer.reset();
            }
        }
    }
};

fn createSurface(instance: gfx.Instance, window: *c.GLFWwindow) !vk.SurfaceKHR {
    var surface: vk.SurfaceKHR = undefined;
    if (c.glfwCreateWindowSurface(instance.handle, window, null, &surface) != .success) {
        return error.SurfaceInitFailed;
    }

    return surface;
}
