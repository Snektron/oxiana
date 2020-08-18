const std = @import("std");
const vk = @import("vulkan");
const c = @import("c.zig");
const gfx = @import("graphics.zig");
const math = @import("math.zig");
const vv = @import("voxel/volume.zig");
const Swapchain = @import("swapchain.zig").Swapchain;
const Renderer = @import("renderer.zig").Renderer;
const asManyPtr = @import("util.zig").asManyPtr;
const Allocator = std.mem.Allocator;

const VoxelVolume = vv.VoxelVolume(8, u8); // 2 ^ 8 = 256

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

const sensivity = .{
    .mouse = 0.001,
    .roll = 1.2,
    .movement = 100,
};

const Input = struct {
    mouse_captured: bool,
    last_mouse_pos: math.Vec(f32, 2),
    mouse_pos: math.Vec(f32, 2),
    forward: bool,
    backward: bool,
    left: bool,
    right: bool,
    up: bool,
    down: bool,
    roll_left: bool,
    roll_right: bool,

    fn init() Input {
        return .{
            .mouse_captured = false,
            .last_mouse_pos = math.vec2(f32, 0, 0),
            .mouse_pos = math.vec2(f32, 0, 0),
            .forward = false,
            .backward = false,
            .left = false,
            .right = false,
            .up = false,
            .down = false,
            .roll_left = false,
            .roll_right = false,
        };
    }

    fn bind(self: *Input, window: *c.GLFWwindow) void {
        c.glfwSetWindowUserPointer(window, @ptrCast(*c_void, self));
        _ = c.glfwSetKeyCallback(window, keyCallback);
        _ = c.glfwSetCursorPosCallback(window, cursorCallback);
        _ = c.glfwSetMouseButtonCallback(window, mouseButtonCallback);
    }

    fn keyCallback(window: ?*c.GLFWwindow, key: c_int, scancode: c_int, action: c_int, mods: c_int) callconv(.C) void {
        const self = @ptrCast(*Input, @alignCast(@alignOf(Input), c.glfwGetWindowUserPointer(window).?));
        const down = action == c.GLFW_PRESS or action == c.GLFW_REPEAT;
        switch (key) {
            c.GLFW_KEY_ESCAPE => {
                if (down)
                self.mouse_captured = false;
                c.glfwSetInputMode(window, c.GLFW_CURSOR, c.GLFW_CURSOR_NORMAL);
            },
            c.GLFW_KEY_W => self.forward = down,
            c.GLFW_KEY_S => self.backward = down,
            c.GLFW_KEY_D => self.right = down,
            c.GLFW_KEY_A => self.left = down,
            c.GLFW_KEY_SPACE => self.up = down,
            c.GLFW_KEY_LEFT_SHIFT => self.down = down,
            c.GLFW_KEY_Q => self.roll_left = down,
            c.GLFW_KEY_E => self.roll_right = down,
            else => {},
        }
    }

    fn cursorCallback(window: ?*c.GLFWwindow, x: f64, y: f64) callconv(.C) void {
        const self = @ptrCast(*Input, @alignCast(@alignOf(Input), c.glfwGetWindowUserPointer(window).?));
        self.mouse_pos = math.vec2(f32, @floatCast(f32, x), @floatCast(f32, y));
    }

    fn mouseButtonCallback(window: ?*c.GLFWwindow, button: c_int, action: c_int, mods: c_int) callconv(.C) void {
        const self = @ptrCast(*Input, @alignCast(@alignOf(Input), c.glfwGetWindowUserPointer(window).?));

        if (button == c.GLFW_MOUSE_BUTTON_LEFT and action == c.GLFW_PRESS and !self.mouse_captured) {
            self.mouse_captured = true;
            c.glfwSetInputMode(window, c.GLFW_CURSOR, c.GLFW_CURSOR_DISABLED);
        }
    }

    fn update(self: *Input) void {
        self.last_mouse_pos = self.mouse_pos;
    }

    fn forwardMovement(self: Input) f32 {
        return @intToFloat(f32, @boolToInt(self.forward)) - @intToFloat(f32, @boolToInt(self.backward));
    }

    fn rightMovement(self: Input) f32 {
        return @intToFloat(f32, @boolToInt(self.right)) - @intToFloat(f32, @boolToInt(self.left));
    }

    fn upMovement(self: Input) f32 {
        return @intToFloat(f32, @boolToInt(self.up)) - @intToFloat(f32, @boolToInt(self.down));
    }

    fn roll(self: Input) f32 {
        return @intToFloat(f32, @boolToInt(self.roll_right)) - @intToFloat(f32, @boolToInt(self.roll_left));
    }
};

fn encodeR3G3B2(red: u3, green: u3, blue: u2) u8 {
    return (@as(u8, red) << (2 + 3)) | (@as(u8, green) << 2) | @as(u8, blue);
}

fn initVolume(allocator: *Allocator) !*VoxelVolume {
    const volume = try allocator.create(VoxelVolume);
    volume.clear(0);
    const dim = VoxelVolume.side_dim;

    var x: u32 = 0;
    while (x < dim) : (x += 1) {
        var y: u32 = 0;
        while (y < dim) : (y += 1) {
            var z: u32 = 0;
            while (z < dim) : (z += 1) {
                if (x * x + y * y + z * z < (dim + 1) * (dim + 1)) {
                    const color = encodeR3G3B2(@truncate(u3, x), @truncate(u3, y), @truncate(u2, z));
                    volume.voxels[x][y][z] = color;
                }
            }
        }
    }

    return volume;
}

pub const Oxiana = struct {
    allocator: *Allocator,
    volume: *VoxelVolume,
    window: *c.GLFWwindow,
    instance: gfx.Instance,
    surface: vk.SurfaceKHR,
    device: gfx.Device,
    swapchain: Swapchain,
    renderer: Renderer(VoxelVolume),
    camera: math.Camera,
    input: Input,

    pub fn init(allocator: *Allocator) !*Oxiana {
        // Oxiana is heap-allocated as it shouldn't move
        var self = try allocator.create(Oxiana);
        errdefer allocator.destroy(self);

        self.allocator = allocator;
        self.volume = try initVolume(allocator);

        c.glfwWindowHint(c.GLFW_CLIENT_API, c.GLFW_NO_API);
        self.window = c.glfwCreateWindow(
            @intCast(c_int, initial_extent.width),
            @intCast(c_int, initial_extent.height),
            app_name,
            null,
            null
        ) orelse return error.WindowInitFailed;
        errdefer c.glfwDestroyWindow(self.window);

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

        std.log.info("Using device '{}'", .{ self.device.pdev.name() });

        self.swapchain = try Swapchain.init(&self.instance, &self.device, allocator, initial_extent, .{
            .surface = self.surface,
            .swap_image_usage = .{.transfer_dst_bit = true},
            .vsync = false,
        });
        errdefer self.swapchain.deinit();

        self.renderer = try Renderer(VoxelVolume).init(&self.device, self.swapchain.extent, self.volume);
        errdefer self.renderer.deinit();

        self.camera = .{
            .rotation = math.Quaternion(f32).identity,
            .translation = math.vec3(f32, 300, 300, 300),
        };

        self.input = Input.init();
        self.input.bind(self.window);

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
        self.allocator.destroy(self.volume);
        self.allocator.destroy(self);
    }

    pub fn loop(self: *Oxiana) !void {
        var timer = try std.time.Timer.start();
        var t: f32 = 0;
        var frame: usize = 0;

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

            self.input.update();
            c.glfwPollEvents();

            const dt = @intToFloat(f32, timer.lap()) / std.time.ns_per_s;
            t += dt;
            frame += 1;
            if (t > 1) {
                std.debug.print("FPS: {d:.2}\n", .{ @intToFloat(f32, frame) / t});
                t = 0;
                frame = 0;
            }

            if (self.input.mouse_captured) {
                const mouse_movement = self.input.mouse_pos.sub(self.input.last_mouse_pos).scale(-sensivity.mouse);
                self.camera.rotate(math.Quaternion(f32).axisAngle(mouse_movement.swizzle("yx0"), 1));
                self.camera.rotateRoll(-self.input.roll() * sensivity.roll * dt);
                self.camera.moveForward(self.input.forwardMovement() * sensivity.movement * dt);
                self.camera.moveRight(-self.input.rightMovement() * sensivity.movement * dt);
                self.camera.moveUp(-self.input.upMovement() * sensivity.movement * dt);
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
