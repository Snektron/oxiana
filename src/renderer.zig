const std = @import("std");
const vk = @import("vulkan");
const gfx = @import("graphics.zig");
const Swapchain = @import("swapchain.zig").Swapchain;
const Allocator = std.mem.Allocator;

pub const Renderer = struct {
    allocator: *Allocator,
    dev: *const gfx.Device,

    pipeline_layout: vk.PipelineLayout,
    pipeline: vk.Pipeline,

    cmd_pool: vk.CommandPool,
    cmd_bufs: []vk.CommandBuffer,

    pub fn init(allocator: *Allocator, dev: *const gfx.Device, swapchain: *const Swapchain) !Renderer {
        var self = Renderer{
            .allocator = allocator,
            .dev = dev,
            .pipeline_layout = .null_handle,
            .pipeline = .null_handle,
            .cmd_pool = .null_handle,
            .cmd_bufs = &[_]vk.CommandBuffer{},
        };
        errdefer self.deinit();

        try self.createPipeline();
        try self.createCommandBuffers(swapchain);

        return self;
    }

    pub fn deinit(self: Renderer) void {
        if (self.cmd_pool != .null_handle) {
            self.dev.vkd.freeCommandBuffers(self.dev.handle, self.cmd_pool, @truncate(u32, self.cmd_bufs.len), self.cmd_bufs.ptr);
            self.allocator.free(self.cmd_bufs);
            self.dev.vkd.destroyCommandPool(self.dev.handle, self.cmd_pool, null);
        }
        self.dev.vkd.destroyPipelineLayout(self.dev.handle, self.pipeline_layout, null);
    }

    fn createPipeline(self: *Renderer) !void {
        self.pipeline_layout = try self.dev.vkd.createPipelineLayout(self.dev.handle, .{
            .flags = .{},
            .set_layout_count = 0,
            .p_set_layouts = undefined,
            .push_constant_range_count = 0,
            .p_push_constant_ranges = undefined,
        }, null);
    }

    fn createCommandBuffers(self: *Renderer, swapchain: *const Swapchain) !void {
        self.cmd_pool = try self.dev.vkd.createCommandPool(self.dev.handle, .{
            .flags = .{},
            .queue_family_index = self.dev.compute_queue.family,
        }, null);

        const cmd_bufs = try self.allocator.alloc(vk.CommandBuffer, swapchain.swap_images.len);
        errdefer self.allocator.free(cmd_bufs);

        try self.dev.vkd.allocateCommandBuffers(self.dev.handle, .{
            .command_pool = self.cmd_pool,
            .level = .primary,
            .command_buffer_count = @truncate(u32, cmd_bufs.len),
        }, cmd_bufs.ptr);
        self.cmd_bufs = cmd_bufs;
    }
};
