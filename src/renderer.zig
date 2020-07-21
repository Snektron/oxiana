const std = @import("std");
const vk = @import("vulkan");
const gfx = @import("graphics.zig");
const Swapchain = @import("swapchain.zig").Swapchain;
const resources = @import("resources");
const Allocator = std.mem.Allocator;

// Thus must kept in sync with the bindings in shaders/traverse.comp
const bindings = [_]vk.DescriptorSetLayoutBinding{
    .{ // layout(binding = 0, rgba8) restrict writeonly uniform image2D render_target;
        .binding = 0,
        .descriptor_type = .storage_image,
        .descriptor_count = 1,
        .stage_flags = .{.compute_bit = true},
        .p_immutable_samplers = null,
    },
};

pub const Renderer = struct {
    allocator: *Allocator,
    dev: *const gfx.Device,

    descriptor_set_layout: vk.DescriptorSetLayout,
    pipeline_layout: vk.PipelineLayout,
    pipeline: vk.Pipeline,

    descriptor_pool: vk.DescriptorPool,

    cmd_pool: vk.CommandPool,
    cmd_bufs: []vk.CommandBuffer,

    pub fn init(allocator: *Allocator, dev: *const gfx.Device, swapchain: *const Swapchain) !Renderer {
        var self = Renderer{
            .allocator = allocator,
            .dev = dev,
            .descriptor_set_layout = .null_handle,
            .pipeline_layout = .null_handle,
            .pipeline = .null_handle,
            .descriptor_pool = .null_handle,
            .cmd_pool = .null_handle,
            .cmd_bufs = &[_]vk.CommandBuffer{},
        };
        errdefer self.deinit();

        try self.createPipeline();
        try self.createDescriptorSet(swapchain);
        try self.createCommandBuffers(swapchain);

        return self;
    }

    pub fn deinit(self: Renderer) void {
        if (self.cmd_pool != .null_handle) {
            self.dev.vkd.freeCommandBuffers(self.dev.handle, self.cmd_pool, @truncate(u32, self.cmd_bufs.len), self.cmd_bufs.ptr);
            self.allocator.free(self.cmd_bufs);
            self.dev.vkd.destroyCommandPool(self.dev.handle, self.cmd_pool, null);
        }
        self.dev.vkd.destroyDescriptorPool(self.dev.handle, self.descriptor_pool, null);
        self.dev.vkd.destroyPipeline(self.dev.handle, self.pipeline, null);
        self.dev.vkd.destroyPipelineLayout(self.dev.handle, self.pipeline_layout, null);
        self.dev.vkd.destroyDescriptorSetLayout(self.dev.handle, self.descriptor_set_layout, null);
    }

    fn createPipeline(self: *Renderer) !void {
        self.descriptor_set_layout = try self.dev.vkd.createDescriptorSetLayout(self.dev.handle, .{
            .flags = .{},
            .binding_count = @intCast(u32, bindings.len),
            .p_bindings = &bindings,
        }, null);

        self.pipeline_layout = try self.dev.vkd.createPipelineLayout(self.dev.handle, .{
            .flags = .{},
            .set_layout_count = 1,
            .p_set_layouts = @ptrCast([*]const vk.DescriptorSetLayout, &self.descriptor_set_layout),
            .push_constant_range_count = 0,
            .p_push_constant_ranges = undefined,
        }, null);

        const traversal_shader = try self.dev.vkd.createShaderModule(self.dev.handle, .{
            .flags = .{},
            .code_size = resources.traverse_comp.len,
            .p_code = @ptrCast([*]const u32, resources.traverse_comp),
        }, null);
        defer self.dev.vkd.destroyShaderModule(self.dev.handle, traversal_shader, null);

        const cpci = vk.ComputePipelineCreateInfo{
            .flags = .{},
            .stage = .{
                .flags = .{},
                .stage = .{.compute_bit = true},
                .module = traversal_shader,
                .p_name = "main",
                .p_specialization_info = null,
            },
            .layout = self.pipeline_layout,
            .base_pipeline_handle = .null_handle,
            .base_pipeline_index = 0,
        };

        _ = try self.dev.vkd.createComputePipelines(
            self.dev.handle,
            .null_handle,
            1,
            @ptrCast([*]const vk.ComputePipelineCreateInfo, &cpci),
            null,
            @ptrCast([*]vk.Pipeline, &self.pipeline),
        );
    }

    fn createDescriptorSet(self: *Renderer, swapchain: *const Swapchain) !void {
        const n_swap_images = @intCast(u32, swapchain.swap_images.len);
        var pool_sizes: [bindings.len]vk.DescriptorPoolSize = undefined;
        var n_pool_sizes: u32 = 0;

        for (bindings) |binding| {
            for (pool_sizes[0 .. n_pool_sizes]) |*pool_size| {
                if (pool_size.@"type" == binding.descriptor_type) {
                    pool_size.descriptor_count += binding.descriptor_count * n_swap_images;
                    break;
                }
            } else {
                pool_sizes[n_pool_sizes] = .{
                    .@"type" = binding.descriptor_type,
                    .descriptor_count = binding.descriptor_count * n_swap_images,
                };
                n_pool_sizes += 1;
            }
        }

        self.descriptor_pool = try self.dev.vkd.createDescriptorPool(self.dev.handle, .{
            .flags = .{},
            .max_sets = n_swap_images,
            .pool_size_count = n_pool_sizes,
            .p_pool_sizes = &pool_sizes,
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
