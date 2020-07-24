const std = @import("std");
const vk = @import("vulkan");
const gfx = @import("graphics.zig");
const Swapchain = @import("swapchain.zig").Swapchain;
const StructOfArrays = @import("soa.zig").StructOfArrays;
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
    const FrameResourceArray = StructOfArrays(struct {
        descriptor_sets: vk.DescriptorSet,
        cmd_bufs: vk.CommandBuffer,
        render_targets: vk.Image,
        render_target_views: vk.ImageView,
    });

    allocator: *Allocator,
    dev: *const gfx.Device,

    descriptor_set_layout: vk.DescriptorSetLayout,
    pipeline_layout: vk.PipelineLayout,
    pipeline: vk.Pipeline,

    descriptor_pool: vk.DescriptorPool,
    cmd_pool: vk.CommandPool,

    frame_resources: FrameResourceArray,
    render_target_memory: vk.DeviceMemory,

    pub fn init(allocator: *Allocator, dev: *const gfx.Device, swapchain: *const Swapchain) !Renderer {
        var self = Renderer{
            .allocator = allocator,
            .dev = dev,
            .descriptor_set_layout = .null_handle,
            .pipeline_layout = .null_handle,
            .pipeline = .null_handle,
            .descriptor_pool = .null_handle,
            .cmd_pool = .null_handle,
            .frame_resources = try FrameResourceArray.alloc(allocator, swapchain.swap_images.len),
            .render_target_memory = .null_handle,
        };

        // To make deinit on error easier
        for (self.frame_resources.slice("descriptor_sets")) |*set| set.* = .null_handle;
        for (self.frame_resources.slice("cmd_bufs")) |*cmd_buf| cmd_buf.* = .null_handle;
        for (self.frame_resources.slice("render_targets")) |*rt| rt.* = .null_handle;
        for (self.frame_resources.slice("render_target_views")) |*rtv| rtv.* = .null_handle;

        errdefer self.deinit();

        try self.createPipeline();
        try self.createDescriptorSets();
        try self.createCommandBuffers();
        try self.createRenderTargets(swapchain);
        // // Resource creation done at this point
        self.updateDescriptorSets();

        return self;
    }

    pub fn deinit(self: Renderer) void {
        self.dev.vkd.freeMemory(self.dev.handle, self.render_target_memory, null);

        for (self.frame_resources.slice("render_target_views")) |rtv| {
            self.dev.vkd.destroyImageView(self.dev.handle, rtv, null);
        }

        for (self.frame_resources.slice("render_targets")) |rt| {
            self.dev.vkd.destroyImage(self.dev.handle, rt, null);
        }

        // The descriptor sets and command buffers do not need to be free'd, as they are freed when
        // their respective pool is destroyed.

        self.dev.vkd.destroyCommandPool(self.dev.handle, self.cmd_pool, null);
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

    fn createDescriptorSets(self: *Renderer) !void {
        const n_swap_images = @truncate(u32, self.frame_resources.len);
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

        const layouts = try self.allocator.alloc(vk.DescriptorSetLayout, n_swap_images);
        defer self.allocator.free(layouts);

        for (layouts) |*layout| layout.* = self.descriptor_set_layout;

        try self.dev.vkd.allocateDescriptorSets(self.dev.handle, .{
            .descriptor_pool = self.descriptor_pool,
            .descriptor_set_count = @truncate(u32, layouts.len),
            .p_set_layouts = layouts.ptr,
        }, self.frame_resources.slice("descriptor_sets").ptr);
    }

    fn createCommandBuffers(self: *Renderer) !void {
        self.cmd_pool = try self.dev.vkd.createCommandPool(self.dev.handle, .{
            .flags = .{.reset_command_buffer_bit = true},
            .queue_family_index = self.dev.compute_queue.family,
        }, null);

        try self.dev.vkd.allocateCommandBuffers(self.dev.handle, .{
            .command_pool = self.cmd_pool,
            .level = .primary,
            .command_buffer_count = @truncate(u32, self.frame_resources.len),
        }, self.frame_resources.slice("cmd_bufs").ptr);
    }

    fn createRenderTargets(self: *Renderer, swapchain: *const Swapchain) !void {
        const format = .r8g8b8a8_unorm; // Format always supported as storage image, see vk spec table 63.
        for (self.frame_resources.slice("render_targets")) |*rt| {
            rt.* = try self.dev.vkd.createImage(self.dev.handle, .{
                .flags = .{},
                .image_type = .@"2d",
                .format = format, 
                .extent = .{.width = swapchain.extent.width, .height = swapchain.extent.height, .depth = 1},
                .mip_levels = 1,
                .array_layers = 1,
                .samples = .{.@"1_bit" = true},
                .tiling = .optimal,
                .usage = .{.storage_bit = true, .transfer_src_bit = true},
                .sharing_mode = .exclusive,
                .queue_family_index_count = 0,
                .p_queue_family_indices = undefined,
                .initial_layout = .@"undefined",
            }, null);
        }

        // https://www.khronos.org/registry/vulkan/specs/1.2-extensions/html/vkspec.html#resources-association
        // According to the notes under VkMemoryRequirements, the size (and probably the alignment) are equal for
        // images created with a same specific set of parameters, which is the case here. We simply query for the
        // requirements of one and multiply the size by the number of images.

        // Assuming there is at least one swap image seems reasonable.
        var mem_reqs = self.dev.vkd.getImageMemoryRequirements(self.dev.handle, self.frame_resources.at("render_targets", 0).*);
        const adjusted_size = std.mem.alignForwardGeneric(vk.DeviceSize, mem_reqs.size, mem_reqs.alignment);
        mem_reqs.size = adjusted_size * self.frame_resources.len;

        self.render_target_memory = try self.dev.allocate(mem_reqs, .{.device_local_bit = true});

        for (self.frame_resources.slice("render_targets")) |rt, i| {
            try self.dev.vkd.bindImageMemory(self.dev.handle, rt, self.render_target_memory, adjusted_size * i);
        }

        for (self.frame_resources.slice("render_target_views")) |*rtv, i| {
            const rt = self.frame_resources.at("render_targets", i).*;

            rtv.* = try self.dev.vkd.createImageView(self.dev.handle, .{
                .flags = .{},
                .image = rt,
                .view_type = .@"2d",
                .format = format,
                .components = .{.r = .identity, .g = .identity, .b = .identity, .a = .identity},
                .subresource_range = .{
                    .aspect_mask = .{.color_bit = true},
                    .base_mip_level = 0,
                    .level_count = 1,
                    .base_array_layer = 0,
                    .layer_count = 1,
                },
            }, null);
        }
    }

    fn updateDescriptorSets(self: Renderer) void {
        for (self.frame_resources.slice("descriptor_sets")) |set, i| {
            const render_target_write = vk.DescriptorImageInfo{
                .sampler = .null_handle,
                .image_view = self.frame_resources.at("render_target_views", i).*,
                .image_layout = .general,
            };

            const writes = [_]vk.WriteDescriptorSet{
                .{
                    .dst_set = set,
                    .dst_binding = bindings[0].binding,
                    .dst_array_element = 0,
                    .descriptor_count = 1,
                    .descriptor_type = bindings[0].descriptor_type,
                    .p_image_info = @ptrCast([*]const vk.DescriptorImageInfo, &render_target_write),
                    .p_buffer_info = undefined,
                    .p_texel_buffer_view = undefined,
                },
            };

            // Could do a single updateDescriptorSets, but that would require allocating an array of writes.
            self.dev.vkd.updateDescriptorSets(self.dev.handle, @intCast(u32, writes.len), &writes, 0, undefined);
        }
    }
};
