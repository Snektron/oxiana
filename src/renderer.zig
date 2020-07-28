const std = @import("std");
const vk = @import("vulkan");
const gfx = @import("graphics.zig");
const math = @import("math.zig");
const Swapchain = @import("swapchain.zig").Swapchain;
const resources = @import("resources");
const Allocator = std.mem.Allocator;
const asManyPtr = @import("util.zig").asManyPtr;

const max_frames_in_flight = 2;

// Keep in sync with shaders/traverse.comp
const workgroup_size = vk.Extent2D{.width = 8, .height = 8};

// Keep in sync with shaders/traverse.comp
const bindings = [_]vk.DescriptorSetLayoutBinding{
    .{ // layout(binding = 0, rgba8) restrict writeonly uniform image2D render_target;
        .binding = 0,
        .descriptor_type = .storage_image,
        .descriptor_count = 1,
        .stage_flags = .{.compute_bit = true},
        .p_immutable_samplers = null,
    },
};

// Keep in sync with shaders/traverse.comp
const PushConstantBuffer = extern struct {
    // Use vec4's to get alignment correct
    forward: math.Vec(f32, 4),
    up: math.Vec(f32, 4),
    translation: math.Vec(f32, 4),
};

pub const Renderer = struct {
    const FrameData = struct {
        frame_fence: vk.Fence,
        cmd_buf: vk.CommandBuffer,
    };

    dev: *const gfx.Device,

    descriptor_set_layout: vk.DescriptorSetLayout,
    pipeline_layout: vk.PipelineLayout,
    pipeline: vk.Pipeline,

    descriptor_pool: vk.DescriptorPool,

    frame_index: usize,
    frame_resources: struct {
        frame_fences: [max_frames_in_flight]vk.Fence,
        descriptor_sets: [max_frames_in_flight]vk.DescriptorSet,
        cmd_pools: [max_frames_in_flight]vk.CommandPool,
        cmd_bufs: [max_frames_in_flight]vk.CommandBuffer,
        render_targets: [max_frames_in_flight]vk.Image,
        render_target_views: [max_frames_in_flight]vk.ImageView,
    },

    render_target_memory: vk.DeviceMemory,

    pub fn init(dev: *const gfx.Device, extent: vk.Extent2D) !Renderer {
        var self = Renderer{
            .dev = dev,
            .descriptor_set_layout = .null_handle,
            .pipeline_layout = .null_handle,
            .pipeline = .null_handle,
            .descriptor_pool = .null_handle,
            .frame_index = 0,
            .frame_resources = undefined,
            .render_target_memory = .null_handle,
        };

        // To make deinit on error easier
        for (self.frame_resources.frame_fences) |*fence| fence.* = .null_handle;
        for (self.frame_resources.descriptor_sets) |*set| set.* = .null_handle;
        for (self.frame_resources.cmd_pools) |*cmd_pool| cmd_pool.* = .null_handle;
        for (self.frame_resources.cmd_bufs) |*cmd_buf| cmd_buf.* = .null_handle;
        for (self.frame_resources.render_targets) |*rt| rt.* = .null_handle;
        for (self.frame_resources.render_target_views) |*rtv| rtv.* = .null_handle;

        errdefer self.deinit();

        try self.createFences();
        try self.createPipeline();
        try self.createDescriptorSets();
        try self.createCommandBuffers();
        try self.createRenderTargets(extent);
        // Resource creation done at this point
        self.updateDescriptorSets();

        return self;
    }

    pub fn deinit(self: Renderer) void {
        self.deinitRenderTargets();

        // The descriptor sets and command buffers do not need to be free'd, as they are freed when
        // their respective pool is destroyed.
        for (self.frame_resources.cmd_pools) |cmd_pool| {
            self.dev.vkd.destroyCommandPool(self.dev.handle, cmd_pool, null);
        }
        self.dev.vkd.destroyDescriptorPool(self.dev.handle, self.descriptor_pool, null);

        self.dev.vkd.destroyPipeline(self.dev.handle, self.pipeline, null);
        self.dev.vkd.destroyPipelineLayout(self.dev.handle, self.pipeline_layout, null);
        self.dev.vkd.destroyDescriptorSetLayout(self.dev.handle, self.descriptor_set_layout, null);

        for (self.frame_resources.frame_fences) |fence| {
            self.dev.vkd.destroyFence(self.dev.handle, fence, null);
        }
    }

    pub fn deinitRenderTargets(self: Renderer) void {
        for (self.frame_resources.render_target_views) |rtv| {
            self.dev.vkd.destroyImageView(self.dev.handle, rtv, null);
        }

        for (self.frame_resources.render_targets) |rt| {
            self.dev.vkd.destroyImage(self.dev.handle, rt, null);
        }

        self.dev.vkd.freeMemory(self.dev.handle, self.render_target_memory, null);
    }

    fn createFences(self: *Renderer) !void {
        for (self.frame_resources.frame_fences) |*fence, i| {
            fence.* = try self.dev.vkd.createFence(self.dev.handle, .{.flags = .{.signaled_bit = true}}, null);
        }
    }

    fn createPipeline(self: *Renderer) !void {
        self.descriptor_set_layout = try self.dev.vkd.createDescriptorSetLayout(self.dev.handle, .{
            .flags = .{},
            .binding_count = @intCast(u32, bindings.len),
            .p_bindings = &bindings,
        }, null);

        const pcr = vk.PushConstantRange{
            .stage_flags = .{.compute_bit = true},
            .offset = 0,
            .size = @sizeOf(PushConstantBuffer), 
        };

        self.pipeline_layout = try self.dev.vkd.createPipelineLayout(self.dev.handle, .{
            .flags = .{},
            .set_layout_count = 1,
            .p_set_layouts = asManyPtr(&self.descriptor_set_layout),
            .push_constant_range_count = 1,
            .p_push_constant_ranges = asManyPtr(&pcr),
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
            asManyPtr(&cpci),
            null,
            asManyPtr(&self.pipeline),
        );
    }

    fn createDescriptorSets(self: *Renderer) !void {
        var pool_sizes: [bindings.len]vk.DescriptorPoolSize = undefined;
        var n_pool_sizes: u32 = 0;

        for (bindings) |binding| {
            for (pool_sizes[0 .. n_pool_sizes]) |*pool_size| {
                if (pool_size.@"type" == binding.descriptor_type) {
                    pool_size.descriptor_count += binding.descriptor_count * max_frames_in_flight;
                    break;
                }
            } else {
                pool_sizes[n_pool_sizes] = .{
                    .@"type" = binding.descriptor_type,
                    .descriptor_count = binding.descriptor_count * max_frames_in_flight,
                };
                n_pool_sizes += 1;
            }
        }

        self.descriptor_pool = try self.dev.vkd.createDescriptorPool(self.dev.handle, .{
            .flags = .{},
            .max_sets = max_frames_in_flight,
            .pool_size_count = n_pool_sizes,
            .p_pool_sizes = &pool_sizes,
        }, null);

        var layouts: [max_frames_in_flight]vk.DescriptorSetLayout = undefined;
        for (layouts) |*layout| layout.* = self.descriptor_set_layout;

        try self.dev.vkd.allocateDescriptorSets(self.dev.handle, .{
            .descriptor_pool = self.descriptor_pool,
            .descriptor_set_count = @truncate(u32, layouts.len),
            .p_set_layouts = &layouts,
        }, &self.frame_resources.descriptor_sets);
    }

    fn createCommandBuffers(self: *Renderer) !void {
        for (self.frame_resources.cmd_pools) |*cmd_pool, i| {
            cmd_pool.* = try self.dev.vkd.createCommandPool(self.dev.handle, .{
                .flags = .{},
                .queue_family_index = self.dev.compute_queue.family,
            }, null);

            try self.dev.vkd.allocateCommandBuffers(self.dev.handle, .{
                .command_pool = cmd_pool.*,
                .level = .primary,
                .command_buffer_count = 1,
            }, asManyPtr(&self.frame_resources.cmd_bufs[i]));
        }
    }

    fn createRenderTargets(self: *Renderer, extent: vk.Extent2D) !void {
        const format = .r8g8b8a8_unorm; // Format always supported as storage image, see vk spec table 63.
        for (self.frame_resources.render_targets) |*rt| {
            rt.* = try self.dev.vkd.createImage(self.dev.handle, .{
                .flags = .{},
                .image_type = .@"2d",
                .format = format, 
                .extent = .{.width = extent.width, .height = extent.height, .depth = 1},
                .mip_levels = 1,
                .array_layers = 1,
                .samples = .{.@"1_bit" = true},
                .tiling = .optimal,
                .usage = .{.storage_bit = true, .transfer_src_bit = true, .transfer_dst_bit = true},
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
        var mem_reqs = self.dev.vkd.getImageMemoryRequirements(self.dev.handle, self.frame_resources.render_targets[0]);
        const adjusted_size = std.mem.alignForwardGeneric(vk.DeviceSize, mem_reqs.size, mem_reqs.alignment);
        mem_reqs.size = adjusted_size * max_frames_in_flight;

        self.render_target_memory = try self.dev.allocate(mem_reqs, .{.device_local_bit = true});

        for (self.frame_resources.render_targets) |rt, i| {
            try self.dev.vkd.bindImageMemory(self.dev.handle, rt, self.render_target_memory, adjusted_size * i);
        }

        for (self.frame_resources.render_target_views) |*rtv, i| {
            const rt = self.frame_resources.render_targets[i];

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
        for (self.frame_resources.descriptor_sets) |set, i| {
            const render_target_write = vk.DescriptorImageInfo{
                .sampler = .null_handle,
                .image_view = self.frame_resources.render_target_views[i],
                .image_layout = .general,
            };

            const writes = [_]vk.WriteDescriptorSet{
                .{
                    .dst_set = set,
                    .dst_binding = bindings[0].binding,
                    .dst_array_element = 0,
                    .descriptor_count = 1,
                    .descriptor_type = bindings[0].descriptor_type,
                    .p_image_info = asManyPtr(&render_target_write),
                    .p_buffer_info = undefined,
                    .p_texel_buffer_view = undefined,
                },
            };

            // Could do a single updateDescriptorSets, but that would require allocating an array of writes.
            self.dev.vkd.updateDescriptorSets(self.dev.handle, @intCast(u32, writes.len), &writes, 0, undefined);
        }
    }

    pub fn resize(self: *Renderer, extent: vk.Extent2D) !void {
        self.deinitRenderTargets();

        // Clear to null_handle to make deinit on error easier
        for (self.frame_resources.render_targets) |*rt| rt.* = .null_handle;
        for (self.frame_resources.render_target_views) |*rtv| rtv.* = .null_handle;

        try self.createRenderTargets(extent);
        self.updateDescriptorSets();
    }

    pub fn render(self: *Renderer, extent: vk.Extent2D, swapchain_image: vk.Image, cam: math.Camera) !FrameData {
        const index = self.frame_index;
        self.frame_index = (self.frame_index + 1) % max_frames_in_flight;

        const fence = self.frame_resources.frame_fences[index];
        const cmd_pool = self.frame_resources.cmd_pools[index];
        const cmd_buf = self.frame_resources.cmd_bufs[index];
        const render_target = self.frame_resources.render_targets[index];
        const descriptor_set = self.frame_resources.descriptor_sets[index];

        // Make sure the previous frame is finished rendering.
        _ = try self.dev.vkd.waitForFences(self.dev.handle, 1, asManyPtr(&fence), vk.TRUE, std.math.maxInt(u64));

        // Instead of having a single pool, we have a pool for each frame. This way if we allocate more than one
        // command buffer per frame, we can reset them all in one go. Furthermore, the command pool can use a
        // linear allocation scheme [citation needed]
        try self.dev.vkd.resetCommandPool(self.dev.handle, cmd_pool, .{});

        try self.dev.vkd.beginCommandBuffer(cmd_buf, .{
            .flags = .{.one_time_submit_bit = true},
            .p_inheritance_info = null,
        });

        const subresource_range = vk.ImageSubresourceRange{
            .aspect_mask = .{.color_bit = true},
            .base_mip_level = 0,
            .level_count = 1,
            .base_array_layer = 0,
            .layer_count = 1,
        };

        {
            const barriers = [_]vk.ImageMemoryBarrier{
                .{
                    .src_access_mask = .{},
                    .dst_access_mask = .{},
                    .old_layout = .@"undefined",
                    .new_layout = .general,
                    .src_queue_family_index = self.dev.compute_queue.family,
                    .dst_queue_family_index = self.dev.compute_queue.family,
                    .image = render_target,
                    .subresource_range = subresource_range,  
                },
            };
            self.dev.vkd.cmdPipelineBarrier(
                cmd_buf,
                .{.top_of_pipe_bit = true},
                .{.compute_shader_bit = true},
                .{},
                0, undefined,
                0, undefined,
                barriers.len, &barriers
            );
        }

        self.dev.vkd.cmdBindPipeline(cmd_buf, .compute, self.pipeline);
        self.dev.vkd.cmdBindDescriptorSets(
            cmd_buf,
            .compute,
            self.pipeline_layout,
            0,
            1,
            asManyPtr(&descriptor_set),
            0,
            undefined,
        );

        const push_constants = PushConstantBuffer{
            .forward = cam.rotation.forward().swizzle("xyz0"),
            .up = cam.rotation.up().swizzle("xyz0"),
            .translation = cam.translation.swizzle("xyz0"),
        };

        self.dev.vkd.cmdPushConstants(
            cmd_buf,
            self.pipeline_layout,
            .{.compute_bit = true},
            0,
            @sizeOf(PushConstantBuffer),
            @ptrCast(*const c_void, &push_constants),
        );

        self.dev.vkd.cmdDispatch(
            cmd_buf,
            (extent.width + workgroup_size.width - 1) / workgroup_size.width,
            (extent.height + workgroup_size.height - 1) / workgroup_size.height,
            1,
        );

        {
            const barriers = [_]vk.ImageMemoryBarrier{
                .{
                    .src_access_mask = .{},
                    .dst_access_mask = .{},
                    .old_layout = .general,
                    .new_layout = .transfer_src_optimal,
                    .src_queue_family_index = self.dev.compute_queue.family,
                    .dst_queue_family_index = self.dev.compute_queue.family,
                    .image = render_target,
                    .subresource_range = subresource_range,  
                },
                .{
                    .src_access_mask = .{},
                    .dst_access_mask = .{},
                    .old_layout = .@"undefined",
                    .new_layout = .transfer_dst_optimal,
                    .src_queue_family_index = self.dev.present_queue.family,
                    .dst_queue_family_index = self.dev.compute_queue.family,
                    .image = swapchain_image,
                    .subresource_range = subresource_range,
                }
            };
            self.dev.vkd.cmdPipelineBarrier(
                cmd_buf,
                .{.compute_shader_bit = true},
                .{.transfer_bit = true},
                .{},
                0, undefined,
                0, undefined,
                barriers.len, &barriers
            );
        }

        self.dev.vkd.cmdCopyImage(
            cmd_buf,
            render_target,
            .transfer_src_optimal,
            swapchain_image,
            .transfer_dst_optimal,
            1,
            asManyPtr(&vk.ImageCopy{
                .src_subresource = .{
                    .aspect_mask = .{.color_bit = true},
                    .mip_level = 0,
                    .base_array_layer = 0,
                    .layer_count = 1,
                },
                .src_offset = .{.x = 0, .y = 0, .z = 0},
                .dst_subresource = .{
                    .aspect_mask = .{.color_bit = true},
                    .mip_level = 0,
                    .base_array_layer = 0,
                    .layer_count = 1,
                },
                .dst_offset = .{.x = 0, .y = 0, .z = 0},
                .extent = .{.width = extent.width, .height = extent.height, .depth = 1},
            }),
        );

        {
            const barriers = [_]vk.ImageMemoryBarrier{
                .{
                    .src_access_mask = .{},
                    .dst_access_mask = .{},
                    .old_layout = .transfer_dst_optimal,
                    .new_layout = .present_src_khr,
                    .src_queue_family_index = self.dev.compute_queue.family,
                    .dst_queue_family_index = self.dev.present_queue.family,
                    .image = swapchain_image,
                    .subresource_range = subresource_range,  
                },
            };
            self.dev.vkd.cmdPipelineBarrier(
                cmd_buf,
                .{.transfer_bit = true},
                .{.bottom_of_pipe_bit = true},
                .{},
                0, undefined,
                0, undefined,
                barriers.len, &barriers
            );
        }

        try self.dev.vkd.endCommandBuffer(cmd_buf);

        try self.dev.vkd.resetFences(self.dev.handle, 1, asManyPtr(&fence));
        // If the fence is not submitted, it is not going to get signalled, so anything
        // that fails could potentially ruin the synchronization if that causes
        // the fence to not be submitted.

        return FrameData{
            .frame_fence = fence,
            .cmd_buf = cmd_buf,
        };
    }

    pub fn waitForAllFrames(self: Renderer) !void {
        _ = try self.dev.vkd.waitForFences(
            self.dev.handle,
            max_frames_in_flight,
            &self.frame_resources.frame_fences,
            vk.TRUE,
            std.math.maxInt(u64)
        );
    }
};
