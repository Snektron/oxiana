const std = @import("std");
const vk = @import("vulkan");
const graphics = @import("graphics.zig");
const Instance = graphics.Instance;
const Device = graphics.Device;
const Allocator = std.mem.Allocator;

pub const Swapchain = struct {
    pub const PresentState = enum {
        optimal,
        suboptimal
    };

    instance: *const Instance,
    dev: *const Device,
    allocator: *Allocator,
    surface: vk.SurfaceKHR,

    surface_format: vk.SurfaceFormatKHR,
    present_mode: vk.PresentModeKHR,
    extent: vk.Extent2D,
    handle: vk.SwapchainKHR,

    swap_images: []SwapImage,
    image_index: u32,
    next_image_acquired: vk.Semaphore,

    pub fn init(instance: *const Instance, dev: *const Device, allocator: *Allocator, extent: vk.Extent2D, surface: vk.SurfaceKHR) !Swapchain {
        var self = Swapchain{
            .instance = instance,
            .dev = dev,
            .allocator = allocator,
            .surface = surface,
            .surface_format = undefined,
            .present_mode = undefined,
            .extent = undefined,
            .handle = .null_handle,
            .swap_images = &[_]SwapImage{},
            .image_index = undefined,
            .next_image_acquired = try dev.vkd.createSemaphore(dev.handle, .{.flags = .{}}, null),
        };
        errdefer dev.vkd.destroySemaphore(dev.handle, self.next_image_acquired, null);

        try self.recreate(extent);
        return self;
    }

    pub fn deinit(self: Swapchain) void {
        for (self.swap_images) |si| si.deinit(self.dev);
        self.dev.vkd.destroySemaphore(self.dev.handle, self.next_image_acquired, null);
        self.dev.vkd.destroySwapchainKHR(self.dev.handle, self.handle, null);
    }

    // If this fails, the swapchain is in an undefined but invalid state: `deinit` and `recreate`
    // can still be called.
    fn recreate(self: *Swapchain, new_extent: vk.Extent2D) !void {
        const pdev = self.dev.pdev.handle;
        self.surface_format = try findSurfaceFormat(self.instance.vki, pdev, self.surface, self.allocator);
        self.present_mode = try findPresentMode(self.instance.vki, pdev, self.surface, self.allocator);
        const caps = try self.instance.vki.getPhysicalDeviceSurfaceCapabilitiesKHR(pdev, self.surface);
        self.extent = findActualExtent(caps, new_extent);

        if (self.extent.width == 0 or self.extent.height == 0) {
            return error.InvalidSurfaceDimensions;
        }

        var image_count = caps.min_image_count + 1;
        if (caps.max_image_count > 0) {
            image_count = std.math.min(image_count, caps.max_image_count);
        }

        const concurrent = self.dev.graphics_queue.family != self.dev.present_queue.family; // TODO: compute
        const qfi = [_]u32{self.dev.graphics_queue.family, self.dev.present_queue.family};

        const old_handle = self.handle;
        self.handle = try self.dev.vkd.createSwapchainKHR(self.dev.handle, .{
            .flags = .{},
            .surface = self.surface,
            .min_image_count = image_count,
            .image_format = self.surface_format.format,
            .image_color_space = self.surface_format.color_space,
            .image_extent = self.extent,
            .image_array_layers = 1,
            .image_usage = .{.color_attachment_bit = true, .transfer_dst_bit = true}, // TODO
            .image_sharing_mode = if (concurrent) .concurrent else .exclusive,
            .queue_family_index_count = qfi.len,
            .p_queue_family_indices = &qfi,
            .pre_transform = caps.current_transform,
            .composite_alpha = .{.opaque_bit_khr = true},
            .present_mode = self.present_mode,
            .clipped = vk.TRUE,
            .old_swapchain = self.handle,
        }, null);
        errdefer self.dev.vkd.destroySwapchainKHR(self.dev.handle, self.handle, null);

        if (old_handle != .null_handle) {
            self.dev.vkd.destroySwapchainKHR(self.dev.handle, old_handle, null);
        }

        try self.fetchSwapImages();

        const result = try self.dev.vkd.acquireNextImageKHR(
            self.dev.handle,
            self.handle,
            std.math.maxInt(u64),
            self.next_image_acquired,
            .null_handle
        );

        if (result.result != .success) {
            return error.ImageAcquireFailed;
        }

        std.mem.swap(vk.Semaphore, &self.swap_images[result.image_index].image_acquired, &self.next_image_acquired);
        self.image_index = result.image_index;
    }

    fn fetchSwapImages(self: *Swapchain) !void {
        var count: u32 = undefined;
        _ = try self.dev.vkd.getSwapchainImagesKHR(self.dev.handle, self.handle, &count, null);
        const images = try self.allocator.alloc(vk.Image, count);
        defer self.allocator.free(images);
        _ = try self.dev.vkd.getSwapchainImagesKHR(self.dev.handle, self.handle, &count, images.ptr);

        // Deinit old swap images, if any
        for (self.swap_images) |si| si.deinit(self.dev);

        if (count != self.swap_images.len) {
            self.swap_images = try self.allocator.realloc(self.swap_images, count);
        }

        var i: usize = 0;
        errdefer {
            for (self.swap_images[0 .. i]) |si| si.deinit(self.dev);

            // Free the swap images to prevent double deinitialization of above swap images.
            self.swap_images = self.allocator.shrink(self.swap_images, 0);
        }

        for (images) |image| {
            self.swap_images[i] = try SwapImage.init(self.dev, image, self.surface_format.format);
            i += 1;
        }
    }

    pub fn waitForAllFences(self: Swapchain) !void {
        for (self.swap_images) |si| si.waitForFence(self.dev) catch {};
    }

    pub fn currentImage(self: Swapchain) vk.Image {
        return self.swap_images[self.image_index].image;
    }

    pub fn currentSwapImage(self: Swapchain) *const SwapImage {
        return &self.swap_images[self.image_index];
    }

    pub fn present(self: *Swapchain) !PresentState {
        // Simple method:
        // 1) Acquire next image
        // 2) Wait for and reset fence of the acquired image
        // 3) Submit command buffer with fence of acquired image,
        //    dependendent on the semaphore signalled by the first step.
        // 4) Present current frame, dependent on semaphore signalled by previous step
        // Problem: This way we can't reference the current image while rendering.
        // Better method: Shuffle the steps around such that acquire next image is the last step,
        // leaving the swapchain in a state with the current image.
        // 1) Wait for and reset fence of current image
        // 2) Submit command buffer, signalling fence of current image and dependent on
        //    the semaphore signalled by step 4.
        // 3) Present current frame, dependent on semaphore signalled by the submit
        // 4) Acquire next image, signalling its semaphore
        // One problem that arises is that we can't know beforehand which semaphore to signal,
        // so we keep an extra auxilery semaphore that is swapped around

        // Step 1: Make sure the current frame has finished rendering
        const current = self.currentSwapImage();
        try current.waitForFence(self.dev);
        try self.dev.vkd.resetFences(self.dev.handle, 1, @ptrCast([*]const vk.Fence, &current.frame_fence));

        // Step 2: Submit the command buffer
        // TODO: Move somewhere else
        const wait_stage = [_]vk.PipelineStageFlags{.{.top_of_pipe_bit = true}};
        try self.dev.vkd.queueSubmit(self.dev.graphics_queue.handle, 1, &[_]vk.SubmitInfo{.{
            .wait_semaphore_count = 1,
            .p_wait_semaphores = @ptrCast([*]const vk.Semaphore, &current.image_acquired),
            .p_wait_dst_stage_mask = &wait_stage,
            .command_buffer_count = 1,
            .p_command_buffers = @ptrCast([*]const vk.CommandBuffer, &cmdbuf),
            .signal_semaphore_count = 1,
            .p_signal_semaphores = @ptrCast([*]const vk.Semaphore, &current.render_finished),
        }}, current.frame_fence);

        // Step 3: Present the current frame
        _ = try self.dev.vkd.queuePresentKHR(self.dev.present_queue.handle, .{
            .wait_semaphore_count = 1,
            .p_wait_semaphores = @ptrCast([*]const vk.Semaphore, &current.render_finished),
            .swapchain_count = 1,
            .p_swapchains = @ptrCast([*]const vk.SwapchainKHR, &self.handle),
            .p_image_indices = @ptrCast([*]const u32, &self.image_index),
            .p_results = null,
        });

        // Step 4: Acquire next frame
        const result = try self.dev.vkd.acquireNextImageKHR(
            self.dev.handle,
            self.handle,
            std.math.maxInt(u64),
            self.next_image_acquired,
            .null_handle,
        );

        std.mem.swap(vk.Semaphore, &self.swap_images[result.image_index].image_acquired, &self.next_image_acquired);
        self.image_index = result.image_index;

        return switch (result.result) {
            .success => .optimal,
            .suboptimal_khr => .suboptimal,
            else => unreachable,
        };
    }
};

fn findSurfaceFormat(vki: graphics.InstanceDispatch, pdev: vk.PhysicalDevice, surface: vk.SurfaceKHR, allocator: *Allocator) !vk.SurfaceFormatKHR {
    var count: u32 = undefined;
    _ = try vki.getPhysicalDeviceSurfaceFormatsKHR(pdev, surface, &count, null);
    const surface_formats = try allocator.alloc(vk.SurfaceFormatKHR, count);
    defer allocator.free(surface_formats);
    _ = try vki.getPhysicalDeviceSurfaceFormatsKHR(pdev, surface, &count, surface_formats.ptr);

    const preferred = vk.SurfaceFormatKHR{
        .format = .b8g8r8a8_srgb,
        .color_space = .srgb_nonlinear_khr,
    };
    for (surface_formats) |sfmt| {
        if (std.meta.eql(sfmt, preferred)) {
            return preferred;
        }
    }

    return surface_formats[0]; // There must always be at least one supported surface format
}

fn findPresentMode(vki: graphics.InstanceDispatch, pdev: vk.PhysicalDevice, surface: vk.SurfaceKHR, allocator: *Allocator) !vk.PresentModeKHR {
    var count: u32 = undefined;
    _ = try vki.getPhysicalDeviceSurfacePresentModesKHR(pdev, surface, &count, null);
    const present_modes = try allocator.alloc(vk.PresentModeKHR, count);
    defer allocator.free(present_modes);
    _ = try vki.getPhysicalDeviceSurfacePresentModesKHR(pdev, surface, &count, present_modes.ptr);

    const preferred = [_]vk.PresentModeKHR{
        .mailbox_khr,
        .immediate_khr,
    };

    for (preferred) |mode| {
        if (std.mem.indexOfScalar(vk.PresentModeKHR, present_modes, mode) != null) {
            return mode;
        }
    }

    return .fifo_khr;
}

fn findActualExtent(caps: vk.SurfaceCapabilitiesKHR, extent: vk.Extent2D) vk.Extent2D {
    if (caps.current_extent.width != 0xFFFF_FFFF) {
        return caps.current_extent;
    } else {
        return .{
            .width = std.math.clamp(extent.width, caps.min_image_extent.width, caps.max_image_extent.width),
            .height = std.math.clamp(extent.height, caps.min_image_extent.height, caps.max_image_extent.height),
        };
    }
}

const SwapImage = struct {
    image: vk.Image,
    view: vk.ImageView,
    image_acquired: vk.Semaphore,
    render_finished: vk.Semaphore,
    frame_fence: vk.Fence,

    fn init(dev: *const Device, image: vk.Image, format: vk.Format) !SwapImage {
        const view = try dev.vkd.createImageView(dev.handle, .{
            .flags = .{},
            .image = image,
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
        errdefer dev.vkd.destroyImageView(dev.handle, view, null);

        const image_acquired = try dev.vkd.createSemaphore(dev.handle, .{.flags = .{}}, null);
        errdefer dev.vkd.destroySemaphore(dev.handle, image_acquired, null);

        const render_finished = try dev.vkd.createSemaphore(dev.handle, .{.flags = .{}}, null);
        errdefer dev.vkd.destroySemaphore(dev.handle, image_acquired, null);

        const frame_fence = try dev.vkd.createFence(dev.handle, .{.flags = .{.signaled_bit = true}}, null);
        errdefer dev.vkd.destroyFence(dev.handle, frame_fence, null);

        return SwapImage{
            .image = image,
            .view = view,
            .image_acquired = image_acquired,
            .render_finished = render_finished,
            .frame_fence = frame_fence,
        };
    }

    fn deinit(self: SwapImage, dev: *const Device) void {
        self.waitForFence(dev) catch return;
        dev.vkd.destroyImageView(dev.handle, self.view, null);
        dev.vkd.destroySemaphore(dev.handle, self.image_acquired, null);
        dev.vkd.destroySemaphore(dev.handle, self.render_finished, null);
        dev.vkd.destroyFence(dev.handle, self.frame_fence, null);
    }

    fn waitForFence(self: SwapImage, dev: *const Device) !void {
        _ = try dev.vkd.waitForFences(dev.handle, 1, @ptrCast([*]const vk.Fence, &self.frame_fence), vk.TRUE, std.math.maxInt(u64));
    }
};
