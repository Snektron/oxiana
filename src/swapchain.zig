const std = @import("std");
const vk = @import("vulkan");
const gfx = @import("graphics.zig");
const StructOfArrays = @import("soa.zig").StructOfArrays;
const Instance = gfx.Instance;
const Device = gfx.Device;
const Allocator = std.mem.Allocator;

pub const Swapchain = struct {
    pub const PresentState = enum {
        optimal,
        suboptimal
    };

    pub const CreateInfo = struct {
        surface: vk.SurfaceKHR,
        swap_image_usage: vk.ImageUsageFlags,
        vsync: bool = false,
        format_features: vk.FormatFeatureFlags = .{},
    };

    const SwapImage = struct {
        image: vk.Image,
        image_acquired: vk.Semaphore,
        render_finished: vk.Semaphore,
    };

    const SwapImageArray = StructOfArrays(SwapImage);

    instance: *const Instance,
    dev: *const Device,
    allocator: *Allocator,

    surface_format: vk.SurfaceFormatKHR,
    present_mode: vk.PresentModeKHR,
    extent: vk.Extent2D,
    handle: vk.SwapchainKHR,

    swap_images: SwapImageArray,

    image_index: u32,
    next_image_acquired: vk.Semaphore,

    pub fn init(instance: *const Instance, dev: *const Device, allocator: *Allocator, extent: vk.Extent2D, create_info: CreateInfo) !Swapchain {
        var self = Swapchain{
            .instance = instance,
            .dev = dev,
            .allocator = allocator,
            .surface_format = undefined,
            .present_mode = undefined,
            .extent = undefined,
            .handle = .null_handle,
            .swap_images = SwapImageArray.empty(allocator),
            .image_index = undefined,
            .next_image_acquired = try dev.vkd.createSemaphore(dev.handle, .{.flags = .{}}, null),
        };
        errdefer dev.vkd.destroySemaphore(dev.handle, self.next_image_acquired, null);

        try self.recreate(extent, create_info);
        return self;
    }

    pub fn deinit(self: Swapchain) void {
        self.deinitSwapImageArray();
        self.dev.vkd.destroySemaphore(self.dev.handle, self.next_image_acquired, null);
        self.dev.vkd.destroySwapchainKHR(self.dev.handle, self.handle, null);
    }

    // If this fails, the swapchain is in an undefined but invalid state: `deinit` and `recreate`
    // can still be called.
    pub fn recreate(self: *Swapchain, new_extent: vk.Extent2D, create_info: CreateInfo) !void {
        const pdev = self.dev.pdev.handle;
        self.surface_format = try findSurfaceFormat(self.instance.vki, pdev, create_info, self.allocator);
        self.present_mode = try findPresentMode(self.instance.vki, pdev, create_info.surface, self.allocator);

        // There seems to be a bug in the validation layers that causes a message to be printed about
        // fifo being unsupported if getPhysicalDeviceSurfacePresentModesKHR is not called. To work around
        // this for now, just override the value after calling that function. (It needs to be called with a
        // a valid pointer as well it seems).
        if (create_info.vsync) {
            self.present_mode = .fifo_khr;
        }

        const caps = try self.instance.vki.getPhysicalDeviceSurfaceCapabilitiesKHR(pdev, create_info.surface);
        self.extent = findActualExtent(caps, new_extent);

        if (self.extent.width == 0 or self.extent.height == 0) {
            return error.InvalidSurfaceDimensions;
        }

        if (!caps.supported_usage_flags.contains(create_info.swap_image_usage)) {
            return error.UnsupportedSwapImageUsage;
        }

        var image_count = caps.min_image_count + 1;
        if (caps.max_image_count > 0) {
            image_count = std.math.min(image_count, caps.max_image_count);
        }

        var qfi: [3]u32 = undefined;
        var n_concurrent_queues: u32 = 0;

        for ([_]u32{self.dev.graphics_queue.family, self.dev.compute_queue.family, self.dev.present_queue.family}) |fam| {
            if (std.mem.indexOfScalar(u32, qfi[0 .. n_concurrent_queues], fam) == null) {
                qfi[n_concurrent_queues] = fam;
                n_concurrent_queues += 1;
            }
        }

        const old_handle = self.handle;
        self.handle = try self.dev.vkd.createSwapchainKHR(self.dev.handle, .{
            .flags = .{},
            .surface = create_info.surface,
            .min_image_count = image_count,
            .image_format = self.surface_format.format,
            .image_color_space = self.surface_format.color_space,
            .image_extent = self.extent,
            .image_array_layers = 1,
            .image_usage = create_info.swap_image_usage,
            .image_sharing_mode = if (n_concurrent_queues > 1) .concurrent else .exclusive,
            .queue_family_index_count = n_concurrent_queues,
            .p_queue_family_indices = &qfi,
            .pre_transform = caps.current_transform,
            .composite_alpha = .{.opaque_bit_khr = true},
            .present_mode = self.present_mode,
            .clipped = vk.TRUE,
            .old_swapchain = self.handle,
        }, null);
        errdefer self.dev.vkd.destroySwapchainKHR(self.dev.handle, self.handle, null);
        
        // Destroy the handle *after* acquiring the first frame, the give the presentation engine the opportunity
        // to finish presenting to the old frames. It's technically valid to nuke the swapchain at any point,
        // but this chould be a little more efficient.
        defer if (old_handle != .null_handle) {
            self.dev.vkd.destroySwapchainKHR(self.dev.handle, old_handle, null);
        };

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

        std.mem.swap(vk.Semaphore, self.swap_images.at("image_acquired", result.image_index), &self.next_image_acquired);
        self.image_index = result.image_index;
    }

    fn fetchSwapImages(self: *Swapchain) !void {
        var count: u32 = undefined;
        _ = try self.dev.vkd.getSwapchainImagesKHR(self.dev.handle, self.handle, &count, null);

        if (self.swap_images.len != count) {
            // Play it safe for now and reinitialize everything - this is not likely to happen very often
            self.deinitSwapImageArray();
            self.swap_images.realloc(count) catch |err| {
                // The items of the swap image array are already freed, if we were to simply
                // return the error now, they would be free'd again in the deinit function, so simply free
                // the swap images and set the size to 0.
                self.swap_images.shrink(0);
                return err;
            };
            try self.initSwapImageArray();
        }

        _ = try self.dev.vkd.getSwapchainImagesKHR(self.dev.handle, self.handle, &count, self.swap_images.slice("image").ptr);
    }

    pub fn acquireNextSwapImage(self: Swapchain) !SwapImage {
        return SwapImage{
            .image = self.swap_images.at("image", self.image_index).*,
            .image_acquired = self.swap_images.at("image_acquired", self.image_index).*,
            .render_finished = self.swap_images.at("render_finished", self.image_index).*,
        };
    }

    pub fn swapBuffers(self: *Swapchain) !PresentState {
        const render_finished = self.swap_images.at("render_finished", self.image_index).*;

        _ = try self.dev.vkd.queuePresentKHR(self.dev.present_queue.handle, .{
            .wait_semaphore_count = 1,
            .p_wait_semaphores = @ptrCast([*]const vk.Semaphore, &render_finished),
            .swapchain_count = 1,
            .p_swapchains = @ptrCast([*]const vk.SwapchainKHR, &self.handle),
            .p_image_indices = @ptrCast([*]const u32, &self.image_index),
            .p_results = null,
        });

        const result = try self.dev.vkd.acquireNextImageKHR(
            self.dev.handle,
            self.handle,
            std.math.maxInt(u64),
            self.next_image_acquired,
            .null_handle,
        );

        std.mem.swap(vk.Semaphore, self.swap_images.at("image_acquired", result.image_index), &self.next_image_acquired);
        self.image_index = result.image_index;

        return switch (result.result) {
            .success => .optimal,
            .suboptimal_khr => .suboptimal,
            else => unreachable,
        };   
    }

    fn createSwapImageView(self: Swapchain, image_index: u32) !vk.ImageView {
        return try self.dev.vkd.createImageView(self.dev.handle, .{
            .flags = .{},
            .image = self.swap_images.at("image", i).*,
            .view_type = .@"2d",
            .format = self.format,
            .components = .{.r = .identity, .g = .identity, .b = .identity, .a = .identity},
            .subresource_range = .{
                .aspect_mask = .{.color_bit = true},
                .base_mip_level = 0,
                .level_count = 1,
                .base_array_layer = 0,
                .layer_count = 1, // SwapchainCreateInfo.image_array_layers
            },
        }, null);
    }

    fn initSwapImageArray(self: *Swapchain) !void {
        var i: usize = 0;
        errdefer while (i > 0) {
            i -= 1;
            self.deinitSwapImage(i);
        };

        while (i < self.swap_images.len) : (i += 1) {
            try self.initSwapImage(i);
        }
    }

    fn initSwapImage(self: *Swapchain, index: usize) !void {
        const image_acquired = self.swap_images.at("image_acquired", index);
        image_acquired.* = try self.dev.vkd.createSemaphore(self.dev.handle, .{.flags = .{}}, null);
        errdefer self.dev.vkd.destroySemaphore(self.dev.handle, image_acquired.*, null);

        const render_finished = self.swap_images.at("render_finished", index);
        render_finished.* = try self.dev.vkd.createSemaphore(self.dev.handle, .{.flags = .{}}, null);
        errdefer self.dev.vkd.destroySemaphore(self.dev.handle, render_finished.*, null);
    }

    fn deinitSwapImage(self: Swapchain, index: usize) void {
        const image_acquired = self.swap_images.at("image_acquired", index);
        self.dev.vkd.destroySemaphore(self.dev.handle, image_acquired.*, null);

        const render_finished = self.swap_images.at("render_finished", index);
        self.dev.vkd.destroySemaphore(self.dev.handle, render_finished.*, null);
    }

    fn deinitSwapImageArray(self: Swapchain) void {
        var i: usize = 0;
        while (i < self.swap_images.len) : (i += 1) {
            self.deinitSwapImage(i);
        }
    }
};

fn findSurfaceFormat(vki: gfx.InstanceDispatch, pdev: vk.PhysicalDevice, create_info: Swapchain.CreateInfo, allocator: *Allocator) !vk.SurfaceFormatKHR {
    var count: u32 = undefined;
    _ = try vki.getPhysicalDeviceSurfaceFormatsKHR(pdev, create_info.surface, &count, null);
    const surface_formats = try allocator.alloc(vk.SurfaceFormatKHR, count);
    defer allocator.free(surface_formats);
    _ = try vki.getPhysicalDeviceSurfaceFormatsKHR(pdev, create_info.surface, &count, surface_formats.ptr);

    // According to https://www.khronos.org/registry/vulkan/specs/1.2-extensions/html/vkspec.html#VUID-VkImageViewCreateInfo-usage-02274
    const required_format_features = create_info.format_features.merge(.{
        .sampled_image_bit = create_info.swap_image_usage.sampled_bit,
        .storage_image_bit = create_info.swap_image_usage.storage_bit,
        .color_attachment_bit = create_info.swap_image_usage.color_attachment_bit,
        .depth_stencil_attachment_bit = create_info.swap_image_usage.depth_stencil_attachment_bit,
        .transfer_src_bit = create_info.swap_image_usage.transfer_src_bit,
        .transfer_dst_bit = create_info.swap_image_usage.transfer_dst_bit,
    });

    const preferred = vk.SurfaceFormatKHR{
        .format = .b8g8r8a8_srgb,
        .color_space = .srgb_nonlinear_khr,
    };
    var surface_format: ?vk.SurfaceFormatKHR = null;

    for (surface_formats) |sfmt| {
        const fprops = vki.getPhysicalDeviceFormatProperties(pdev, sfmt.format);
        // According to the spec, swapchain images are always created with optimal tiling.
        const tiling_features = fprops.optimal_tiling_features;
        if (!tiling_features.contains(required_format_features)) {
            continue;
        }

        if (std.meta.eql(sfmt, preferred)) {
            return preferred;
        } else if (surface_format == null) {
            surface_format = sfmt;
        }
    }

    return surface_format orelse error.NoSurfaceFormat;
}

fn findPresentMode(vki: gfx.InstanceDispatch, pdev: vk.PhysicalDevice, surface: vk.SurfaceKHR, allocator: *Allocator) !vk.PresentModeKHR {
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
