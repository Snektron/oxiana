const std = @import("std");
const vk = @import("vulkan");
const mem = std.mem;
const Allocator = mem.Allocator;

const BaseDispatch = struct {
    vkCreateInstance: vk.PfnCreateInstance,

    usingnamespace vk.BaseWrapper(@This());
};

pub const InstanceDispatch = struct {
    vkDestroyInstance: vk.PfnDestroyInstance,
    vkEnumeratePhysicalDevices: vk.PfnEnumeratePhysicalDevices,
    vkDestroySurfaceKHR: vk.PfnDestroySurfaceKHR,
    vkGetPhysicalDeviceQueueFamilyProperties: vk.PfnGetPhysicalDeviceQueueFamilyProperties,
    vkGetPhysicalDeviceProperties: vk.PfnGetPhysicalDeviceProperties,
    vkGetPhysicalDeviceFormatProperties: vk.PfnGetPhysicalDeviceFormatProperties,
    vkGetPhysicalDeviceMemoryProperties: vk.PfnGetPhysicalDeviceMemoryProperties,
    vkGetPhysicalDeviceSurfaceFormatsKHR: vk.PfnGetPhysicalDeviceSurfaceFormatsKHR,
    vkGetPhysicalDeviceSurfacePresentModesKHR: vk.PfnGetPhysicalDeviceSurfacePresentModesKHR,
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR: vk.PfnGetPhysicalDeviceSurfaceCapabilitiesKHR,
    vkGetPhysicalDeviceSurfaceSupportKHR: vk.PfnGetPhysicalDeviceSurfaceSupportKHR,
    vkEnumerateDeviceExtensionProperties: vk.PfnEnumerateDeviceExtensionProperties,
    vkGetDeviceProcAddr: vk.PfnGetDeviceProcAddr,
    vkCreateDevice: vk.PfnCreateDevice,

    usingnamespace vk.InstanceWrapper(@This());
};

pub const DeviceDispatch = struct {
    vkDestroyDevice: vk.PfnDestroyDevice,
    vkGetDeviceQueue: vk.PfnGetDeviceQueue,
    vkQueueSubmit: vk.PfnQueueSubmit,
    vkQueuePresentKHR: vk.PfnQueuePresentKHR,
    vkCreateSwapchainKHR: vk.PfnCreateSwapchainKHR,
    vkGetSwapchainImagesKHR: vk.PfnGetSwapchainImagesKHR,
    vkAcquireNextImageKHR: vk.PfnAcquireNextImageKHR,
    vkDestroySwapchainKHR: vk.PfnDestroySwapchainKHR,
    vkCreateImageView: vk.PfnCreateImageView,
    vkDestroyImageView: vk.PfnDestroyImageView,
    vkCreateSemaphore: vk.PfnCreateSemaphore,
    vkDestroySemaphore: vk.PfnDestroySemaphore,
    vkCreateFence: vk.PfnCreateFence,
    vkDestroyFence: vk.PfnDestroyFence,
    vkWaitForFences: vk.PfnWaitForFences,
    vkResetFences: vk.PfnResetFences,
    vkCreatePipelineLayout: vk.PfnCreatePipelineLayout,
    vkDestroyPipelineLayout: vk.PfnDestroyPipelineLayout,
    vkCreateCommandPool: vk.PfnCreateCommandPool,
    vkDestroyCommandPool: vk.PfnDestroyCommandPool,
    vkResetCommandPool: vk.PfnResetCommandPool,
    vkAllocateCommandBuffers: vk.PfnAllocateCommandBuffers,
    vkFreeCommandBuffers: vk.PfnFreeCommandBuffers,
    vkCreateDescriptorSetLayout: vk.PfnCreateDescriptorSetLayout,
    vkDestroyDescriptorSetLayout: vk.PfnDestroyDescriptorSetLayout,
    vkCreateShaderModule: vk.PfnCreateShaderModule,
    vkDestroyShaderModule: vk.PfnDestroyShaderModule,
    vkCreateComputePipelines: vk.PfnCreateComputePipelines,
    vkDestroyPipeline: vk.PfnDestroyPipeline,
    vkCreateDescriptorPool: vk.PfnCreateDescriptorPool,
    vkDestroyDescriptorPool: vk.PfnDestroyDescriptorPool,
    vkAllocateDescriptorSets: vk.PfnAllocateDescriptorSets,
    vkUpdateDescriptorSets: vk.PfnUpdateDescriptorSets,
    vkCreateImage: vk.PfnCreateImage,
    vkDestroyImage: vk.PfnDestroyImage,
    vkGetImageMemoryRequirements: vk.PfnGetImageMemoryRequirements,
    vkBindImageMemory: vk.PfnBindImageMemory,
    vkAllocateMemory: vk.PfnAllocateMemory,
    vkFreeMemory: vk.PfnFreeMemory,
    vkBeginCommandBuffer: vk.PfnBeginCommandBuffer,
    vkEndCommandBuffer: vk.PfnEndCommandBuffer,
    vkCmdPipelineBarrier: vk.PfnCmdPipelineBarrier,
    vkCmdClearColorImage: vk.PfnCmdClearColorImage,
    vkCmdCopyImage: vk.PfnCmdCopyImage,
    vkCmdBindPipeline: vk.PfnCmdBindPipeline,
    vkCmdBindDescriptorSets: vk.PfnCmdBindDescriptorSets,
    vkCmdDispatch: vk.PfnCmdDispatch,
    vkCmdPushConstants: vk.PfnCmdPushConstants,

    usingnamespace vk.DeviceWrapper(@This());
};

pub const Instance = struct {
    vki: InstanceDispatch,
    handle: vk.Instance,

    pub fn init(
        loader: vk.PfnGetInstanceProcAddr,
        inst_exts: []const [*:0]const u8,
        app_info: vk.ApplicationInfo,
    ) !Instance {
        const vkb = try BaseDispatch.load(loader);

        const handle = try vkb.createInstance(.{
            .flags = .{},
            .p_application_info = &app_info,
            .enabled_layer_count = 0,
            .pp_enabled_layer_names = undefined,
            .enabled_extension_count = @intCast(u32, inst_exts.len),
            .pp_enabled_extension_names = inst_exts.ptr,
        }, null);

        return Instance{
            // Instance is leaked if the following fails.
            .vki = try InstanceDispatch.load(handle, loader),
            .handle = handle,
        };
    }

    pub fn deinit(self: Instance) void {
        self.vki.destroyInstance(self.handle, null);
    }

    pub fn physicalDeviceInfos(self: Instance, allocator: *Allocator) ![]PhysicalDeviceInfo {
        var device_count: u32 = undefined;
        _ = try self.vki.enumeratePhysicalDevices(self.handle, &device_count, null);
        const pdevs = try allocator.alloc(vk.PhysicalDevice, device_count);
        defer allocator.free(pdevs);
        _ = try self.vki.enumeratePhysicalDevices(self.handle, &device_count, pdevs.ptr);

        const pdev_infos = try allocator.alloc(PhysicalDeviceInfo, device_count);
        errdefer allocator.free(pdev_infos);

        for (pdev_infos) |*info, i| {
            info.* = PhysicalDeviceInfo.init(self.vki, pdevs[i]);
        }

        return pdev_infos;
    }

    pub fn findAndCreateDevice(self: Instance, allocator: *Allocator, surface: vk.SurfaceKHR, extensions: []const [*:0]const u8) !Device {
        const pdev_infos = try self.physicalDeviceInfos(allocator);
        defer allocator.free(pdev_infos);

        var tmp_extensions = try allocator.alloc([*:0]const u8, extensions.len);
        defer allocator.free(tmp_extensions);

        for (pdev_infos) |pdev| {
            mem.copy([*:0]const u8, tmp_extensions, extensions);
            if (!(try pdev.supportsSurface(self.vki, surface))) {
                std.log.info(.graphics, "Cannot use device '{}': Surface not supported", .{pdev.name()});
                continue;
            }

            var unsupported_extensions = try pdev.filterUnsupportedExtensions(self.vki, allocator, tmp_extensions);
            if (unsupported_extensions.len > 0) {
                std.log.info(.graphics, "Cannot use device '{}': {} required extension(s) not supported", .{
                    pdev.name(),
                    unsupported_extensions.len,
                });
                for (unsupported_extensions) |ext| {
                    std.log.info(.graphics, "- Extension '{}' is not supported", .{mem.spanZ(ext)});
                }
                continue;
            }

            const queues = pdev.allocateQueues(self.vki, surface, allocator) catch |err| {
                const message = switch (err) {
                    error.NoGraphicsQueue => "No graphics queue",
                    error.NoComputeQueue => "No compute queue",
                    error.NoPresentQueue => "No present queue",
                    else => |narrow| return narrow,
                };

                std.log.info(.graphics, "Cannot use device '{}': {}", .{pdev.name(), message});
                continue;
            };
            defer queues.deinit(allocator);

            // Just pick the first available suitable device
            return try Device.init(self, pdev, extensions, queues);
        }

        return error.NoSuitableDevice;
    }
};

pub const QueueAllocation = struct {
    family_propsv: []vk.QueueFamilyProperties,
    graphics_family: u32,
    compute_family: u32,
    present_family: u32,

    fn deinit(self: QueueAllocation, allocator: *Allocator) void {
        allocator.free(self.family_propsv);
    }
};

pub const PhysicalDeviceInfo = struct {
    handle: vk.PhysicalDevice,
    props: vk.PhysicalDeviceProperties,
    mem_props: vk.PhysicalDeviceMemoryProperties,

    fn init(vki: InstanceDispatch, handle: vk.PhysicalDevice) PhysicalDeviceInfo {
        return PhysicalDeviceInfo{
            .handle = handle,
            .props = vki.getPhysicalDeviceProperties(handle),
            .mem_props = vki.getPhysicalDeviceMemoryProperties(handle),
        };
    }

    pub fn name(self: PhysicalDeviceInfo) []const u8 {
        const len = std.mem.indexOfScalar(u8, &self.props.device_name, 0).?;
        return self.props.device_name[0 .. len];
    }

    pub fn supportsSurface(self: PhysicalDeviceInfo, vki: InstanceDispatch, surface: vk.SurfaceKHR) !bool {
        var format_count: u32 = undefined;
        _ = try vki.getPhysicalDeviceSurfaceFormatsKHR(self.handle, surface, &format_count, null);

        var present_mode_count: u32 = undefined;
        _ = try vki.getPhysicalDeviceSurfacePresentModesKHR(self.handle, surface, &present_mode_count, null);

        return format_count > 0 and present_mode_count > 0;
    }

    // Filters out any extension that *is* supported. The returned array is a modified subslice of `extensions`.
    pub fn filterUnsupportedExtensions(
        self: PhysicalDeviceInfo,
        vki: InstanceDispatch,
        allocator: *Allocator,
        extensions: [][*:0]const u8,
    ) ![][*:0]const u8 {
        var count: u32 = undefined;
        _ = try vki.enumerateDeviceExtensionProperties(self.handle, null, &count, null);

        const propsv = try allocator.alloc(vk.ExtensionProperties, count);
        defer allocator.free(propsv);

        _ = try vki.enumerateDeviceExtensionProperties(self.handle, null, &count, propsv.ptr);

        var write_index: usize = 0;
        for (extensions) |ext_z| {
            const ext = mem.spanZ(ext_z);
            for (propsv) |props| {
                const len = std.mem.indexOfScalar(u8, &props.extension_name, 0).?;
                const prop_ext = props.extension_name[0 .. len];
                if (std.mem.eql(u8, ext, prop_ext)) {
                    break;
                }
            } else {
                extensions[write_index] = ext;
                write_index += 1;
            }
        }

        return extensions[0 .. write_index];
    }

    pub fn allocateQueues(self: PhysicalDeviceInfo, vki: InstanceDispatch, surface: vk.SurfaceKHR, allocator: *Allocator) !QueueAllocation {
        var family_count: u32 = undefined;
        vki.getPhysicalDeviceQueueFamilyProperties(self.handle, &family_count, null);

        var propsv = try allocator.alloc(vk.QueueFamilyProperties, family_count);
        errdefer allocator.free(propsv);
        vki.getPhysicalDeviceQueueFamilyProperties(self.handle, &family_count, propsv.ptr);

        var graphics_family = blk: {
            for (propsv) |props, family| {
                if (props.queue_flags.contains(.{.graphics_bit = true})) {
                    break :blk @truncate(u32, family);
                }
            }

            return error.NoGraphicsQueue;
        };

        var compute_family = blk: {
            for (propsv) |props, family| {
                if (family != graphics_family and props.queue_flags.contains(.{.compute_bit = true})) {
                    break :blk @truncate(u32, family);
                }
            }

            // Pick the graphics family if there is no separate family
            if (propsv[graphics_family].queue_flags.contains(.{.compute_bit = true})) {
                break :blk graphics_family;
            }

            return error.NoComputeQueue;
        };

        var present_family = blk: {
            for (propsv) |_, i| {
                const family = @truncate(u32, i);
                // Skip these for now, and only consider them if there is no separate family.
                if (family == graphics_family or family == compute_family) {
                    continue;
                }

                if ((try vki.getPhysicalDeviceSurfaceSupportKHR(self.handle, family, surface)) == vk.TRUE) {
                    break :blk family;
                }
            }

            // If there is no separate family, prefer the one with the most queues
            const shared_family_indices = if (propsv[graphics_family].queue_count < propsv[compute_family].queue_count)
                    [_]u32{compute_family, graphics_family}
                else
                    [_]u32{graphics_family, compute_family};

            for (shared_family_indices) |family| {
                if ((try vki.getPhysicalDeviceSurfaceSupportKHR(self.handle, family, surface)) == vk.TRUE) {
                    break :blk family;
                }
            }

            return error.NoPresentQueue;
        };

        return QueueAllocation{
            .family_propsv = propsv,
            .graphics_family = graphics_family,
            .present_family = present_family,
            .compute_family = compute_family,
        };
    }

    pub fn findMemoryTypeIndex(self: PhysicalDeviceInfo, memory_type_bits: u32, flags: vk.MemoryPropertyFlags) !u32 {
        for (self.mem_props.memory_types[0 .. self.mem_props.memory_type_count]) |mem_type, i| {
            if (memory_type_bits & (@as(u32, 1) << @truncate(u5, i)) != 0 and mem_type.property_flags.contains(flags)) {
                return @truncate(u32, i);
            }
        }

        return error.NoSuitableMemoryType;
    }
};

pub const Queue = struct {
    handle: vk.Queue,
    family: u32,
    index: u32,

    fn init(vkd: DeviceDispatch, dev: vk.Device, family: u32, index: u32) Queue {
        return .{
            .handle = vkd.getDeviceQueue(dev, family, index),
            .family = family,
            .index = index,
        };
    }
};

pub const Device = struct {
    vkd: DeviceDispatch,

    pdev: PhysicalDeviceInfo,
    handle: vk.Device,

    graphics_queue: Queue,
    compute_queue: Queue,
    present_queue: Queue,

    pub fn init(instance: Instance, pdev: PhysicalDeviceInfo, extensions: []const [*:0]const u8, qalloc: QueueAllocation) !Device {
        const families = [_]u32{qalloc.graphics_family, qalloc.compute_family, qalloc.present_family};
        const priorities = [_]f32{1} ** families.len;

        var qci_buffer: [families.len]vk.DeviceQueueCreateInfo = undefined;
        var n_unique_families: u32 = 0;

        for (families) |family| {
            for (qci_buffer[0 .. n_unique_families]) |*qci| {
                if (qci.queue_family_index == family) {
                    if (qci.queue_count < qalloc.family_propsv[family].queue_count) {
                        qci.queue_count += 1;
                    }
                    break;
                }
            } else {
                qci_buffer[n_unique_families] = .{
                    .flags = .{},
                    .queue_family_index = family,
                    .queue_count = 1,
                    .p_queue_priorities = &priorities,
                };
                n_unique_families += 1;
            }
        }

        const handle = try instance.vki.createDevice(pdev.handle, .{
            .flags = .{},
            .queue_create_info_count = n_unique_families,
            .p_queue_create_infos = &qci_buffer,
            .enabled_layer_count = 0,
            .pp_enabled_layer_names = undefined,
            .enabled_extension_count = @intCast(u32, extensions.len),
            .pp_enabled_extension_names = extensions.ptr,
            .p_enabled_features = null,
        }, null);

        // Device is leaked if the following fails.
        const vkd = try DeviceDispatch.load(handle, instance.vki.vkGetDeviceProcAddr);
        errdefer vkd.destroyDevice(handle, null);

        var queues: [families.len]Queue = undefined;
        for (queues) |*queue, i| {
            const family = families[i];

            const qci = for (qci_buffer[0 .. n_unique_families]) |*qci| {
                if (qci.queue_family_index == family) {
                    break qci;
                }
            } else unreachable;

            // Use the queue_count field to check how many queues of this family
            // were already allocated. Some will have to share if there aren't enough...
            if (qci.queue_count > 0) {
                qci.queue_count -= 1;
            }

            queue.* = Queue.init(vkd, handle, family, qci.queue_count);
        }

        return Device{
            .vkd = vkd,
            .pdev = pdev,
            .handle = handle,

            // Indices according to `families`.
            .graphics_queue = queues[0],
            .compute_queue = queues[1],
            .present_queue = queues[2],
        };
    }

    pub fn deinit(self: Device) void {
        self.vkd.destroyDevice(self.handle, null);
    }

    pub fn allocate(self: Device, requirements: vk.MemoryRequirements, flags: vk.MemoryPropertyFlags) !vk.DeviceMemory {
        return try self.vkd.allocateMemory(self.handle, .{
            .allocation_size = requirements.size, // TODO: Alignment
            .memory_type_index = try self.pdev.findMemoryTypeIndex(requirements.memory_type_bits, flags),
        }, null);
    }
};
