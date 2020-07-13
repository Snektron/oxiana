const std = @import("std");
const vk = @import("vulkan");
const mem = std.mem;
const Allocator = mem.Allocator;

const BaseDispatch = struct {
    vkCreateInstance: vk.PfnCreateInstance,

    usingnamespace vk.BaseWrapper(@This());
};

const InstanceDispatch = struct {
    vkDestroyInstance: vk.PfnDestroyInstance,
    vkEnumeratePhysicalDevices: vk.PfnEnumeratePhysicalDevices,
    vkDestroySurfaceKHR: vk.PfnDestroySurfaceKHR,
    vkGetPhysicalDeviceQueueFamilyProperties: vk.PfnGetPhysicalDeviceQueueFamilyProperties,
    vkGetPhysicalDeviceProperties: vk.PfnGetPhysicalDeviceProperties,
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

const DeviceDispatch = struct {
    vkDestroyDevice: vk.PfnDestroyDevice,

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
            .vki = try InstanceDispatch.load(handle, loader),
            .handle = handle,
        };
    }

    pub fn deinit(self: Instance) void {
        self.vki.vkDestroyInstance(self.handle, null);
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

    pub fn findAndCreateDevice(self: Instance, allocator: *Allocator, surface: vk.SurfaceKHR, extensions: []const []const u8) !Device {
        const pdev_infos = try self.physicalDeviceInfos(allocator);
        defer allocator.free(pdev_infos);

        var tmp_extensions = try allocator.alloc([]const u8, extensions.len);
        defer allocator.free(tmp_extensions);

        for (pdev_infos) |pdev| {
            mem.copy([]const u8, tmp_extensions, extensions);
            if (!(try pdev.supportsSurface(self.vki, surface))) {
                std.log.info(.graphics, "Cannot use device '{}': Surface not supported\n", .{pdev.name()});
                continue;
            }

            var unsupported_extensions = try pdev.filterUnsupportedExtensions(self.vki, allocator, tmp_extensions);
            if (unsupported_extensions.len > 0) {
                std.log.info(.graphics, "Cannot use device '{}': {} unsupported extension(s)\n", .{pdev.name(), unsupported_extensions.len});
                for (unsupported_extensions) |ext| {
                    std.log.info(.graphics, "  Extension '{}' not supported\n", .{ext});
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

                std.log.info(.graphics, "Cannot use device '{}': {}\n", .{pdev.name(), message});
                continue;
            };

            // Just pick the first available suitable device
            return try Device.init(self, pdev, queues);
        }

        return error.NoSuitableDevice;
    }
};

pub const QueueAllocation = struct {
    graphics_family: u32,
    present_family: u32,
    compute_family: u32,
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
        extensions: [][]const u8,
    ) ![][]const u8 {
        var count: u32 = undefined;
        _ = try vki.enumerateDeviceExtensionProperties(self.handle, null, &count, null);

        const propsv = try allocator.alloc(vk.ExtensionProperties, count);
        defer allocator.free(propsv);

        _ = try vki.enumerateDeviceExtensionProperties(self.handle, null, &count, propsv.ptr);

        var write_index: usize = 0;
        for (extensions) |ext| {
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
        defer allocator.free(propsv);
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
            .graphics_family = graphics_family,
            .present_family = present_family,
            .compute_family = compute_family,
        };
    }
};

pub const Queue = struct {
    handle: vk.Queue,
    family: u32,

    fn init(vkd: DeviceDispatch, dev: vk.Device, family: u32) Queue {
        return .{
            .handle = vkd.getDeviceQueue(dev, family, 0),
            .family = family,
        };
    }
};

pub const Device = struct {
    vkd: DeviceDispatch,

    pdev_info: PhysicalDeviceInfo,
    handle: vk.Device,

    present_queue: Queue,
    graphics_queue: Queue,
    compute_queue: Queue,

    pub fn init(instance: Instance, pdev_info: PhysicalDeviceInfo, queues: QueueAllocation) !Device {
        return error.Wip;
    }

    pub fn deinit(self: Device) void {
        self.vkd.destroyDevice(self.handle);
    }
};
