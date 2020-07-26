const std = @import("std");
const c = @import("c.zig");
const Oxiana = @import("oxiana.zig").Oxiana;

pub fn main() !void {
    if (c.glfwInit() != c.GLFW_TRUE) return error.GlfwInitFailed;
    defer c.glfwTerminate();

    const ox = try Oxiana.init(std.heap.page_allocator);
    defer ox.deinit();
    try ox.loop();
}
