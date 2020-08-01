const std = @import("std");

pub fn VoxelVolume(comptime log2_side_dim_: u4, comptime T: type) type {
    return struct {
        const Self = @This();
        pub const ElementType = T;
        pub const log2_side_dim = log2_side_dim_;
        pub const side_dim = 1 << log2_side_dim;
        voxels: [side_dim][side_dim][side_dim]T,

        pub fn clear(self: *Self, data: T) void {
            for (self.voxels) |*plane| {
                for (plane) |*row| {
                    for (row) |*voxel| voxel.* = data;
                }
            }
        }
    };
}
