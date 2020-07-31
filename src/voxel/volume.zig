const std = @import("std");

pub fn VoxelVolume(comptime log2_side_dim: u4, comptime T: type) type {
    return struct {
        const Self = @This();
        pub const ElementType = T;
        pub const side_dim = 1 << log2_side_dim;
        voxels: [side_dim][side_dim][side_dim]T;
    };
}
