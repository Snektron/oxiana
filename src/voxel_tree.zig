const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const NodeOffset = u32;

// The root node is located at offset 0. Pointers to this node are alwaysinvalid,
// as it would create a graph from the tree. This value is used to indicate that the node is a leaf.
pub const INVALID_NODE_OFFSET: NodeOffset = 0;

pub const ChildDescriptor = extern struct {
    color: [4]u8,
    node_offset: NodeOffset,

    pub fn isLeaf(self: ChildDescriptor) bool {
        return self.node_offset == INVALID_NODE_OFFSET;
    }
};

pub const QueryResult = struct {
    node_offset: NodeOffset,
    height: u32,
};

pub fn VoxelTree(comptime N: u32) type {
    std.debug.assert(N > 1);

    // Calculate voxel tree maxima
    var height: u32 = 0;
    // Calculate in a higher precision to not have to worry about overflows
    // (u32_max + 1) * u32_max < u64_max
    var side_dim: u64 = 1;
    while (side_dim * N <= 1 << u32.bit_count) {
        height += 1;
        side_dim *= N;
    }

    return struct {
        pub const Self = @This();
        pub const children_per_edge = N;
        pub const children_per_node = N * N * N;
        pub const max_height = height;
        pub const max_side_dim_minus_one = @intCast(u32, side_dim - 1);

        pub const Node = extern struct {
            children: [N][N][N]ChildDescriptor,
        };

        nodes: std.ArrayList(Node),
        height: u32,
        side_dim: u32,

        pub fn init(allocator: *Allocator) Self {
            return .{
                .nodes = std.ArrayList(Node).init(0),
                .height = 0,
                .side_dim = 0,
            };
        }
    };
}

test "VoxelTree - refAllDecls" {
    std.meta.refAllDecls(@This());
}

test "VoxelTree - maxima" {
    testing.expectEqual(@as(u32, 32), VoxelTree(2).max_height);
    testing.expectEqual(@as(u32, (1 << 32) - 1), VoxelTree(2).max_side_dim_minus_one);

    testing.expectEqual(@as(u32, 13), VoxelTree(5).max_height);
    testing.expectEqual((std.math.powi(u32, 5, 13) catch unreachable) - 1, VoxelTree(5).max_side_dim_minus_one);

    testing.expectEqual(@as(u32, 1), VoxelTree((1 << 32) - 1).max_height);
    testing.expectEqual(@as(u32, (1 << 32) - 2), VoxelTree((1 << 32) - 1).max_side_dim_minus_one);
}

