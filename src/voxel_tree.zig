const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const NodeOffset = u32;
pub const Color = [4]u8;

pub const Coordinate = struct {
    pub const AxisType = u32;
    x: AxisType,
    y: AxisType,
    z: AxisType,
};

// The root node is located at offset 0. Pointers to this node are alwaysinvalid,
// as it would create a graph from the tree. This value is used to indicate that the node is a leaf.
pub const INVALID_NODE_OFFSET: NodeOffset = 0;

pub const ChildDescriptor = extern struct {
    color: Color,
    node_offset: NodeOffset,

    fn empty() ChildDescriptor {
        return .{
            .color = [_]u8{0, 0, 0, 0},
            .node_offset = INVALID_NODE_OFFSET,
        };
    }

    pub fn isLeaf(self: ChildDescriptor) bool {
        return self.node_offset == INVALID_NODE_OFFSET;
    }
};

pub fn VoxelTree(comptime N: Coordinate.AxisType, comptime H: u32) type {
    std.debug.assert(N > 1);
    std.debug.assert(H > 1);

    return struct {
        const Self = @This();

        pub const children_per_edge = N;
        pub const height = H;

        // Calculate in a higher precision to avoid overflows before the subtraction is applied.
        pub const side_dim_minus_one = comptime @intCast(u32, (std.math.powi(u64, N, H) catch unreachable) - 1);

        pub const Node = struct {
            children: [N][N][N]ChildDescriptor,

            fn empty() Node {
                return .{
                    .children = [_][N][N]ChildDescriptor{
                        [_][N]ChildDescriptor{
                            [_]ChildDescriptor{ChildDescriptor.empty()} ** N
                        } ** N
                    } ** N,
                };
            }
        };

        allocator: *Allocator,
        nodes: std.ArrayListUnmanaged(Node),
        free_nodes: std.ArrayListUnmanaged(NodeOffset),
        root_color: Color,

        pub fn init(allocator: *Allocator) Self {
           return Self{
                .allocator = allocator,
                .nodes = .{},
                .free_nodes = .{},
                .root_color = [_]u8{0, 0, 0, 0},
            };
        }
    };
}

test "VoxelTree - refAllDecls" {
    std.meta.refAllDecls(@This());
}
