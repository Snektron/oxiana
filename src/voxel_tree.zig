const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

pub const NodeOffset = u32;
pub const Color = [4]u8;

pub const Index = struct {
    pub const AxisType = u32;
    x: AxisType,
    y: AxisType,
    z: AxisType,

    fn inBounds(self: Index, side_dim_minus_one: AxisType) bool {
        return self.x <= side_dim_minus_one and self.y <= side_dim_minus_one and self.z <= side_dim_minus_one;
    }
};

// The root node is located at offset 0. Pointers to this node are alwaysinvalid,
// as it would create a graph from the tree. This value is used to indicate that the node is a leaf.
pub const root_node_offset: NodeOffset = 0;

pub const ChildDescriptor = extern struct {
    color: Color,
    node_offset: NodeOffset,

    fn empty() ChildDescriptor {
        return .{
            .color = [_]u8{0, 0, 0, 0},
            .node_offset = root_node_offset,
        };
    }

    pub fn isLeaf(self: ChildDescriptor) bool {
        return self.node_offset == root_node_offset;
    }
};

pub const QueryResult = struct {
    parent_offset: NodeOffset,
    // Index of the child relative to the parent
    child_index: Index,
    height: u32,
};

pub fn VoxelTree(comptime N: Index.AxisType, comptime H: u32) type {
    std.debug.assert(N > 1);
    std.debug.assert(H > 0);

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

        pub fn init(allocator: *Allocator) Self {
           return Self{
                .allocator = allocator,
                .nodes = .{},
            };
        }

        pub fn deinit(self: *Self) void {
            self.nodes.deinit(self.allocator);
        }

        pub fn set(self: *Self, index: Index, color: Color) !void {
            if (!index.inBounds(Self.side_dim_minus_one)) {
                return error.OutOfBounds;
            }

            if (self.nodes.items.len == 0) {
                try self.nodes.append(self.allocator, Node.empty());
            }

            var parent_offset = root_node_offset;
            var child_height: u32 = height - 1;
            var parent_coord = index;

            // division rounds down due to -1, so add one after. This can't be done before to avoid overflow
            var child_side_dim = @divTrunc(Self.side_dim_minus_one, Self.children_per_edge) + 1;

            while (true) {
                const child_index = Index{
                    .x = parent_coord.x / child_side_dim,
                    .y = parent_coord.y / child_side_dim,
                    .z = parent_coord.z / child_side_dim,
                };

                const node = &self.nodes.items[parent_offset];
                const child = &node.children[child_index.x][child_index.y][child_index.z];

                if (child_height == 0) {
                    child.color = color;
                    return;
                }

                if (child.node_offset == root_node_offset) {
                    const new_node_offset = @intCast(u32, self.nodes.items.len);
                    try self.nodes.append(self.allocator, Node.empty());
                    child.node_offset = new_node_offset;
                }

                parent_coord.x %= child_side_dim;
                parent_coord.y %= child_side_dim;
                parent_coord.z %= child_side_dim;
                child_height -= 1;
                parent_offset = child.node_offset;
                child_side_dim /= children_per_edge;
            }
        }

        pub fn get(self: Self, index: Index) !?Color {
            const result = (try self.query(index)) orelse return null;
            const node = self.nodes.items[result.parent_offset];
            return node.children[result.child_index.x][result.child_index.y][result.child_index.z].color;
        }

        pub fn query(self: Self, index: Index) !?QueryResult {
            if (!index.inBounds(Self.side_dim_minus_one)) {
                return error.OutOfBounds;
            }

            if (self.nodes.items.len == 0) {
                return null;
            }

            var parent_offset = root_node_offset;
            var child_height: u32 = height - 1;
            var parent_coord = index;

            // division rounds down due to -1, so add one after. This can't be done before to avoid overflow
            var child_side_dim = @divTrunc(Self.side_dim_minus_one, Self.children_per_edge) + 1;

            while (true) {
                const child_index = Index{
                    .x = parent_coord.x / child_side_dim,
                    .y = parent_coord.y / child_side_dim,
                    .z = parent_coord.z / child_side_dim,
                };

                const node = self.nodes.items[parent_offset];
                const child = node.children[child_index.x][child_index.y][child_index.z];
                if (child.node_offset == root_node_offset or child_height == 0) {
                    return QueryResult{
                        .parent_offset = parent_offset,
                        .child_index = child_index,
                        .height = child_height,
                    };
                }

                parent_coord.x %= child_side_dim;
                parent_coord.y %= child_side_dim;
                parent_coord.z %= child_side_dim;
                child_height -= 1;
                parent_offset = child.node_offset;
                child_side_dim /= children_per_edge;
            }
        }
    };
}

test "VoxelTree - refAllDecls" {
    std.meta.refAllDecls(@This());
}

test "VoxelTree - query" {
    var tree = VoxelTree(2, 2).init(std.testing.allocator);
    defer tree.deinit();

    std.debug.print("{}\n", .{ tree.query(.{ .x = 0, .y = 0, .z = 0 }) });

    try tree.nodes.append(tree.allocator, VoxelTree(2, 2).Node.empty());
    std.debug.print("{}\n", .{ tree.query(.{ .x = 2, .y = 0, .z = 0 }) });

    try tree.nodes.append(tree.allocator, VoxelTree(2, 2).Node.empty());
    tree.nodes.items[0].children[0][0][0].node_offset = 1;
    std.debug.print("{}\n", .{ tree.query(.{ .x = 0, .y = 0, .z = 0 }) });
}

test "VoxelTree - query/set" {
    var tree = VoxelTree(2, 2).init(std.testing.allocator);
    defer tree.deinit();

    try tree.set(.{.x = 0, .y = 0, .z = 0}, [_]u8{'t', 'e', 's', 't'});
    std.debug.print("{}\n", .{ tree.get(.{.x = 1, .y = 0, .z = 0}) });
}
