const std = @import("std");
const testing = std.testing;

fn elementConstant(comptime ElementType: type, value: anytype) ElementType {
    return switch (@typeInfo(ElementType)) {
        .Vector => |v| @splat(v.len, @as(v.child, value)),
        .Int, .Float => value,
        else => @compileError("Invalid element type " ++ @typeName(ElementType)), 
    };
}

pub fn Matrix(comptime T: type, comptime M: usize, comptime N: usize) type {
    std.debug.assert(M >= 1 and N >= 1);
    return extern struct {
        const Self = @This();
        pub const ElementType = T;
        pub const cols = M;
        pub const rows = N;
        pub const zero = broadcast(elementConstant(ElementType, 0));

        pub const MemoryLayout = enum {
            row_major,
            column_major,
        };

        elements: [cols][rows]ElementType,

        pub fn broadcast(initial: ElementType) Self {
            @setRuntimeSafety(false);
            var self: Self = undefined;

            for (self.elements) |*col| {
                for (col) |*cell| cell.* = initial;
            }

            return self;
        }

        pub fn fromArray(layout: MemoryLayout, arr: [cols * rows]ElementType) Self {
            @setRuntimeSafety(false);
            var self: Self = undefined;

            switch (layout) {
                .row_major => {
                    for (self.elements) |*col, i| {
                        for (col) |*cell, j| cell.* = arr[j * cols + i];
                    }
                },
                .column_major => {
                    for (self.elements) |*col, i| {
                        for (col) |*cell, j| cell.* = arr[i * rows + j];
                    }
                },
            }

            return self;
        }

        pub fn format(self: Self, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
            if (cols == 1) {
                // Printing a column vector as matrix wastes a lot of space, so print it as [a  b  c]^T instead
                try writer.writeByte('[');
                for (self.elements[0]) |cell, i| {
                    if (i != 0) {
                        try writer.writeByte(' ');
                    }
                    try std.fmt.formatType(cell, fmt, options, writer, std.fmt.default_max_depth);
                }
                try writer.writeAll("]^T");
            } else {
                var row: usize = 0;
                while (row < rows) : (row += 1) {
                    if (row != 0) {
                        try writer.writeByte('\n');
                    }
                    var col: usize = 0;
                    while (col < cols) : (col += 1) {
                        if (col != 0) {
                            try writer.writeByte(' ');
                        }
                        try std.fmt.formatType(self.elements[col][row], fmt, options, writer, std.fmt.default_max_depth);
                    }
                }
            }
        }

        fn MatMulResultType(comptime Rhs: type) type {
            std.debug.assert(cols == Rhs.rows);
            std.debug.assert(ElementType == Rhs.ElementType);
            return Matrix(ElementType, Rhs.cols, rows);
        }

        pub fn matmul(lhs: Self, rhs: anytype) MatMulResultType(@TypeOf(rhs)) {
            @setRuntimeSafety(false);
            var result: MatMulResultType(@TypeOf(rhs)) = undefined;

            for (result.elements) |*col, i| {
                for (col) |*cell, j| {
                    var value = lhs.elements[0][j] * rhs.elements[i][0];
                    var k: usize = 1;
                    while (k < cols) : (k += 1) {
                        value += lhs.elements[k][j] * rhs.elements[i][k];
                    }
                    cell.* = value;
                }
            }

            return result;
        }

        // Element-wise multiplication. See `matmul` for matrix multiplication.
        pub fn mul(lhs: Self, rhs: Self) Self {
            @setRuntimeSafety(false);
            var result: Self = undefined;
            for (result.elements) |*col, i| {
                for (col) |*cell, j| {
                    col.* = lhs.elements[i][j] * rhs.elements[i][j];
                }
            }
            return result;
        }

        pub fn div(lhs: Self, rhs: Self) Self {
            @setRuntimeSafety(false);
            var result: Self = undefined;
            for (result.elements) |*col, i| {
                for (col) |*cell, j| {
                    col.* = lhs.elements[i][j] / rhs.elements[i][j];
                }
            }
            return result;
        }

        pub fn add(lhs: Self, rhs: Self) Self {
            @setRuntimeSafety(false);
            var result: Self = undefined;
            for (result.elements) |*col, i| {
                for (col) |*cell, j| {
                    cell.* = lhs.elements[i][j] + rhs.elements[i][j];
                }
            }
            return result;
        }

        pub fn sub(lhs: Self, rhs: Self) Self {
            @setRuntimeSafety(false);
            var result: Self = undefined;
            for (result.elements) |*col, i| {
                for (col) |*cell, j| {
                    cell.* = lhs.elements[i][j] - rhs.elements[i][j];
                }
            }
            return result;
        }

        pub fn negate(self: Self) Self {
            @setRuntimeSafety(false);
            var result: Self = undefined;
            for (result.elements) |*col, i| {
                for (col) |*cell, j| {
                    cell.* = -self.elements[i][j];
                }
            }
            return result;
        }

        pub fn scale(self: Self, scalar: ElementType) Self {
            var result = self;
            for (result.elements) |*col| {
                for (col) |*cell| cell.* *= scalar;
            }
            return result;
        }

        pub fn transpose(self: Self) Matrix(ElementType, rows, cols) {
            var result: Matrix(ElementType, rows, cols) = undefined;
            for (result.elements) |*col, i| {
                for (col) |*cell, j| {
                    cell.* = self.elements[j][i];
                }
            }
            return result;
        }

        pub usingnamespace if (cols == rows) SquareMatrixMixin(Self) else struct {};
        pub usingnamespace if (cols == 1) ColumnVectorMixin(Self) else struct {};
        pub usingnamespace if (cols == 1 and rows == 3) CrossMixin(Self) else struct {};
    };
}

fn SquareMatrixMixin(comptime Self: type) type {
    std.debug.assert(Self.rows == Self.cols);
    return struct {
        pub const identity = broadcastDiag(
            elementConstant(Self.ElementType, 0),
            elementConstant(Self.ElementType, 1)
        );

        pub fn broadcastDiag(initial: Self.ElementType, diag: Self.ElementType) Self {
            @setRuntimeSafety(false);
            var self: Self = undefined;
            for (self.elements) |*col, i| {
                for (col) |*cell, j| {
                    cell.* = if (i == j) diag else initial;
                }
            }

            return self;
        }
    };
}

fn ColumnVectorMixin(comptime Self: type) type {
    std.debug.assert(Self.cols == 1);
    return struct {
        pub fn dot(lhs: Self, rhs: Self) ElementType {
            var result = lhs.elements[0][0] * rhs.elements[0][0];
            var i: usize = 1;
            while (i < Self.rows) : (i += 1) {
                result += lhs.elements[0][i] * rhs.elements[0][i];
            }
            return result;
        }

        pub fn magnitudeSquared(self: Self) ElementType {
            return dot(self, self);
        }

        pub fn magnitude(self: Self) ElementType {
            return std.math.sqrt(self.magnitudeSquared());
        }

        pub fn normalize(self: Self) Self {
            var mag = self.magnitude();
            if (@typeInfo(ElementType) == .Vector) {
                const scalar: ElementType = undefined;
                for (scalar) |*channel, i| {
                    channel.* = if (mag[i] == 0) 0 else 1 / mag;
                }

                return self.scale(scalar);
            } else {
                return if (mag == 0) // TODO: Fix for SIMD
                        Self.zero
                    else
                        self.scale(elementConstant(ElementType, 1) / mag);
            }
        }

        pub fn swizzle(self: Self, comptime order: []const u8) Vec(Self.ElementType, order.len) {
            var result: Vec(Self.ElementType, order.len) = undefined;

            // TODO: Think of way to handle elements with index > 3
            inline for (order) |c, i| {
                result.elements[0][i] = switch (order[i]) {
                    'x' => self.elements[0][0],
                    'y' => self.elements[0][1],
                    'z' => self.elements[0][2],
                    'w' => self.elements[0][3],
                    '0' => 0,
                    '1' => 1,
                    else => @compileError("Invalid order " ++ order[i]),
                };
            }

            return result;
        }
    };
}

fn CrossMixin(comptime Self: type) type {
    std.debug.assert(Self.cols == 1 and Self.rows == 3);
    return struct {
        pub fn cross(lhs: Self, rhs: Self) Self {
            return vec3(
                Self.ElementType,
                lhs.elements[0][1] * rhs.elements[0][2] - lhs.elements[0][2] * rhs.elements[0][1],
                lhs.elements[0][2] * rhs.elements[0][0] - lhs.elements[0][0] * rhs.elements[0][2],
                lhs.elements[0][0] * rhs.elements[0][1] - lhs.elements[0][1] * rhs.elements[0][0],
            );
        }
    };
}

pub fn Vec(comptime T: type, comptime N: usize) type {
    return Matrix(T, 1, N);
}

pub fn vec(comptime T: type, comptime N: usize, elements: [N]T) Vec(T, N) {
    return .{.elements = [_][N]T{ elements }};
}

pub fn vec2(comptime T: type, x: T, y: T) Vec(T, 2) {
    return vec(T, 2, [_]T{ x, y, z });
}

pub fn vec3(comptime T: type, x: T, y: T, z: T) Vec(T, 3) {
    return vec(T, 3, [_]T{ x, y, z });
}

pub fn vec4(comptime T: type, x: T, y: T, z: T, w: T) Vec(T, 4) {
    return vec(T, 4, [_]T{ x, y, z, w });
}

test "Matrix basic operations" {
    const Mat = Matrix(i32, 4, 3);
    testing.expectEqual(4, Mat.cols);
    testing.expectEqual(3, Mat.rows);

    for (Mat.zero.elements) |col| {
        for (col) |cell| testing.expectEqual(@as(i32, 0), cell);
    }

    for (Mat.broadcast(-123).elements) |col| {
        for (col) |cell| testing.expectEqual(@as(i32, -123), cell);
    }

    const a = Mat.fromArray(.row_major, [_]i32{
        0, 1,  2,  3,
        4, 5,  6,  7,
        8, 9, 10, 11,
    });

    testing.expectEqual([_]i32{0, 4, 8}, a.elements[0]);
    testing.expectEqual([_]i32{1, 5, 9}, a.elements[1]);
    testing.expectEqual([_]i32{2, 6, 10}, a.elements[2]);
    testing.expectEqual([_]i32{3, 7, 11}, a.elements[3]);

    const b = Mat.fromArray(.column_major, [_]i32{
        -10, -20, -30,
        -11, -21, -31,
        -12, -22, -32,
        -13, -23, -33,
    });

    testing.expectEqual([_]i32{-10, -20, -30}, b.elements[0]);
    testing.expectEqual([_]i32{-11, -21, -31}, b.elements[1]);
    testing.expectEqual([_]i32{-12, -22, -32}, b.elements[2]);
    testing.expectEqual([_]i32{-13, -23, -33}, b.elements[3]);

    {
        const t = b.transpose();
        const T = @TypeOf(t);
        testing.expectEqual(3, T.cols);
        testing.expectEqual(4, T.rows);

        testing.expectEqual([_]i32{-10, -11, -12, -13}, t.elements[0]);
        testing.expectEqual([_]i32{-20, -21, -22, -23}, t.elements[1]);
        testing.expectEqual([_]i32{-30, -31, -32, -33}, t.elements[2]);
    }

    {
        const mul_result = Matrix(i32, 3, 3).fromArray(.row_major, [_]i32{
             -74, -134, -194,
            -258, -478, -698,
            -442, -822, -1202,
        });

        testing.expectEqual(mul_result, a.matmul(b.transpose()));
    }
}
