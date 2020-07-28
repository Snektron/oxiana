const std = @import("std");
const linalg = @import("linalg.zig");
const testing = std.testing;

const slerp_dot_threshold = 0.9995;

pub fn Quaternion(comptime T: type) type {
    return extern struct {
        const Self = @This();
        pub const ElementType = T;
        pub const identity = Self{.a = 1, .b = 0, .c = 0, .d = 0};

        a: T, // Real part
        b: T, // Imaginary i part
        c: T, // Imaginary j part
        d: T, // Imaginary k part

        pub fn init(a: T, b: T, c: T, d: T) Self {
            return .{.a = a, .b = b, .c = c, .d = d};
        }

        pub fn axisAngle(axis: linalg.Vec(ElementType, 3), angle: ElementType) Self {
            const cosa2 = std.math.cos(angle * 0.5);
            const sina2 = std.math.sin(angle * 0.5);
            const vec = axis.scale(sina2);
            return .{
                .a = cosa2,
                .b = vec.elements[0][0],
                .c = vec.elements[0][1],
                .d = vec.elements[0][2],
            };
        }

        pub fn add(lhs: Self, rhs: Self) Self {
            return .{
                .a = lhs.a + rhs.a,
                .b = lhs.b + rhs.b,
                .c = lhs.c + rhs.c,
                .d = lhs.d + rhs.d,
            };
        }

        pub fn sub(lhs: Self, rhs: Self) Self {
            return .{
                .a = lhs.a - rhs.a,
                .b = lhs.b - rhs.b,
                .c = lhs.c - rhs.c,
                .d = lhs.d - rhs.d,
            };
        }

        pub fn mul(lhs: Self, rhs: Self) Self {
            return .{
                .a = lhs.a * rhs.a - lhs.b * rhs.b - lhs.c * rhs.c - lhs.d * rhs.d,
                .b = lhs.a * rhs.b + lhs.b * rhs.a + lhs.c * rhs.d - lhs.d * rhs.c,
                .c = lhs.a * rhs.c - lhs.b * rhs.d + lhs.c * rhs.a + lhs.d * rhs.b,
                .d = lhs.a * rhs.d + lhs.b * rhs.c - lhs.c * rhs.b + lhs.d * rhs.a,
            };
        }

        pub fn mulScalar(lhs: Self, rhs: T) Self {
            return .{
                .a = lhs.a * rhs,
                .b = lhs.b * rhs,
                .c = lhs.c * rhs,
                .d = lhs.d * rhs,
            };
        }

        pub fn divScalar(lhs: Self, rhs: T) Self {
            return lhs.mulScalar(1 / rhs);
        }

        pub fn conjugate(self: Self) Self {
            return .{
                .a = self.a,
                .b = -self.b,
                .c = -self.c,
                .d = -self.d,
            };
        }

        pub fn reciprocal(self: Self) Self {
            return self.conjugate().divScalar(self.magnitudeSquared());
        }

        pub fn negate(self: Self) Self {
            return .{
                .a = -self.a,
                .b = -self.b,
                .c = -self.c,
                .d = -self.d,
            };
        }

        pub fn magnitudeSquared(self: Self) T {
            return self.a * self.a + self.b * self.b + self.c * self.c + self.d * self.d;
        }

        pub fn magnitude(self: Self) T {
            return std.math.sqrt(self.magnitudeSquared());
        }

        pub fn normalize(self: Self) Self {
            return self.divScalar(self.magnitude());
        }

        pub fn format(
            self: Self,
            comptime fmt: []const u8,
            options: std.fmt.FormatOptions,
            writer: anytype
        ) !void {
            try std.fmt.formatType(self.a, fmt, options, writer, std.fmt.default_max_depth);

            const suffices = [_]u8{'i', 'j', 'k'};

            for ([_]T{self.b, self.c, self.d}) |elem, i| {
                try writer.writeAll(if (elem < 0) " - " else " + ");
                try std.fmt.formatType(std.math.fabs(elem), fmt, options, writer, std.fmt.default_max_depth);
                try writer.writeByte(suffices[i]);
            }
        }

        pub fn forward(self: Self) linalg.Vec(ElementType, 3) {
            return linalg.vec3(
                ElementType,
                2 * (self.a * self.c + self.b * self.d),
                2 * (self.c * self.d - self.a * self.b),
                1 - 2 * (self.b * self.b + self.c * self.c),
            );
        }

        pub fn up(self: Self) linalg.Vec(ElementType, 3) {
            return linalg.vec3(
                ElementType,
                2 * (self.b * self.c - self.a * self.d),
                1 - 2 * (self.b * self.b + self.d * self.d),
                2 * (self.a * self.b + self.c * self.d),
            );
        }

        pub fn right(self: Self) linalg.Vec(ElementType, 3) {
            return linalg.vec3(
                ElementType,
                1 - 2 * (self.c * self.c + self.d * self.d),
                2 * (self.b * self.c + self.a * self.d),
                2 * (self.b * self.d - self.a * self.c),
            );
        }
    };
}

pub fn abs(q: anytype) @TypeOf(q).ElementType {
    return q.magnitude();
}

pub fn lerp(lhs: anytype, rhs: @TypeOf(lhs), t: @TypeOf(lhs).ElementType) @TypeOf(lhs) {
    return lhs.add(rhs.sub(lhs).mulScalar(t));
}

// https://en.wikipedia.org/wiki/Slerp
pub fn slerp(lhs: anytype, rhs: @TypeOf(lhs), t: @TypeOf(lhs).ElementType) @TypeOf(rhs) {
    const p = lhs.normalize();
    var q = rhs.normalize();

    var dot = p.a * q.a + p.b * q.b + p.c * q.c + p.d * q.d;
    if (dot < 0) {
        q = q.negate();
        dot = -dot;
    }

    if (dot > slerp_dot_threshold) {
        return lerp(p, q, t).normalize();
    }

    const theta_0 = std.math.acos(dot);
    const theta = theta_0 * t;
    const sin_theta = std.math.sin(theta);
    const sin_theta_0 = std.math.sin(theta_0);

    const s0 = std.math.cos(theta) - dot * sin_theta / sin_theta_0;
    const s1 = sin_theta / sin_theta_0;

    return p.mulScalar(s0).add(q.mulScalar(s1));
}

const epsilon = 0.00001;

test "slerp" {
    const p = Quaternion(f32).identity;
    const q = Quaternion(f32).init(0, 1, 0, 0);
    const r = slerp(p, q, 0.5);

    testing.expect(
        std.math.approxEq(f32, r.a, 0.5 * std.math.sqrt2, epsilon) and
        std.math.approxEq(f32, r.b, 0.5 * std.math.sqrt2, epsilon) and
        std.math.approxEq(f32, r.c, 0, epsilon) and
        std.math.approxEq(f32, r.d, 0, epsilon)
    );

    const t = slerp(r, Quaternion(f32).init(0, -1, 1, -1).normalize(), 0.9);
    testing.expect(
        std.math.approxEq(f32, t.a, 0.08890256, epsilon) and
        std.math.approxEq(f32, t.b, 0.63280339, epsilon) and
        std.math.approxEq(f32, t.c, -0.54390082, epsilon) and
        std.math.approxEq(f32, t.d, 0.54390082, epsilon)
    );

    const u = slerp(p, Quaternion(f32).init(1, 0.01, 0, 0).normalize(), 0.9);
    testing.expect(
        std.math.approxEq(f32, u.a, 0.9999595, epsilon) and
        std.math.approxEq(f32, u.b, 0.00899959, epsilon) and
        std.math.approxEq(f32, u.c, 0, epsilon) and
        std.math.approxEq(f32, u.d, 0, epsilon)
    );

}

test "Quaternion basic operations" {
    const Q = Quaternion(f32);
    const p = Q.init(1, 2, 3, 4);
    const q = Q.init(5, 2, 7, -3);

    testing.expectEqual(Q.init(6, 4, 10, 1), p.add(q));
    testing.expectEqual(Q.init(-4, 0, -4, 7), p.sub(q));
    testing.expectEqual(Q.init(10, 4, 14, -6), q.mulScalar(2));
    testing.expectEqual(Q.init(2.5, 1, 3.5, -1.5), q.divScalar(2));
    testing.expectEqual(Q.init(5, -2, -7, 3), q.conjugate());
    testing.expectEqual(Q.init(-5, -2, -7, 3), q.negate());

    testing.expectEqual(@as(f32, 11 * 11),  Q.init(1, 2, 4, 10).magnitudeSquared());
    testing.expectEqual(@as(f32, 9),  Q.init(2, 4, 5, 6).magnitude());
}

test "Quaternion.mul" {
    const p = Quaternion(f32).init(-4, 2, 3, -2);
    const q = Quaternion(f32).init(0, 5, -3, 4);
    const r = p.mul(q);
    testing.expect(r.a == 7 and r.b == -14 and r.c == -6 and r.d == -37);
}

test "Quaternion.reciprocal" {
    const p = Quaternion(f32).init(3, 5, -4, 1);
    const q = p.reciprocal();

    testing.expect(
        std.math.approxEq(f32, q.a, @as(f32, 3) / 51, epsilon) and
        std.math.approxEq(f32, q.b, @as(f32, -5) / 51, epsilon) and
        std.math.approxEq(f32, q.c, @as(f32, 4) / 51, epsilon) and
        std.math.approxEq(f32, q.d, @as(f32, -1) / 51, epsilon)
    );
}

test "Quaternion.normalize" {
    const p = Quaternion(f32).init(5, 1, -3, 5);
    const q = p.normalize();
    const mag = p.magnitude();

    testing.expect(
        std.math.approxEq(f32, q.a, @as(f32, 5) / mag, epsilon) and
        std.math.approxEq(f32, q.b, @as(f32, 1) / mag, epsilon) and
        std.math.approxEq(f32, q.c, @as(f32, -3) / mag, epsilon) and
        std.math.approxEq(f32, q.d, @as(f32, 5) / mag, epsilon)
    );
}
