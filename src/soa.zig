const std = @import("std");
const mem = std.mem;
const testing = std.testing;
const Allocator = mem.Allocator;

fn FieldType(comptime T: type, comptime field_name: []const u8) type {
    for (@typeInfo(T).Struct.fields) |field| {
        if (mem.eql(u8, field.name, field_name)) {
            return field.field_type;
        }
    }

    unreachable;
}

pub fn StructOfArrays(comptime T: type) type {
    return struct {
        const Self = @This();
        buf: [*]align(@alignOf(T)) u8,
        len: usize, // length in items, not in size of above buffer.
        allocator: *Allocator,

        pub fn empty(allocator: *Allocator) Self {
            return Self{
                .buf = @as([*]align(@alignOf(T)) u8, undefined),
                .len = 0,
                .allocator = allocator,
            };
        }

        pub fn alloc(allocator: *Allocator, len: usize) !Self {
            const buf = try allocator.allocWithOptions(u8, len * @sizeOf(T), @alignOf(T), null);
            return Self{
                .buf = buf.ptr,
                .len = len,
                .allocator = allocator,
            };
        }

        // Reallocating invalidates SoA contents!
        // If this returns error.OutOfMemory, the SoA contents is still valid.
        pub fn realloc(self: *Self, new_len: usize) !void {
            self.buf = (try self.allocator.realloc(self.buf[0 .. self.len * @sizeOf(T)], new_len * @sizeOf(T))).ptr;
            self.len = new_len;
        }

        pub fn shrink(self: *Self, new_len: usize) void {
            self.buf = self.allocator.shrink(self.buf[0 .. self.len * @sizeOf(T)], new_len * @sizeOf(T)).ptr;
            self.len = new_len;
        }

        pub fn free(self: Self) void {
            self.allocator.free(self.buf[0 .. self.len * @sizeOf(T)]);
        }

        fn startOffset(self: Self, comptime field: []const u8) usize {
            // byte_offset = field_offset / @sizeOf(T) * len * @sizeOf(T)
            //   = field_offset * len
            return @byteOffsetOf(T, field) * self.len;
        }

        pub fn slice(self: Self, comptime field: []const u8) []FieldType(T, field) {
            const F = FieldType(T, field);
            const offset = self.startOffset(field);
            const ptr = @alignCast(@alignOf(F), &self.buf[offset]);
            return @ptrCast([*]FieldType(T, field), ptr)[0 .. self.len];
        }

        pub fn at(self: Self, comptime field: []const u8, index: usize) *FieldType(T, field) {
            return &self.slice(field)[index];
        }
    };
}

test "SoA" {
    const soa = try StructOfArrays(extern struct {
        a: u16,
        b: [11]u32,
        c: u16,
    }).alloc(testing.allocator, 10);
    defer soa.free();

    testing.expectEqual(@as(usize, 0), soa.startOffset("a"));
    testing.expectEqual(@as(usize, 10 * 4), soa.startOffset("b"));
    testing.expectEqual(@as(usize, 10 * 4 + 11 * 4 * 10), soa.startOffset("c"));

    soa.slice("c")[5] = 123;
    testing.expectEqual(@as(u16, 123), soa.at("c", 5).*);

    soa.slice("b")[4][10] = 94624;
    testing.expectEqual(@as(u32, 94624), soa.at("b", 4)[10]);
}
