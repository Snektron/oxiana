const math = @import("../math.zig");

pub const Camera = struct {
    rotation: math.Quaternion(f32),
    translation: math.Vec(f32, 3),

    pub fn rotate(self: *Camera, q: math.Quaternion(f32)) void {
        self.rotation = self.rotation.mul(q).normalize();
    }

    pub fn rotatePitch(self: *Camera, amount: f32) void {
        const q = math.Quaternion(f32).axisAngle(math.vec3(f32, 1, 0, 0), amount);
        self.rotation = self.rotation.mul(q).normalize();
    }

    pub fn rotateYaw(self: *Camera, amount: f32) void {
        const q = math.Quaternion(f32).axisAngle(math.vec3(f32, 0, 1, 0), amount);
        self.rotation = self.rotation.mul(q).normalize();
    }

    pub fn rotateRoll(self: *Camera, amount: f32) void {
        const q = math.Quaternion(f32).axisAngle(math.vec3(f32, 0, 0, 1), amount);
        self.rotation = self.rotation.mul(q).normalize();
    }

    pub fn moveForward(self: *Camera, amount: f32) void {
        self.translation = self.translation.add(self.rotation.forward().scale(amount));
    }

    pub fn moveRight(self: *Camera, amount: f32) void {
        self.translation = self.translation.add(self.rotation.right().scale(amount));
    }

    pub fn moveUp(self: *Camera, amount: f32) void {
        self.translation = self.translation.add(self.rotation.up().scale(amount));
    }
};
