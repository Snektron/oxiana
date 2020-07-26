fn ManyPtr(comptime Ptr: type) type {
    var type_info = @typeInfo(Ptr);
    type_info.Pointer.size = .Many;
    return @Type(type_info);
}

fn ArrayPtr(comptime Ptr: type) type {
    const type_info = @typeInfo(Ptr);
    var array_ptr_info = @typeInfo(*[1]type_info.Pointer.child);
    array_ptr_info.Pointer.is_const = type_info.Pointer.is_const;
    array_ptr_info.Pointer.is_volatile = type_info.Pointer.is_volatile;
    array_ptr_info.Pointer.alignment = type_info.Pointer.alignment;
    array_ptr_info.Pointer.is_allowzero = type_info.Pointer.is_allowzero;
    return @Type(array_ptr_info);
}

pub fn asManyPtr(ptr: anytype) ManyPtr(@TypeOf(ptr)) {
    return @ptrCast(ArrayPtr(@TypeOf(ptr)), ptr);
}
