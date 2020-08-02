# oxiana

Voxel ray tracing experiment in Zig.

Building requires the Vulkan SDK for vk.xml and debug layers, however, one can build te project without by hardcoding `vk_xml_path` to `deps/vulkan-zig/examples/vk.xml` and removing the `vk_sdk_path` declaration from `build.zig`.
