import re, sys

PRESENT = re.compile(
    r"""            let indices = \[drawable\.index\];\n"""
    r"""            let swapchains = \[swapchain\.swapchain\];\n"""
    r"""            let waits = \[drawable\.released\];\n"""
    r"""            let present = unsafe \{\n"""
    r"""                device\.swapchain_ext\.queue_present\(\n"""
    r"""                    device\.queue\(\),\n"""
    r"""                    &vk::PresentInfoKHR::default\(\)\n"""
    r"""                        \.wait_semaphores\(&waits\)\n"""
    r"""                        \.swapchains\(&swapchains\)\n"""
    r"""                        \.image_indices\(&indices\),\n"""
    r"""                \)\n"""
    r"""            \};\n"""
    r"""            if matches!\(present, Err\(vk::Result::ERROR_OUT_OF_DATE_KHR\) \| Ok\(true\)\) \{\n"""
    r"""                resized = true;\n"""
    r"""            \}\n""")

SUBS = [
    ("use ash::vk;\n", ""),
    ("use nga_vk::*;", "use nga::*;"),
    ("Swapchain", "Surface"),
    ("swapchain", "surface"),
    ("device.name)", "device.name())"),
    ("unsafe { device.device.device_wait_idle().unwrap() };", "device.wait_idle().unwrap();"),
    ("unsafe { device.device.device_wait_idle()? };", "device.wait_idle()?;"),
    ("&frame.arenas[0]", "frame.arena(0)"),
    ("frame.arenas[0]", "frame.arena(0)"),
    ("depth_state(vk::CompareOp::LESS, true)", "DepthState::new(CompareOp::Less, true)"),
    ("depth_state(vk::CompareOp::LESS, false)", "DepthState::new(CompareOp::Less, false)"),
    ("depth_state(vk::CompareOp::ALWAYS, false)", "DepthState::new(CompareOp::Always, false)"),
    ("vk::Format::D32_SFLOAT", "Format::D32Float"),
    ("vk::Format", "Format"),
    ("raster.cull(vk::CullModeFlags::BACK, vk::FrontFace::CLOCKWISE)",
     "raster.cull(Cull::Back, Winding::Clockwise)"),
    ("raster.cull(vk::CullModeFlags::BACK, vk::FrontFace::COUNTER_CLOCKWISE)",
     "raster.cull(Cull::Back, Winding::CounterClockwise)"),
    ("vk::PrimitiveTopology::TRIANGLE_STRIP", "Topology::TriangleStrip"),
    ("vk::PrimitiveTopology::TRIANGLE_LIST", "Topology::TriangleList"),
    ("Stages::DISPATCH", "Stage::COMPUTE"),
    ("Stages::VERTEX", "Stage::VERTEX"),
    ("Stages::FRAGMENT", "Stage::PIXEL"),
    ("Stages::BLIT", "Stage::TRANSFER"),
    ("Stages::INDIRECT", "Stage::INDIRECT"),
    ("Visibility::Device", "Hazard::ALL"),
    ("Visibility::Threadgroup", "Hazard::BUFFERS"),
    ("surface.format", "surface.format()"),
    ("surface.width", "surface.width()"),
    ("surface.height", "surface.height()"),
]

def port(path):
    s = open(path, encoding="utf-8").read()
    s = PRESENT.sub("            resized = surface.present(&device, &drawable);\n", s)
    for old, new in SUBS:
        s = s.replace(old, new)
    # `.width()`/`.height()` must not double up where they were already calls.
    s = s.replace("()()", "()")
    # The barrier argument order flipped to the blog's (before, after, hazard).
    s = re.sub(r"\.barrier\(\s*(Stage::[A-Z_]+),\s*(Stage::[A-Z_]+),", r".barrier(\1, \2,", s)
    open(path, "w", encoding="utf-8").write(s)
    print("ported", path)

for path in sys.argv[1:]:
    port(path)
