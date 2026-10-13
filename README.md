This library abstracts existing graphics APIs (Metal and Vulkan) into a Rust safe GPU compute library.
It is loosley based on Sebastian Aaltonen's blog post [No Graphics API](https://www.sebastianaaltonen.com/blog/no-graphics-api).

TL;DR the current Graphics APIs (DirectX, OpenGL, Vulkan, Metal) were all originally designed for very old GPUs and as they modernised, they have all converged to a similar architecuture.
This library attempts to remove the driver overhead typically associated with most graphics APIs.

Currently only Apple M series and Nvidia Pascal (or later) are supported.

I would like to note that this library is basically the complete opposite of WebGPU. WebGPU supports no modern features and trys to cover the widest possible range of devices. It is genuinely an awful choice for any modern application. It is also not a graphics API and neither is this. These are both abstractions over existing APIs. Unfortunately GPU manufacturers are able to abuse their monopolies to keep their instruction set architectures and drivers proprietary. This makes it impossible to write a compiler that targets different GPUs like we do for CPUs. In order to maximise profits, GPU manufacturers vendor lock GPU code so that is unportable as possible, even if a competitor has a superior product, the difficulty of porting code removes any benifit.

Shaders are written as Rust code and compiled to either Metal Shading Language or Standard Portable Intermediate Representation (SPIRV). Despite SPIRV having "Standard" and "Portable" in the name, it's not supported on Apple silicon (thank you Apple).

Here is an example Rust shader module.

```rust
#[gpu_mod]
mod triangle {
    use nga::*;

    pub struct Vertex {
        pub position: f32x3,
        pub color: f32x3,
    }

    #[repr(C, align(16))]
    pub struct Data {
        pub vertices: Gpu<[Vertex]>,
        pub mvp: f32x4x4,
    }

    pub struct VertexOut {
        pub position: f32x4,
        pub color: f32x3,
    }

    #[vertex]
    fn vertex(root: Gpu<Data>, #[vertex_id] vid: u32) -> VertexOut {
        let v = root.vertices[vid];
        VertexOut {
            position: root
                .mvp
                .mul(f32x4::new(v.position.x, v.position.y, v.position.z, 1.0)),
            color: v.color,
        }
    }

    #[fragment]
    fn fragment(input: VertexOut) -> f32x4 {
        f32x4::new(input.color.x, input.color.y, input.color.z, 1.0)
    }
}
```

The macro expands to a set of constants that you can pass into `nga`:

```rust
    // Create the window and gpu device.
    // Then build the pipeline
    let mut shaders = device.shaders(triangle::MODULE)?;
    let pipeline = shaders
        .raster(triangle::VERTEX, triangle::FRAGMENT)
        .color(surface.format())
        .build()?;

    // This is a bit verbose but basically you can just pass any data that is GPU 
    // allocated straight into the shader. 
    let mesh = Arena::new(&device, 64 * 1024, Memory::Device)?;
    let vertices = mesh.alloc_slice(&triangle).unwrap();
    raster
        .draw(
            pipeline,
            Data {
                vertices: vertices.gpu,
                mvp: f32x4x4([
                    [cos * aspect, sin, 0.0, 0.0],
                    [-sin * aspect, cos, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]),
            },
            0..3,
        )
        .unwrap();
```

See this [Triangle Example](nga/examples/triangle.rs) for a full example.

This library is a work in progress and the API is currently quite awful since it was designed by a Clanker. The Vulkan and Metal backends are almost 10k lines of code and it's not an easy task to build something that works well for both platforms.

I never wanted to write this library. 

In order to write high performance cross-platform GPU code I basically had no choice.

## GPU timings

Call `timers.begin(&frame)` after `context.begin_frame()`. Results belong to the previous use of that frame slot, `FRAMES_IN_FLIGHT` frames earlier.

Use `timers.checked_ms(start, end)` or `checked_ns(start, end)` for measurements. They return `None` for missing timestamps, out-of-order values (including a detected counter wrap), or indices outside the timer set. Equal valid timestamps return `Some(0.0)`. The existing `ms` and `ns` methods remain available and still return zero for invalid intervals.

Record a begin/end pair within each pass rather than deriving a pass's duration from adjacent passes. Raster timestamps target the fragment stage on Metal and color output on Vulkan; compute timestamps wait for prior work. Passes and stages can overlap, and Metal timestamps do not delay subsequent work, so summing these intervals is not a whole-frame timing or proof of a 120 FPS budget. Precise timestamps can themselves affect performance.

## Block-compressed textures

`Format` supports BC1–BC7 and ASTC 4×4 LDR, including their sRGB variants where applicable. Query `Feature::TextureCompressionBc` or `Feature::TextureCompressionAstc` with `device.supports` before use. Unsupported compression formats and compressed render-target, storage, or multisample textures return an allocation error. Other ASTC block footprints and ASTC HDR are not exposed yet.

`format.block_extent()` gives the block dimensions in texels and `block_size()` gives bytes per block. `row_size(width)` and `image_size(width, height)` give tightly packed transfer sizes, rounding partial blocks up, including mip levels smaller than a block. `size()` remains bytes per texel for uncompressed formats and panics for compressed formats.

`heap.upload_layer(&texture, level, layer, bytes)` expects exactly `format.image_size(mip_width, mip_height)` bytes of already-compressed data, not decoded texels. The same block layout applies to `compute.read_texture_at`. Allocate sufficient readback space and align compressed readback offsets to `block_size()` bytes. Vulkan streams large uploads in complete block rows; the final copy can contain a partial block row at the mip boundary.