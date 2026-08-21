//! The vector and matrix types a shader is written in, with real host bodies.
//!
//! Nothing here runs on a GPU. It exists so that a `#[gpu]` function is ordinary Rust that
//! `rustc` type-checks, which is what replaces `slangc`'s diagnostics. Every operation is also
//! an `ir::Intrinsic`, so the two backends know what to emit for it; a method with no intrinsic
//! is a method a shader cannot call.
//!
//! Component access is a field for one component (`v.x`) and a method for more (`v.xyz()`),
//! because Rust has no multi-field access and inventing one would mean the body stopped
//! compiling.

use crate::{Sampler, Scalar};

macro_rules! vector {
    ($($name:ident: $element:ty as $scalar:ident, align $align:tt, $($field:ident),+;)*) => {
        $(
            #[repr(C, align($align))]
            #[derive(Clone, Copy, Debug, Default, PartialEq)]
            #[allow(non_camel_case_types)]
            pub struct $name { $(pub $field: $element),+ }

            /// The array form the host used before the fields existed, kept so a struct
            /// literal reads the same on both sides of the ABI.
            #[allow(non_snake_case)]
            pub const fn $name(components: [$element; count!($($field)+)]) -> $name {
                let [$($field),+] = components;
                $name { $($field),+ }
            }

            impl $name {
                pub const fn new($($field: $element),+) -> Self {
                    Self { $($field),+ }
                }

                pub const fn splat(value: $element) -> Self {
                    Self { $($field: value),+ }
                }

                pub fn min(self, other: Self) -> Self {
                    Self { $($field: if other.$field < self.$field { other.$field } else { self.$field }),+ }
                }

                pub fn max(self, other: Self) -> Self {
                    Self { $($field: if other.$field > self.$field { other.$field } else { self.$field }),+ }
                }

                pub fn clamp(self, low: Self, high: Self) -> Self {
                    self.max(low).min(high)
                }

                pub fn select(self, mask: bool, other: Self) -> Self {
                    if mask { self } else { other }
                }
            }

            impl core::ops::Add for $name {
                type Output = Self;
                fn add(self, other: Self) -> Self { Self { $($field: self.$field + other.$field),+ } }
            }

            impl core::ops::Sub for $name {
                type Output = Self;
                fn sub(self, other: Self) -> Self { Self { $($field: self.$field - other.$field),+ } }
            }

            impl core::ops::Mul for $name {
                type Output = Self;
                fn mul(self, other: Self) -> Self { Self { $($field: self.$field * other.$field),+ } }
            }

            impl core::ops::Div for $name {
                type Output = Self;
                fn div(self, other: Self) -> Self { Self { $($field: self.$field / other.$field),+ } }
            }

            impl core::ops::Mul<$element> for $name {
                type Output = Self;
                fn mul(self, other: $element) -> Self { Self { $($field: self.$field * other),+ } }
            }

            impl core::ops::Div<$element> for $name {
                type Output = Self;
                fn div(self, other: $element) -> Self { Self { $($field: self.$field / other),+ } }
            }

            impl core::ops::Add<$element> for $name {
                type Output = Self;
                fn add(self, other: $element) -> Self { Self { $($field: self.$field + other),+ } }
            }

            impl core::ops::Sub<$element> for $name {
                type Output = Self;
                fn sub(self, other: $element) -> Self { Self { $($field: self.$field - other),+ } }
            }

            impl core::ops::Mul<$name> for $element {
                type Output = $name;
                fn mul(self, other: $name) -> $name { other * self }
            }

            impl core::ops::AddAssign for $name {
                fn add_assign(&mut self, other: Self) { *self = *self + other; }
            }

            impl core::ops::SubAssign for $name {
                fn sub_assign(&mut self, other: Self) { *self = *self - other; }
            }

            impl core::ops::MulAssign<$element> for $name {
                fn mul_assign(&mut self, other: $element) { *self = *self * other; }
            }

            impl crate::ConstantValue for $name {
                const SCALAR: Scalar = Scalar::$scalar;
            }
        )*
    };
}

macro_rules! count {
    ($($field:ident)+) => { [$(count!(@one $field)),+].len() };
    (@one $field:ident) => { () };
}

vector! {
    // A three-vector is twelve bytes and four-byte aligned, not sixteen: that is what
    // `packed_float3` is on Metal and what scalar block layout gives on SPIR-V. Rust's own
    // `align(16)` would have made the host struct disagree with both.
    f32x2: f32 as F32x2, align 8, x, y;
    f32x3: f32 as F32x3, align 4, x, y, z;
    f32x4: f32 as F32x4, align 16, x, y, z, w;
    u32x2: u32 as U32x2, align 8, x, y;
    u32x3: u32 as U32x3, align 4, x, y, z;
    u32x4: u32 as U32x4, align 16, x, y, z, w;
    i32x2: i32 as I32x2, align 8, x, y;
    i32x3: i32 as I32x3, align 4, x, y, z;
    i32x4: i32 as I32x4, align 16, x, y, z, w;
}

/// The float-only half of the vocabulary. Written out per width because the component count is
/// part of every one of these.
macro_rules! float_vector {
    ($($name:ident, $($field:ident),+;)*) => {
        $(
            impl $name {
                pub fn dot(self, other: Self) -> f32 {
                    0.0 $(+ self.$field * other.$field)+
                }

                pub fn length(self) -> f32 {
                    self.dot(self).sqrt()
                }

                pub fn distance(self, other: Self) -> f32 {
                    (self - other).length()
                }

                pub fn normalize(self) -> Self {
                    self / self.length()
                }

                pub fn abs(self) -> Self { Self { $($field: self.$field.abs()),+ } }
                pub fn floor(self) -> Self { Self { $($field: self.$field.floor()),+ } }
                pub fn ceil(self) -> Self { Self { $($field: self.$field.ceil()),+ } }
                pub fn round(self) -> Self { Self { $($field: self.$field.round()),+ } }
                pub fn fract(self) -> Self { Self { $($field: self.$field.fract()),+ } }
                pub fn sqrt(self) -> Self { Self { $($field: self.$field.sqrt()),+ } }
                pub fn sign(self) -> Self { Self { $($field: self.$field.signum()),+ } }
                pub fn exp(self) -> Self { Self { $($field: self.$field.exp()),+ } }
                pub fn log(self) -> Self { Self { $($field: self.$field.ln()),+ } }
                pub fn sin(self) -> Self { Self { $($field: self.$field.sin()),+ } }
                pub fn cos(self) -> Self { Self { $($field: self.$field.cos()),+ } }

                pub fn pow(self, other: Self) -> Self {
                    Self { $($field: self.$field.powf(other.$field)),+ }
                }

                pub fn mix(self, other: Self, t: f32) -> Self {
                    self + (other - self) * t
                }

                pub fn step(self, edge: Self) -> Self {
                    Self { $($field: (self.$field >= edge.$field) as u32 as f32),+ }
                }

                pub fn smoothstep(self, low: Self, high: Self) -> Self {
                    let t = ((self - low) / (high - low)).clamp(Self::splat(0.0), Self::splat(1.0));
                    t * t * (Self::splat(3.0) - t * 2.0)
                }

                pub fn reflect(self, normal: Self) -> Self {
                    self - normal * (2.0 * self.dot(normal))
                }
            }

            impl core::ops::Neg for $name {
                type Output = Self;
                fn neg(self) -> Self { Self { $($field: -self.$field),+ } }
            }
        )*
    };
}

float_vector! {
    f32x2, x, y;
    f32x3, x, y, z;
    f32x4, x, y, z, w;
}

impl f32x3 {
    pub fn cross(self, other: f32x3) -> f32x3 {
        f32x3::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }

    pub fn xy(self) -> f32x2 {
        f32x2::new(self.x, self.y)
    }
}

impl f32x4 {
    pub fn xy(self) -> f32x2 {
        f32x2::new(self.x, self.y)
    }

    pub fn xyz(self) -> f32x3 {
        f32x3::new(self.x, self.y, self.z)
    }

    pub fn zw(self) -> f32x2 {
        f32x2::new(self.z, self.w)
    }
}

impl u32x3 {
    pub fn xy(self) -> u32x2 {
        u32x2::new(self.x, self.y)
    }
}

impl u32x4 {
    pub fn xy(self) -> u32x2 {
        u32x2::new(self.x, self.y)
    }

    pub fn xyz(self) -> u32x3 {
        u32x3::new(self.x, self.y, self.z)
    }
}

/// Column-major, the layout both backends and every host matrix library agree on.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq)]
#[allow(non_camel_case_types)]
pub struct f32x4x4(pub [[f32; 4]; 4]);

impl f32x4x4 {
    pub fn mul(self, other: f32x4) -> f32x4 {
        let mut out = [0.0f32; 4];
        for row in 0..4 {
            for column in 0..4 {
                out[row] += self.0[column][row] * [other.x, other.y, other.z, other.w][column];
            }
        }
        f32x4(out)
    }

    pub fn transpose(self) -> f32x4x4 {
        let mut out = [[0.0f32; 4]; 4];
        for row in 0..4 {
            for column in 0..4 {
                out[row][column] = self.0[column][row];
            }
        }
        f32x4x4(out)
    }
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq)]
#[allow(non_camel_case_types)]
pub struct f32x3x3(pub [f32x3; 3]);

impl f32x3x3 {
    pub fn mul(self, other: f32x3) -> f32x3 {
        self.0[0] * other.x + self.0[1] * other.y + self.0[2] * other.z
    }
}

// -- everything that only means something on a device ------------------------------------

/// A threadgroup barrier. Nothing to do on the host; the node it parses to is the point.
pub fn barrier() {}

/// Ends a fragment invocation without writing. Diverges on a device, returns here.
pub fn discard() {}

macro_rules! sampled {
    ($($name:ident,)*) => {
        $(
            impl crate::$name {
                pub fn sample(self, sampler: Sampler, uv: f32x2) -> f32x4 {
                    let _ = (self, sampler, uv);
                    f32x4::splat(0.0)
                }

                pub fn sample_level(self, sampler: Sampler, uv: f32x2, level: f32) -> f32x4 {
                    let _ = (self, sampler, uv, level);
                    f32x4::splat(0.0)
                }

                pub fn load(self, at: u32x2) -> f32x4 {
                    let _ = (self, at);
                    f32x4::splat(0.0)
                }

                pub fn dimensions(self) -> u32x2 {
                    let _ = self;
                    u32x2::splat(0)
                }
            }
        )*
    };
}

sampled! {
    Texture2d,
    Texture2dW,
    Texture2dRW,
    Texture2dArray,
    Texture3d,
    Texture3dW,
    Texture3dRW,
    TextureCube,
    Depth2d,
}

impl crate::Texture2dRW {
    pub fn store(self, at: u32x2, value: f32x4) {
        let _ = (self, at, value);
    }
}

impl crate::Texture2dW {
    pub fn store(self, at: u32x2, value: f32x4) {
        let _ = (self, at, value);
    }
}
