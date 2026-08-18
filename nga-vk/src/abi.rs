use ash::vk;

pub struct TypeAbi {
    pub name: &'static str,
    pub decl: &'static str,
    pub size: usize,
    pub align: usize,
    pub fields: &'static [(&'static str, usize)],
}

pub trait DeviceAbi: Copy {
    const ABI: TypeAbi;
}

/// Metal function constants become SPIR-V specialization constants: the index in
/// `set_constant(values, index, ..)` is the `SpecId` slangc emits for `[SpecializationConstant]`.
pub struct Specialization {
    pub entries: Vec<vk::SpecializationMapEntry>,
    pub data: Vec<u8>,
}

impl Specialization {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            data: Vec::new(),
        }
    }
}

impl Default for Specialization {
    fn default() -> Self {
        Self::new()
    }
}

pub trait ConstantValue: Copy {
    const SLANG_TYPE: &'static str;
}

pub fn set_constant<T: ConstantValue>(values: &mut Specialization, index: u32, value: T) {
    let bytes =
        unsafe { std::slice::from_raw_parts(&value as *const T as *const u8, size_of::<T>()) };
    values.entries.push(
        vk::SpecializationMapEntry::default()
            .constant_id(index)
            .offset(values.data.len() as u32)
            .size(size_of::<T>()),
    );
    values.data.extend_from_slice(bytes);
}

pub trait Constants {
    fn slang() -> String
    where
        Self: Sized;

    fn values(&self) -> Specialization;
}

macro_rules! vector {
    ($($name:ident: [$element:ty; $count:literal] align $align:tt = $slang:literal,)*) => {
        $(
            #[repr(C, align($align))]
            #[derive(Clone, Copy, Debug, PartialEq)]
            pub struct $name(pub [$element; $count]);

            impl ConstantValue for $name {
                const SLANG_TYPE: &'static str = $slang;
            }
        )*
    };
}

vector! {
    f32x2: [f32; 2] align 8 = "float2",
    f32x3: [f32; 3] align 16 = "float3",
    f32x4: [f32; 4] align 16 = "float4",
    u32x2: [u32; 2] align 8 = "uint2",
    u32x3: [u32; 3] align 16 = "uint3",
    u32x4: [u32; 4] align 16 = "uint4",
    i32x2: [i32; 2] align 8 = "int2",
    i32x3: [i32; 3] align 16 = "int3",
    i32x4: [i32; 4] align 16 = "int4",
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct f32x3x3(pub [f32x3; 3]);

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct f32x4x4(pub [[f32; 4]; 4]);

impl ConstantValue for bool {
    const SLANG_TYPE: &'static str = "bool";
}

impl ConstantValue for f32 {
    const SLANG_TYPE: &'static str = "float";
}

impl ConstantValue for u32 {
    const SLANG_TYPE: &'static str = "uint";
}

impl ConstantValue for i32 {
    const SLANG_TYPE: &'static str = "int";
}

impl ConstantValue for u64 {
    const SLANG_TYPE: &'static str = "uint64_t";
}
