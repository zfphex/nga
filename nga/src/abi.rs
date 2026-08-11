use minmetal::{DataType, FunctionConstantValues};

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

pub trait ConstantValue: Copy {
    const DATA_TYPE: DataType;
    const MSL_TYPE: &'static str;
}

pub fn set_constant<T: ConstantValue>(values: &FunctionConstantValues, index: usize, value: T) {
    let bytes =
        unsafe { std::slice::from_raw_parts(&value as *const T as *const u8, size_of::<T>()) };
    values.set_bytes(index, T::DATA_TYPE, bytes);
}

pub trait Constants {
    const FIELDS: &'static [(&'static str, &'static str)];

    fn values(&self) -> FunctionConstantValues;

    fn msl() -> String {
        Self::FIELDS
            .iter()
            .enumerate()
            .map(|(index, (name, ty))| {
                format!("constant {ty} {name} [[function_constant({index})]];\n")
            })
            .collect()
    }
}

macro_rules! vector {
    ($($name:ident: [$element:ty; $count:literal] align $align:tt = $msl:literal @ $data:ident,)*) => {
        $(
            #[repr(C, align($align))]
            #[derive(Clone, Copy, Debug, PartialEq)]
            pub struct $name(pub [$element; $count]);

            impl ConstantValue for $name {
                const DATA_TYPE: DataType = DataType::$data;
                const MSL_TYPE: &'static str = $msl;
            }
        )*
    };
}

vector! {
    f32x2: [f32; 2] align 8 = "float2" @ Float2,
    f32x3: [f32; 3] align 16 = "float3" @ Float3,
    f32x4: [f32; 4] align 16 = "float4" @ Float4,
    u32x2: [u32; 2] align 8 = "uint2" @ UInt2,
    u32x3: [u32; 3] align 16 = "uint3" @ UInt3,
    u32x4: [u32; 4] align 16 = "uint4" @ UInt4,
    i32x2: [i32; 2] align 8 = "int2" @ Int2,
    i32x3: [i32; 3] align 16 = "int3" @ Int3,
    i32x4: [i32; 4] align 16 = "int4" @ Int4,
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct f32x3x3(pub [f32x3; 3]);

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct f32x4x4(pub [[f32; 4]; 4]);

impl ConstantValue for bool {
    const DATA_TYPE: DataType = DataType::Bool;
    const MSL_TYPE: &'static str = "bool";
}

impl ConstantValue for f32 {
    const DATA_TYPE: DataType = DataType::Float;
    const MSL_TYPE: &'static str = "float";
}

impl ConstantValue for u32 {
    const DATA_TYPE: DataType = DataType::UInt;
    const MSL_TYPE: &'static str = "uint";
}

impl ConstantValue for i32 {
    const DATA_TYPE: DataType = DataType::Int;
    const MSL_TYPE: &'static str = "int";
}
