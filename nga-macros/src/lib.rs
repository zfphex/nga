use proc_macro::{Delimiter, Group, Ident, Literal, Punct, Spacing, Span, TokenStream, TokenTree};

type Fail = (Span, String);

struct Field {
    name: String,
    tokens: Vec<TokenTree>,
    span: Span,
}

enum Ty {
    Msl(String),
    User(String),
    Array(Box<Ty>, u32),
    Ptr { mutable: bool, pointee: Box<Ty> },
}

fn scalar(name: &str) -> Option<&'static str> {
    Some(match name {
        "f32" => "float",
        "u32" => "uint",
        "i32" => "int",
        "u16" => "ushort",
        "i16" => "short",
        "u8" => "uchar",
        "i8" => "char",
        "u64" => "ulong",
        "i64" => "long",
        "bool" => "bool",
        "f32x2" => "float2",
        "f32x3" => "float3",
        "f32x4" => "float4",
        "u32x2" => "uint2",
        "u32x3" => "uint3",
        "u32x4" => "uint4",
        "i32x2" => "int2",
        "i32x3" => "int3",
        "i32x4" => "int4",
        "f32x3x3" => "float3x3",
        "f32x4x4" => "float4x4",
        "AtomicU32" => "atomic_uint",
        "Commands" => "nga_commands",
        "Pipeline" => "nga_pipeline",
        "Texture2d" => "nga_texture2d",
        "Texture2dW" => "nga_texture2d_w",
        "Texture2dRW" => "nga_texture2d_rw",
        "Texture2dArray" => "nga_texture2d_array",
        "Texture3d" => "nga_texture3d",
        "Texture3dW" => "nga_texture3d_w",
        "Texture3dRW" => "nga_texture3d_rw",
        "TextureCube" => "nga_texturecube",
        "Depth2d" => "nga_depth2d",
        "Sampler" => "nga_sampler",
        "ExecRange" => "nga_exec_range",
        "Draw" => "nga_draw",
        "DrawIndexed" => "nga_draw_indexed",
        "Dispatch" => "nga_dispatch",
        "DispatchThreads" => "nga_dispatch_threads",
        _ => return None,
    })
}

fn compile_error(span: Span, message: &str) -> TokenStream {
    let mut literal = Literal::string(message);
    literal.set_span(span);
    let mut bang = Punct::new('!', Spacing::Alone);
    bang.set_span(span);
    let mut group = Group::new(Delimiter::Brace, TokenStream::from(TokenTree::Literal(literal)));
    group.set_span(span);
    [
        TokenTree::Ident(Ident::new("compile_error", span)),
        TokenTree::Punct(bang),
        TokenTree::Group(group),
    ]
    .into_iter()
    .collect()
}

fn punct(token: Option<&TokenTree>, character: char) -> bool {
    matches!(token, Some(TokenTree::Punct(p)) if p.as_char() == character)
}

fn parse_struct(tokens: &[TokenTree]) -> Result<(Ident, Vec<Field>), Fail> {
    let mut i = 0;
    loop {
        match tokens.get(i) {
            Some(TokenTree::Ident(id)) => match id.to_string().as_str() {
                "struct" => break,
                "enum" | "union" => {
                    return Err((id.span(), "#[gpu_struct] only applies to structs".into()));
                }
                _ => i += 1,
            },
            Some(_) => i += 1,
            None => {
                return Err((Span::call_site(), "expected a struct definition".into()));
            }
        }
    }

    let Some(TokenTree::Ident(name)) = tokens.get(i + 1) else {
        return Err((Span::call_site(), "expected a struct name".into()));
    };
    let body = match tokens.get(i + 2) {
        Some(TokenTree::Group(g)) if g.delimiter() == Delimiter::Brace => g,
        Some(TokenTree::Group(g)) if g.delimiter() == Delimiter::Parenthesis => {
            return Err((g.span(), "tuple structs cannot be shared with MSL".into()));
        }
        Some(t) if punct(Some(t), '<') => {
            return Err((t.span(), "generic structs cannot be shared with MSL".into()));
        }
        _ => return Err((name.span(), "expected a braced struct body".into())),
    };

    let body: Vec<TokenTree> = body.stream().into_iter().collect();
    let mut fields = Vec::new();
    let mut i = 0;
    while i < body.len() {
        while punct(body.get(i), '#') {
            i += 2;
        }
        if matches!(&body[i], TokenTree::Ident(id) if id.to_string() == "pub") {
            i += 1;
            if matches!(body.get(i), Some(TokenTree::Group(g)) if g.delimiter() == Delimiter::Parenthesis)
            {
                i += 1;
            }
        }
        let (TokenTree::Ident(name), true) = (&body[i], punct(body.get(i + 1), ':')) else {
            return Err((body[i].span(), "expected `name: Type`".into()));
        };
        i += 2;
        let start = i;
        let mut depth = 0i32;
        while i < body.len() {
            match &body[i] {
                TokenTree::Punct(p) if p.as_char() == '<' => depth += 1,
                TokenTree::Punct(p) if p.as_char() == '>' => depth -= 1,
                TokenTree::Punct(p) if p.as_char() == ',' && depth == 0 => break,
                _ => {}
            }
            i += 1;
        }
        fields.push(Field {
            name: name.to_string(),
            tokens: body[start..i].to_vec(),
            span: body[start].span(),
        });
        i += 1;
    }
    Ok((name.clone(), fields))
}

fn parse_ty(mut tokens: &[TokenTree], slice_ok: bool) -> Result<Ty, Fail> {
    let Some(first) = tokens.first() else {
        return Err((Span::call_site(), "expected a type".into()));
    };
    let span = first.span();
    while tokens.len() > 3
        && matches!(&tokens[0], TokenTree::Ident(_))
        && punct(tokens.get(1), ':')
        && punct(tokens.get(2), ':')
    {
        tokens = &tokens[3..];
    }

    match &tokens[0] {
        TokenTree::Group(g) if g.delimiter() == Delimiter::Bracket && tokens.len() == 1 => {
            let inner: Vec<TokenTree> = g.stream().into_iter().collect();
            let Some(semi) = inner.iter().position(|t| punct(Some(t), ';')) else {
                if !slice_ok {
                    return Err((span, "unsized fields are not allowed; use [T; N]".into()));
                }
                return parse_ty(&inner, false);
            };
            let [TokenTree::Literal(len)] = &inner[semi + 1..] else {
                return Err((span, "array length must be an integer literal".into()));
            };
            let Ok(len) = len.to_string().parse::<u32>() else {
                return Err((span, "array length must be an integer literal".into()));
            };
            let element = parse_ty(&inner[..semi], false)?;
            if let (Ty::Msl(t), 2..=4) = (&element, len)
                && matches!(
                    t.as_str(),
                    "float" | "uint" | "int" | "ushort" | "short" | "uchar" | "char"
                )
            {
                return Ok(Ty::Msl(format!("packed_{t}{len}")));
            }
            Ok(Ty::Array(Box::new(element), len))
        }
        TokenTree::Ident(id) => {
            let name = id.to_string();
            if tokens.len() == 1 {
                return Ok(match scalar(&name) {
                    Some(msl) => Ty::Msl(msl.into()),
                    None => Ty::User(name),
                });
            }
            let mutable = match name.as_str() {
                "Gpu" => false,
                "GpuMut" => true,
                _ => return Err((span, format!("`{name}` cannot be shared with MSL"))),
            };
            let generics = &tokens[1..];
            if !punct(generics.first(), '<') || !punct(generics.last(), '>') {
                return Err((span, format!("expected `{name}<'static, T>`")));
            }
            let [lifetime, static_, comma, pointee @ ..] = &generics[1..generics.len() - 1] else {
                return Err((span, format!("expected `{name}<'static, T>`")));
            };
            if !punct(Some(lifetime), '\'')
                || !matches!(static_, TokenTree::Ident(id) if id.to_string() == "static")
                || !punct(Some(comma), ',')
            {
                return Err((
                    span,
                    format!("`{name}` fields must use the `'static` lifetime"),
                ));
            }
            match parse_ty(pointee, true)? {
                pointee @ (Ty::Msl(_) | Ty::User(_)) => Ok(Ty::Ptr {
                    mutable,
                    pointee: Box::new(pointee),
                }),
                _ => Err((
                    span,
                    "a GPU pointer must point at a scalar, vector or #[gpu_struct] type".into(),
                )),
            }
        }
        TokenTree::Punct(p) if p.as_char() == '&' => Err((
            span,
            "references cannot be shared with MSL; use Gpu<'static, T>".into(),
        )),
        TokenTree::Punct(p) if p.as_char() == '*' => Err((
            span,
            "raw pointers cannot be shared with MSL; use Gpu<'static, T>".into(),
        )),
        TokenTree::Group(g) if g.delimiter() == Delimiter::Parenthesis => {
            Err((span, "tuple types cannot be shared with MSL".into()))
        }
        _ => Err((span, "this type cannot be shared with MSL".into())),
    }
}

fn declare(ty: &Ty, name: &str) -> String {
    match ty {
        Ty::Msl(t) | Ty::User(t) => format!("{t} {name}"),
        Ty::Ptr { mutable, pointee } => {
            let (Ty::Msl(t) | Ty::User(t)) = &**pointee else {
                unreachable!()
            };
            match mutable {
                true => format!("device {t}* {name}"),
                false => format!("device const {t}* {name}"),
            }
        }
        Ty::Array(element, len) => format!("{}[{len}]", declare(element, name)),
    }
}

fn users(ty: &Ty, out: &mut Vec<String>) {
    match ty {
        Ty::User(t) => out.push(t.clone()),
        Ty::Ptr { pointee, .. } => users(pointee, out),
        Ty::Array(element, _) => users(element, out),
        _ => {}
    }
}

#[proc_macro_attribute]
pub fn gpu_struct(attr: TokenStream, item: TokenStream) -> TokenStream {
    match expand(attr, item.clone()) {
        Ok(expanded) => expanded,
        Err((span, message)) => item
            .into_iter()
            .chain(compile_error(span, &message))
            .collect(),
    }
}

fn expand(attr: TokenStream, item: TokenStream) -> Result<TokenStream, Fail> {
    let tokens: Vec<TokenTree> = item.clone().into_iter().collect();
    let attr: Vec<TokenTree> = attr.into_iter().collect();
    let align = match attr.as_slice() {
        [] => None,
        [TokenTree::Ident(id), TokenTree::Group(g)]
            if id.to_string() == "align" && g.delimiter() == Delimiter::Parenthesis =>
        {
            Some(g.stream().to_string())
        }
        _ => {
            return Err((
                attr[0].span(),
                "#[gpu_struct] takes nothing or `align(N)`".into(),
            ));
        }
    };

    let (name, fields) = parse_struct(&tokens)?;
    let mut decl = match &align {
        Some(align) => format!("\nstruct alignas({align}) {name} {{\n"),
        None => format!("\nstruct {name} {{\n"),
    };
    let mut offsets = String::new();
    let mut nested = Vec::new();
    for field in &fields {
        let ty = match parse_ty(&field.tokens, false) {
            Ok(ty) => ty,
            Err((_, message)) => {
                return Err((field.span, format!("field `{}`: {message}", field.name)));
            }
        };
        users(&ty, &mut nested);
        decl += &format!("    {};\n", declare(&ty, &field.name));
        offsets += &format!(
            "(\"{}\", ::core::mem::offset_of!({name}, {0})),",
            field.name
        );
    }
    decl += "};\n";

    let asserts = nested
        .iter()
        .map(|t| format!("assert::<{t}>();"))
        .collect::<String>();
    let repr = match &align {
        Some(align) => format!("#[repr(C, align({align}))]"),
        None => "#[repr(C)]".into(),
    };

    let out = format!(
        "impl ::nga::DeviceAbi for {name} {{
             const ABI: ::nga::TypeAbi = ::nga::TypeAbi {{
                 name: \"{name}\",
                 decl: {decl:?},
                 size: ::core::mem::size_of::<{name}>(),
                 align: ::core::mem::align_of::<{name}>(),
                 fields: &[{offsets}],
             }};
         }}
         const _: fn() = || {{ fn assert<T: ::nga::DeviceAbi>() {{}} {asserts} }};"
    );
    let prefix: TokenStream = format!("{repr} #[derive(Clone, Copy)]").parse().unwrap();
    Ok(prefix
        .into_iter()
        .chain(item)
        .chain(out.parse::<TokenStream>().unwrap())
        .collect())
}

#[proc_macro_derive(Constants)]
pub fn constants(item: TokenStream) -> TokenStream {
    let tokens: Vec<TokenTree> = item.into_iter().collect();
    let (name, fields) = match parse_struct(&tokens) {
        Ok(parsed) => parsed,
        Err((span, message)) => return compile_error(span, &message),
    };

    let mut format_string = String::new();
    let mut types = String::new();
    let mut sets = String::new();
    for (index, field) in fields.iter().enumerate() {
        let ty: TokenStream = field.tokens.iter().cloned().collect();
        format_string += &format!(
            "constant {{}} {} [[function_constant({index})]];\\n",
            field.name
        );
        types += &format!("<{ty} as ::nga::ConstantValue>::MSL_TYPE,");
        sets += &format!(
            "::nga::set_constant(&values, {index}, self.{});",
            field.name
        );
    }

    format!(
        "impl ::nga::Constants for {name} {{
             fn msl() -> ::std::string::String {{
                 ::std::format!(\"{format_string}\", {types})
             }}
             fn values(&self) -> ::nga::FunctionConstantValues {{
                 let values = ::nga::FunctionConstantValues::new();
                 {sets}
                 values
             }}
         }}"
    )
    .parse()
    .unwrap()
}
