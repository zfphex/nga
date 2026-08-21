mod gpu;

use proc_macro::{Delimiter, Group, Ident, Literal, Punct, Spacing, Span, TokenStream, TokenTree};

type Fail = (Span, String);

struct Field {
    name: String,
    tokens: Vec<TokenTree>,
    span: Span,
}

enum Ty {
    Builtin,
    User(String),
    Array(Box<Ty>),
    Ptr(Box<Ty>),
}

/// Every type name `#[gpu_struct]` knows without being told. Anything else is a user struct,
/// and the expansion asserts it implements `DeviceAbi`. The spellings the two backends give
/// these live in `msl.rs` and `spirv.rs`; all this decides is which of the two kinds a field is.
fn scalar(name: &str) -> bool {
    matches!(
        name,
        "f32" | "f16" | "u32" | "i32" | "u16" | "i16" | "u8" | "i8" | "u64" | "i64" | "bool"
            | "f16x2" | "f16x3" | "f16x4"
            | "f32x2" | "f32x3" | "f32x4"
            | "u32x2" | "u32x3" | "u32x4"
            | "i32x2" | "i32x3" | "i32x4"
            | "f32x3x3" | "f32x4x4"
            | "AtomicU32"
            | "Texture2d" | "Texture2dW" | "Texture2dRW" | "Texture2dArray"
            | "Texture3d" | "Texture3dW" | "Texture3dRW" | "TextureCube"
            | "Depth2d" | "Sampler"
            // Metal-only. Indirect command buffers have no `#[gpu]` spelling, so the shaders
            // that encode through these are the hand-written `.metal` ones.
            | "Commands" | "PipelineHandle"
            | "Draw" | "DrawIndexed" | "Dispatch" | "DispatchThreads"
    )
}

fn compile_error(span: Span, message: &str) -> TokenStream {
    let mut literal = Literal::string(message);
    literal.set_span(span);
    let mut bang = Punct::new('!', Spacing::Alone);
    bang.set_span(span);
    let mut group = Group::new(
        Delimiter::Brace,
        TokenStream::from(TokenTree::Literal(literal)),
    );
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
            return Err((
                g.span(),
                "tuple structs cannot be shared with the GPU".into(),
            ));
        }
        Some(t) if punct(Some(t), '<') => {
            return Err((
                t.span(),
                "generic structs cannot be shared with the GPU".into(),
            ));
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
            if len.to_string().parse::<u32>().is_err() {
                return Err((span, "array length must be an integer literal".into()));
            }
            let element = parse_ty(&inner[..semi], false)?;
            Ok(Ty::Array(Box::new(element)))
        }
        TokenTree::Ident(id) => {
            let name = id.to_string();
            if tokens.len() == 1 {
                return Ok(if scalar(&name) {
                    Ty::Builtin
                } else {
                    Ty::User(name)
                });
            }
            if name != "Gpu" && name != "GpuMut" {
                return Err((span, format!("`{name}` cannot be shared with the GPU")));
            }
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
                pointee @ (Ty::Builtin | Ty::User(_)) => Ok(Ty::Ptr(Box::new(pointee))),
                _ => Err((
                    span,
                    "a GPU pointer must point at a scalar, vector or #[gpu_struct] type".into(),
                )),
            }
        }
        TokenTree::Punct(p) if p.as_char() == '&' => Err((
            span,
            "references cannot be shared with the GPU; use Gpu<'static, T>".into(),
        )),
        TokenTree::Punct(p) if p.as_char() == '*' => Err((
            span,
            "raw pointers cannot be shared with the GPU; use Gpu<'static, T>".into(),
        )),
        TokenTree::Group(g) if g.delimiter() == Delimiter::Parenthesis => {
            Err((span, "tuple types cannot be shared with the GPU".into()))
        }
        _ => Err((span, "this type cannot be shared with the GPU".into())),
    }
}

fn users(ty: &Ty, out: &mut Vec<String>) {
    match ty {
        Ty::User(t) => out.push(t.clone()),
        Ty::Ptr(pointee) => users(pointee, out),
        Ty::Array(element) => users(element, out),
        _ => {}
    }
}

/// A shader function. The body is re-emitted unchanged so `rustc` checks it, and a `GpuFn`
/// const beside it carries the same body as IR for `nga`'s two backends.
#[proc_macro_attribute]
pub fn gpu(attr: TokenStream, item: TokenStream) -> TokenStream {
    gpu::gpu(attr, item)
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
        let ty_tokens: TokenStream = field.tokens.iter().cloned().collect();
        let ir = match gpu::ir_ty(&field.tokens) {
            Ok(ir) => ir,
            Err((_, message)) => {
                return Err((field.span, format!("field `{}`: {message}", field.name)));
            }
        };
        offsets += &format!(
            "::nga::FieldAbi {{
                 name: \"{}\",
                 offset: ::core::mem::offset_of!({name}, {0}),
                 size: ::core::mem::size_of::<{ty_tokens}>(),
                 ty: {ir},
             }},",
            field.name
        );
    }
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

    let mut sets = String::new();
    for (index, field) in fields.iter().enumerate() {
        sets += &format!(
            "::nga::Constant::push(&mut values, {index}, self.{});",
            field.name
        );
    }

    format!(
        "impl ::nga::Constants for {name} {{
             fn values(&self) -> ::std::vec::Vec<::nga::Constant> {{
                 let mut values = ::std::vec::Vec::new();
                 {sets}
                 values
             }}
         }}"
    )
    .parse()
    .unwrap()
}
