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

/// Every type name a shared struct knows without being told. Anything else is a user struct,
/// and the expansion asserts it implements `DeviceAbi`. The spellings the two backends give
/// these live in `msl.rs` and `spirv.rs`; all this decides is which of the two kinds a field is.
fn scalar(name: &str) -> bool {
    matches!(
        name,
        "f32" | "f16" | "u32" | "i32" | "u16" | "i16" | "u8" | "i8" | "u64" | "i64" | "bool"
            | "f16x2" | "f16x3" | "f16x4"
            | "f32x2" | "f32x3" | "f32x4"
            | "packed_f32x3"
            | "u32x2" | "u32x3" | "u32x4"
            | "i32x2" | "i32x3" | "i32x4"
            | "f32x3x3" | "f32x4x4"
            | "AtomicU32"
            | "Texture2d" | "Texture2dW" | "Texture2dRW" | "Texture2dArray"
            | "Texture3d" | "Texture3dW" | "Texture3dRW" | "TextureCube"
            | "Depth2d" | "Sampler"
            // Metal-only handles used by Rust shaders that encode indirect command buffers.
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
                    return Err((
                        id.span(),
                        "#[gpu_mod] shares structs, not enums or unions".into(),
                    ));
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
                    "a GPU pointer must point at a scalar, vector or shared struct".into(),
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

/// A module of shader code: every `struct` in it is shared with the GPU, every `fn` is a shader
/// function, and `ENTRIES` beside them lists the ones that are entry points.
///
/// A proc macro sees one item and can look nothing up, so a struct's field offsets have to be
/// recorded where the struct is written — but the macro that records them does not have to be
/// spelled there. This one is spelled once, on the module, and sees every item inside it:
///
/// ```ignore
/// #[gpu_mod]
/// mod triangle {
///     use nga::*;
///     pub struct Vertex { pub position: f32x3, pub color: f32x3 }
///     #[vertex] fn vertex(root: Gpu<'static, Data>, #[vertex_id] vid: u32) -> VertexOut { .. }
///     #[fragment] fn fragment(input: VertexOut) -> f32x4 { .. }
///     fn saturate(x: f32) -> f32 { .. }   // no stage attribute: a helper
/// }
/// ```
///
/// A `fn` carries `#[vertex]`, `#[fragment]`, `#[compute(x, y, z)]`, `#[object]`,
/// `#[mesh(Vertex, vertices, primitives)]` or `#[tile(Imageblock)]` when it is an entry point
/// and nothing when it is a helper. A `struct` that wants an alignment writes the `#[repr]` Rust
/// already has, and the macro leaves it alone. `#[host]` on any item passes it through
/// untouched, which is where a host-side type or helper goes.
///
/// The module has to be inline: `#[gpu_mod] mod name;` gives the macro a path and no body, and
/// there is no stable way to read the file it names.
#[proc_macro_attribute]
pub fn gpu_mod(attr: TokenStream, item: TokenStream) -> TokenStream {
    match expand_mod(attr, item.clone()) {
        Ok(expanded) => expanded,
        Err((span, message)) => item
            .into_iter()
            .chain(compile_error(span, &message))
            .collect(),
    }
}

/// One item and the attributes in front of it, up to the brace that closes it or the `;` that
/// ends it. Attribute groups are brackets and a body is a brace, so the two never collide.
fn items(body: &[TokenTree]) -> Vec<Vec<TokenTree>> {
    let mut out = Vec::new();
    let mut start = 0;
    let mut i = 0;
    while i < body.len() {
        if punct(body.get(i), '#') {
            // `#![allow(..)]` is the module's, not the next item's, so it stands alone.
            if punct(body.get(i + 1), '!') {
                i += 3;
                out.push(body[start..i].to_vec());
                start = i;
                continue;
            }
            i += 2;
            continue;
        }
        match &body[i] {
            TokenTree::Group(g) if g.delimiter() == Delimiter::Brace => {
                i += 1;
                out.push(body[start..i].to_vec());
                start = i;
            }
            t if punct(Some(t), ';') => {
                i += 1;
                out.push(body[start..i].to_vec());
                start = i;
            }
            _ => i += 1,
        }
    }
    if start < body.len() {
        out.push(body[start..].to_vec());
    }
    out
}

fn expand_mod(attr: TokenStream, item: TokenStream) -> Result<TokenStream, Fail> {
    if let Some(first) = attr.into_iter().next() {
        return Err((first.span(), "#[gpu_mod] takes no arguments".into()));
    }
    let tokens: Vec<TokenTree> = item.into_iter().collect();
    let Some(keyword) = tokens
        .iter()
        .position(|token| matches!(token, TokenTree::Ident(id) if id.to_string() == "mod"))
    else {
        return Err((
            Span::call_site(),
            "#[gpu_mod] only applies to modules".into(),
        ));
    };
    let Some(TokenTree::Ident(name)) = tokens.get(keyword + 1) else {
        return Err((tokens[keyword].span(), "expected a module name".into()));
    };
    let body = match tokens.get(keyword + 2) {
        Some(TokenTree::Group(g)) if g.delimiter() == Delimiter::Brace => g,
        _ => {
            return Err((
                name.span(),
                "#[gpu_mod] needs the module body; a macro cannot read the file `mod name;` names"
                    .into(),
            ));
        }
    };

    // Built as tokens rather than as text: a body's spans are what makes a type error in a
    // shader point at the shader, and printing the module out and parsing it back loses them.
    let mut out = TokenStream::new();
    let mut entries: Vec<String> = Vec::new();
    let body: Vec<TokenTree> = body.stream().into_iter().collect();
    for item in items(&body) {
        // `#![allow(..)]` and friends belong to the module, not to any item.
        if punct(item.first(), '#') && punct(item.get(1), '!') {
            out.extend(item);
            continue;
        }

        // The attributes in front of the item: `#[host]` and the stage ones are ours, and
        // everything else — `#[repr]`, a doc comment, `#[allow]` — belongs to the item.
        let mut kept: Vec<TokenTree> = Vec::new();
        let mut stage: Option<TokenStream> = None;
        let mut host = false;
        let mut repr = true;
        let mut i = 0;
        while punct(item.get(i), '#') {
            let Some(TokenTree::Group(g)) = item.get(i + 1) else {
                return Err((item[i].span(), "expected `#[..]`".into()));
            };
            let inner: Vec<TokenTree> = g.stream().into_iter().collect();
            match inner.first() {
                Some(TokenTree::Ident(id)) => match id.to_string().as_str() {
                    "host" => host = true,
                    "vertex" | "fragment" | "compute" | "object" | "mesh" | "tile" => {
                        stage = Some(g.stream())
                    }
                    // A struct spelling its own `#[repr]` keeps it, but it has to be a `C` one:
                    // the offsets both backends write out are the ones Rust chose, and Rust's
                    // own layout is free to reorder fields.
                    "repr" => {
                        if !inner
                            .iter()
                            .any(|t| matches!(t, TokenTree::Group(g) if g.stream().to_string().split(',').any(|part| part.trim() == "C")))
                        {
                            return Err((
                                item[i].span(),
                                "a shared struct needs `#[repr(C)]`, or `#[repr(C, align(N))]`"
                                    .into(),
                            ));
                        }
                        repr = false;
                        kept.extend_from_slice(&item[i..i + 2]);
                    }
                    _ => kept.extend_from_slice(&item[i..i + 2]),
                },
                _ => kept.extend_from_slice(&item[i..i + 2]),
            }
            i += 2;
        }

        let rest: Vec<TokenTree> = item[i..].to_vec();
        // What kind of item it is, past the visibility.
        let mut j = 0;
        if matches!(rest.get(j), Some(TokenTree::Ident(id)) if id.to_string() == "pub") {
            j += 1;
            if matches!(rest.get(j), Some(TokenTree::Group(g)) if g.delimiter() == Delimiter::Parenthesis)
            {
                j += 1;
            }
        }
        let kind = match rest.get(j) {
            Some(TokenTree::Ident(id)) => id.to_string(),
            _ => String::new(),
        };
        // `unsafe fn`, `const fn` and the rest would otherwise read as an item kind of their
        // own and pass through untagged, which fails much further away.
        let mut k = j;
        while matches!(rest.get(k), Some(TokenTree::Ident(id))
            if matches!(id.to_string().as_str(), "unsafe" | "const" | "async" | "extern"))
            || matches!(rest.get(k), Some(TokenTree::Literal(_)))
        {
            k += 1;
        }
        if k > j && matches!(rest.get(k), Some(TokenTree::Ident(id)) if id.to_string() == "fn") {
            return Err((
                rest[j].span(),
                format!("a shader function cannot be `{kind}`"),
            ));
        }

        let attrs: TokenStream = kept.into_iter().collect();
        let rest: TokenStream = rest.into_iter().collect();
        let expanded = match (host, kind.as_str()) {
            (false, "struct") => expand_struct(rest, repr)?,
            (false, "fn") => {
                if stage.is_some() {
                    let Some(TokenTree::Ident(name)) = item.get(i + j + 1) else {
                        return Err((item[i + j].span(), "expected a function name".into()));
                    };
                    entries.push(name.to_string().to_uppercase());
                }
                gpu::expand(stage.unwrap_or_default(), rest)?
            }
            _ => rest,
        };
        out.extend(attrs);
        out.extend(expanded);
    }

    let list = format!(
        "#[allow(dead_code)]
         pub const ENTRIES: &[&::nga::ir::GpuFn] = &[{}];",
        entries
            .iter()
            .map(|e| format!("&{e}"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    out.extend(list.parse::<TokenStream>().map_err(|_| {
        (
            name.span(),
            format!("could not emit the entry list:\n{list}"),
        )
    })?);

    let prefix: TokenStream = tokens[..keyword + 2].iter().cloned().collect();
    Ok(prefix
        .into_iter()
        .chain([TokenTree::Group(Group::new(Delimiter::Brace, out))])
        .collect())
}

/// The `DeviceAbi` impl, and the `#[repr(C)]` and `#[derive(Clone, Copy)]` a shared struct
/// needs, unless `repr` is false because the struct already spells its own.
fn expand_struct(item: TokenStream, repr: bool) -> Result<TokenStream, Fail> {
    let tokens: Vec<TokenTree> = item.clone().into_iter().collect();
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
    let prefix: TokenStream = format!(
        "{} #[derive(Clone, Copy)]",
        if repr { "#[repr(C)]" } else { "" }
    )
    .parse()
    .unwrap();
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
