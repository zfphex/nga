use crate::{Fail, punct};
use proc_macro::{Delimiter, Spacing, Span, TokenStream, TokenTree};

pub fn ir_ty(tokens: &[TokenTree]) -> Result<String, Fail> {
    ir_ty_collect(tokens, &mut Vec::new())
}

pub fn ir_ty_collect(mut tokens: &[TokenTree], users: &mut Vec<String>) -> Result<String, Fail> {
    let Some(first) = tokens.first() else {
        return Ok("::nga::ir::Ty::Void".into());
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
        TokenTree::Group(g) if g.delimiter() == Delimiter::Parenthesis && g.stream().is_empty() => {
            Ok("::nga::ir::Ty::Void".into())
        }
        TokenTree::Group(g) if g.delimiter() == Delimiter::Bracket && tokens.len() == 1 => {
            let inner: Vec<TokenTree> = g.stream().into_iter().collect();
            let Some(semi) = inner.iter().position(|t| punct(Some(t), ';')) else {
                return ir_ty_collect(&inner, users);
            };
            let [TokenTree::Literal(len)] = &inner[semi + 1..] else {
                return Err((span, "array length must be an integer literal".into()));
            };
            let Ok(len) = len.to_string().parse::<u32>() else {
                return Err((span, "array length must be an integer literal".into()));
            };
            Ok(format!(
                "::nga::ir::Ty::Array(&{}, {len})",
                ir_ty_collect(&inner[..semi], users)?
            ))
        }
        TokenTree::Ident(id) => {
            let name = id.to_string();
            if tokens.len() == 1 {
                return Ok(named_ty(&name, users));
            }
            let mutable = match name.as_str() {
                "Gpu" => "false",
                "GpuRw" => "true",
                _ => return Err((span, format!("`{name}` cannot cross to the GPU"))),
            };
            let generics = &tokens[1..];
            if !punct(generics.first(), '<') || !punct(generics.last(), '>') {
                return Err((span, format!("expected `{name}<T>`")));
            }
            let pointee = &generics[1..generics.len() - 1];
            if pointee.first().is_some_and(|token| punct(Some(token), '\'')) {
                return Err((
                    span,
                    format!("`{name}` carries no lifetime; write `{name}<T>`"),
                ));
            }
            if pointee.is_empty() {
                return Err((span, format!("expected `{name}<T>`")));
            }
            Ok(format!(
                "::nga::ir::Ty::Ptr {{ mutable: {mutable}, pointee: &{} }}",
                ir_ty_collect(pointee, users)?
            ))
        }
        TokenTree::Punct(p) if p.as_char() == '&' => Err((
            span,
            "references cannot cross to the GPU; use Gpu<T>".into(),
        )),
        TokenTree::Punct(p) if p.as_char() == '*' => Err((
            span,
            "raw pointers cannot be shared with the GPU; use Gpu<T>".into(),
        )),
        TokenTree::Group(g) if g.delimiter() == Delimiter::Parenthesis => {
            Err((span, "tuple types cannot be shared with the GPU".into()))
        }
        _ => Err((span, "this type cannot cross to the GPU".into())),
    }
}

fn named_ty(name: &str, users: &mut Vec<String>) -> String {
    let prim = |p: &str| format!("::nga::ir::Ty::Scalar(::nga::ir::Prim::{p})");
    let vector = |p: &str, n: u8| format!("::nga::ir::Ty::Vec(::nga::ir::Prim::{p}, {n})");
    let handle = |k: &str| format!("::nga::ir::Ty::Handle(::nga::ir::HandleKind::{k})");
    match name {
        "f32" => prim("F32"),
        "f16" => prim("F16"),
        "i32" => prim("I32"),
        "u32" => prim("U32"),
        "i64" => prim("I64"),
        "u64" => prim("U64"),
        "i16" => prim("I16"),
        "u16" => prim("U16"),
        "i8" => prim("I8"),
        "u8" => prim("U8"),
        "bool" => prim("Bool"),

        "usize" => prim("U32"),
        "packed_f32x3" => "::nga::ir::Ty::PackedVec(::nga::ir::Prim::F32, 3)".into(),
        "f32x3x3" => "::nga::ir::Ty::Mat(3, 3)".into(),
        "f32x4x4" => "::nga::ir::Ty::Mat(4, 4)".into(),

        "AtomicU32" => prim("U32"),
        "ThreadgroupAtomicU32" => "::nga::ir::Ty::ThreadgroupAtomicU32".into(),
        "Texture2d" => handle("Texture2d"),
        "Texture2dW" | "Texture2dRW" => handle("Texture2dRW"),
        "Texture3d" => handle("Texture3d"),
        "Texture3dW" | "Texture3dRW" => handle("Texture3dRW"),
        "Texture2dArray" => handle("Texture2dArray"),
        "TextureCube" => handle("TextureCube"),
        "TextureCubeRW" => handle("TextureCubeRW"),
        "Depth2d" => handle("Depth2d"),
        "Sampler" => handle("Sampler"),
        "Commands" => handle("Commands"),
        "PipelineHandle" => handle("Pipeline"),
        "Draw" | "DrawIndexed" | "Dispatch" | "DispatchThreads" => handle("Opaque"),
        _ => {
            if let Some((prim, count_str)) = name.split_once('x')
                && let Ok(count @ 2..=4) = count_str.parse::<u8>()
                && let Some(p) = match prim {
                    "f32" => Some("F32"),
                    "f16" => Some("F16"),
                    "i32" => Some("I32"),
                    "u32" => Some("U32"),
                    "u16" => Some("U16"),
                    "i16" => Some("I16"),
                    "u8" => Some("U8"),
                    "i8" => Some("I8"),
                    _ => None,
                }
            {
                vector(p, count)
            } else {
                users.push(name.to_string());
                format!("::nga::ir::Ty::Struct(&<{name} as ::nga::DeviceAbi>::ABI)")
            }
        }
    }
}

fn intrinsic(name: &str) -> Option<&'static str> {
    Some(match name {
        "unpack" => "Unpack",
        "sqrt" => "Sqrt",
        "inverse_sqrt" | "rsqrt" => "InverseSqrt",
        "abs" => "Abs",
        "signum" | "sign" => "Sign",
        "floor" => "Floor",
        "ceil" => "Ceil",
        "round" => "Round",
        "fract" => "Fract",
        "sin" => "Sin",
        "cos" => "Cos",
        "tan" => "Tan",
        "asin" => "Asin",
        "acos" => "Acos",
        "atan" => "Atan",
        "atan2" => "Atan2",
        "exp" => "Exp",
        "exp2" => "Exp2",
        "ln" | "log" => "Log",
        "log2" => "Log2",
        "powf" | "powi" | "pow" => "Pow",
        "min" => "Min",
        "max" => "Max",
        "clamp" => "Clamp",
        "mix" | "lerp" => "Mix",
        "step" => "Step",
        "smoothstep" => "SmoothStep",
        "dot" => "Dot",
        "cross" => "Cross",
        "normalize" => "Normalize",
        "length" => "Length",
        "distance" => "Distance",
        "reflect" => "Reflect",
        "select" => "Select",
        "any" => "Any",
        "all" => "All",
        "mul" | "mul_mat" => "Mul",
        "transpose" => "Transpose",
        "barrier" => "Barrier",
        "threadgroup_barrier" => "ThreadgroupBarrier",
        "atomic_add" => "AtomicAdd",
        "atomic_min" => "AtomicMin",
        "atomic_max" => "AtomicMax",
        "atomic_exchange" => "AtomicExchange",
        "offset" => "Offset",
        "sample" => "Sample",
        "sample_level" => "SampleLevel",
        "load" => "Load",
        "store" => "Store",
        "dimensions" => "Dimensions",
        "dfdx" => "Dfdx",
        "dfdy" => "Dfdy",
        "mesh_set_threadgroups" => "MeshSetThreadgroups",
        "mesh_set_primitive_count" => "MeshSetPrimitiveCount",
        "mesh_set_vertex" => "MeshSetVertex",
        "mesh_set_index" => "MeshSetIndex",
        "tile_read" => "TileRead",
        "tile_write" => "TileWrite",
        "command_set_pipeline" => "CommandSetPipeline",
        "command_set_vertex_buffer" => "CommandSetVertexBuffer",
        "command_draw" => "CommandDraw",
        _ => return None,
    })
}

fn binop(t: &[TokenTree], i: usize) -> Option<(&'static str, u8, usize)> {
    let TokenTree::Punct(p) = t.get(i)? else {
        return None;
    };
    if p.spacing() == Spacing::Joint
        && let Some(TokenTree::Punct(q)) = t.get(i + 1)
    {

        if q.as_char() == '=' && !matches!(p.as_char(), '=' | '!' | '<' | '>') {
            return None;
        }
        let two = match (p.as_char(), q.as_char()) {
            ('&', '&') => ("AndAnd", 1),
            ('|', '|') => ("OrOr", 2),
            ('=', '=') => ("Eq", 4),
            ('!', '=') => ("Ne", 4),
            ('<', '=') => ("Le", 4),
            ('>', '=') => ("Ge", 4),
            ('<', '<') => ("Shl", 8),
            ('>', '>') => ("Shr", 8),
            _ => ("", 0),
        };
        if two.1 != 0 {
            return Some((two.0, two.1, 2));
        }
    }
    let one = match p.as_char() {
        '<' => ("Lt", 4),
        '>' => ("Gt", 4),
        '|' => ("Or", 5),
        '^' => ("Xor", 6),
        '&' => ("And", 7),
        '+' => ("Add", 9),
        '-' => ("Sub", 9),
        '*' => ("Mul", 10),
        '/' => ("Div", 10),
        '%' => ("Rem", 10),
        _ => return None,
    };
    Some((one.0, one.1, 1))
}

struct B {
    nodes: Vec<String>,
    locals: Vec<String>,

    scope: Vec<(String, u32)>,
    params: Vec<String>,
    calls: Vec<String>,

    no_struct: bool,
}

impl B {
    fn push(&mut self, node: String) -> u32 {
        self.nodes.push(node);
        self.nodes.len() as u32 - 1
    }

    fn local(&mut self, name: &str, ty: String) -> u32 {
        self.locals
            .push(format!("::nga::ir::Local {{ name: {name:?}, ty: {ty} }}"));
        let index = self.locals.len() as u32 - 1;
        self.scope.push((name.into(), index));
        index
    }

    fn find(&self, name: &str) -> Option<String> {
        if let Some((_, i)) = self.scope.iter().rev().find(|(n, _)| n == name) {
            return Some(format!("::nga::ir::Node::Local({i})"));
        }
        let i = self.params.iter().position(|p| p == name)?;
        Some(format!("::nga::ir::Node::Param({i})"))
    }
}

fn ident_at(t: &[TokenTree], i: usize) -> Option<String> {
    match t.get(i) {
        Some(TokenTree::Ident(id)) => Some(id.to_string()),
        _ => None,
    }
}

fn group_at(t: &[TokenTree], i: usize, delimiter: Delimiter) -> Option<Vec<TokenTree>> {
    match t.get(i) {
        Some(TokenTree::Group(g)) if g.delimiter() == delimiter => {
            Some(g.stream().into_iter().collect())
        }
        _ => None,
    }
}

fn span_at(t: &[TokenTree], i: usize) -> Span {
    t.get(i).map(|t| t.span()).unwrap_or_else(Span::call_site)
}

fn commas(t: &[TokenTree], angles: bool) -> Vec<Vec<TokenTree>> {
    let mut out = Vec::new();
    let mut current = Vec::new();
    let mut depth = 0i32;
    for token in t {
        if angles && punct(Some(token), '<') {
            depth += 1;
        }
        if angles && punct(Some(token), '>') {
            depth -= 1;
        }
        if depth == 0 && punct(Some(token), ',') {
            out.push(std::mem::take(&mut current));
        } else {
            current.push(token.clone());
        }
    }
    if !current.is_empty() {
        out.push(current);
    }
    out
}

fn block(b: &mut B, t: &[TokenTree], tail_returns: bool) -> Result<u32, Fail> {
    let depth = b.scope.len();
    let mut stmts = Vec::new();
    let mut i = 0;
    while i < t.len() {
        if punct(t.get(i), ';') {
            i += 1;
            continue;
        }
        let node = stmt(b, t, &mut i, tail_returns)?;
        stmts.push(node.to_string());
    }
    b.scope.truncate(depth);
    Ok(b.push(format!("::nga::ir::Node::Block(&[{}])", stmts.join(", "))))
}

fn stmt(b: &mut B, t: &[TokenTree], i: &mut usize, tail_returns: bool) -> Result<u32, Fail> {
    let span = span_at(t, *i);
    match ident_at(t, *i).as_deref() {
        Some("let") => {
            *i += 1;
            if ident_at(t, *i).as_deref() == Some("mut") {
                *i += 1;
            }
            let Some(name) = ident_at(t, *i) else {
                return Err((span, "expected a binding name".into()));
            };
            *i += 1;
            let mut annotation = None;
            if punct(t.get(*i), ':') {
                *i += 1;
                let start = *i;
                while *i < t.len() && !punct(t.get(*i), '=') {
                    *i += 1;
                }
                annotation = Some(ir_ty(&t[start..*i])?);
            }
            if !punct(t.get(*i), '=') {
                return Err((span, "a shader `let` must have an initialiser".into()));
            }
            *i += 1;
            let value = expr(b, t, i, 0, annotation.clone())?;
            let local = b.local(&name, annotation.unwrap_or("::nga::ir::Ty::Infer".into()));
            Ok(b.push(format!(
                "::nga::ir::Node::Let {{ local: {local}, value: {value} }}"
            )))
        }
        Some("if") => {
            *i += 1;
            conditional(b, t, i)
        }
        Some("while") => {
            *i += 1;
            b.no_struct = true;
            let cond = expr(b, t, i, 0, None)?;
            b.no_struct = false;
            let Some(body) = group_at(t, *i, Delimiter::Brace) else {
                return Err((span, "expected a `while` body".into()));
            };
            *i += 1;
            let body = block(b, &body, false)?;
            Ok(b.push(format!(
                "::nga::ir::Node::While {{ cond: {cond}, body: {body} }}"
            )))
        }
        Some("loop") => {
            *i += 1;
            let Some(body) = group_at(t, *i, Delimiter::Brace) else {
                return Err((span, "expected a `loop` body".into()));
            };
            *i += 1;
            let body = block(b, &body, false)?;
            Ok(b.push(format!("::nga::ir::Node::Loop {{ body: {body} }}")))
        }
        Some("for") => {
            *i += 1;
            let Some(name) = ident_at(t, *i) else {
                return Err((span, "expected a loop variable".into()));
            };
            *i += 1;
            if ident_at(t, *i).as_deref() != Some("in") {
                return Err((span, "expected `in`".into()));
            }
            *i += 1;
            b.no_struct = true;
            let start = expr(b, t, i, 0, None)?;
            if !punct(t.get(*i), '.') || !punct(t.get(*i + 1), '.') {
                return Err((
                    span,
                    "a shader `for` iterates `a..b` and nothing else".into(),
                ));
            }
            *i += 2;
            let end = expr(b, t, i, 0, None)?;
            b.no_struct = false;
            let Some(body) = group_at(t, *i, Delimiter::Brace) else {
                return Err((span, "expected a `for` body".into()));
            };
            *i += 1;
            let depth = b.scope.len();
            let local = b.local(&name, "::nga::ir::Ty::Infer".into());
            let body = block(b, &body, false)?;
            b.scope.truncate(depth);
            Ok(b.push(format!(
                "::nga::ir::Node::Range {{ local: {local}, start: {start}, end: {end}, body: {body} }}"
            )))
        }
        Some("return") => {
            *i += 1;
            let value = if punct(t.get(*i), ';') || *i >= t.len() {
                u32::MAX
            } else {
                expr(b, t, i, 0, None)?
            };
            Ok(b.push(format!("::nga::ir::Node::Return({value})")))
        }
        Some("break") => {
            *i += 1;
            Ok(b.push("::nga::ir::Node::Break".into()))
        }
        Some("continue") => {
            *i += 1;
            Ok(b.push("::nga::ir::Node::Continue".into()))
        }
        _ => {
            let place_start = *i;
            let value = expr(b, t, i, 0, None)?;

            let compound = match t.get(*i) {
                Some(TokenTree::Punct(p))
                    if p.spacing() == Spacing::Joint
                        && punct(t.get(*i + 1), '=')
                        && !matches!(p.as_char(), '=' | '!' | '<' | '>') =>
                {
                    binop(&[t[*i].clone()], 0).map(|(op, ..)| op)
                }
                _ => None,
            };
            if let Some(op) = compound {
                *i += 2;
                let rhs = expr(b, t, i, 0, None)?;

                let mut again = place_start;
                let place = expr(b, t, &mut again, 0, None)?;
                let value = b.push(format!(
                    "::nga::ir::Node::Binary(::nga::ir::BinOp::{op}, {value}, {rhs})"
                ));
                return Ok(b.push(format!(
                    "::nga::ir::Node::Store {{ place: {place}, value: {value} }}"
                )));
            }
            if punct(t.get(*i), '=') && !punct(t.get(*i + 1), '=') {
                *i += 1;
                let rhs = expr(b, t, i, 0, None)?;
                return Ok(b.push(format!(
                    "::nga::ir::Node::Store {{ place: {value}, value: {rhs} }}"
                )));
            }

            if tail_returns && *i >= t.len() {
                return Ok(b.push(format!("::nga::ir::Node::Return({value})")));
            }
            Ok(value)
        }
    }
}

fn conditional(b: &mut B, t: &[TokenTree], i: &mut usize) -> Result<u32, Fail> {
    let span = span_at(t, *i);
    b.no_struct = true;
    let cond = expr(b, t, i, 0, None)?;
    b.no_struct = false;
    let Some(then) = group_at(t, *i, Delimiter::Brace) else {
        return Err((span, "expected an `if` body".into()));
    };
    *i += 1;
    let then = block(b, &then, false)?;
    let mut els = u32::MAX;
    if ident_at(t, *i).as_deref() == Some("else") {
        *i += 1;
        if ident_at(t, *i).as_deref() == Some("if") {
            *i += 1;
            els = conditional(b, t, i)?;
        } else {
            let Some(body) = group_at(t, *i, Delimiter::Brace) else {
                return Err((span, "expected an `else` body".into()));
            };
            *i += 1;
            els = block(b, &body, false)?;
        }
    }
    Ok(b.push(format!(
        "::nga::ir::Node::If {{ cond: {cond}, then: {then}, els: {els} }}"
    )))
}

fn expr(
    b: &mut B,
    t: &[TokenTree],
    i: &mut usize,
    min: u8,
    want: Option<String>,
) -> Result<u32, Fail> {
    let mut lhs = unary(b, t, i, want)?;
    while let Some((op, power, len)) = binop(t, *i) {
        if power < min {
            break;
        }
        *i += len;
        let rhs = expr(b, t, i, power + 1, None)?;
        lhs = b.push(format!(
            "::nga::ir::Node::Binary(::nga::ir::BinOp::{op}, {lhs}, {rhs})"
        ));
    }
    Ok(lhs)
}

fn unary(b: &mut B, t: &[TokenTree], i: &mut usize, want: Option<String>) -> Result<u32, Fail> {
    if punct(t.get(*i), '-') {
        *i += 1;
        let value = unary(b, t, i, None)?;
        return Ok(b.push(format!(
            "::nga::ir::Node::Unary(::nga::ir::UnOp::Neg, {value})"
        )));
    }
    if punct(t.get(*i), '!') {
        *i += 1;
        let value = unary(b, t, i, None)?;
        return Ok(b.push(format!(
            "::nga::ir::Node::Unary(::nga::ir::UnOp::Not, {value})"
        )));
    }
    if punct(t.get(*i), '*') {

        *i += 1;
        return unary(b, t, i, want);
    }
    postfix(b, t, i, want)
}

fn postfix(b: &mut B, t: &[TokenTree], i: &mut usize, want: Option<String>) -> Result<u32, Fail> {
    let mut node = primary(b, t, i, want)?;
    loop {
        if punct(t.get(*i), '.') && !punct(t.get(*i + 1), '.') {
            let span = span_at(t, *i);
            *i += 1;
            let Some(name) = ident_at(t, *i) else {
                return Err((span, "expected a field or method name".into()));
            };
            *i += 1;
            match group_at(t, *i, Delimiter::Parenthesis) {
                Some(args) => {
                    *i += 1;

                    if args.is_empty()
                        && (2..=4).contains(&name.len())
                        && name.chars().all(|c| "xyzw".contains(c))
                    {
                        node = b.push(format!(
                            "::nga::ir::Node::Field {{ base: {node}, name: {name:?} }}"
                        ));
                        continue;
                    }
                    let Some(op) = intrinsic(&name) else {
                        return Err((span, format!("`{name}` is not a shader intrinsic")));
                    };
                    let mut list = vec![node.to_string()];
                    for arg in commas(&args, false) {
                        let mut j = 0;
                        list.push(expr(b, &arg, &mut j, 0, None)?.to_string());
                    }
                    node = b.push(format!(
                        "::nga::ir::Node::Intrinsic {{ op: ::nga::ir::Intrinsic::{op}, args: &[{}] }}",
                        list.join(", ")
                    ));
                }
                None => {
                    node = b.push(format!(
                        "::nga::ir::Node::Field {{ base: {node}, name: {name:?} }}"
                    ));
                }
            }
            continue;
        }
        if let Some(index) = group_at(t, *i, Delimiter::Bracket) {
            *i += 1;
            let mut j = 0;
            let index = expr(b, &index, &mut j, 0, None)?;
            node = b.push(format!(
                "::nga::ir::Node::Index {{ base: {node}, index: {index} }}"
            ));
            continue;
        }
        if ident_at(t, *i).as_deref() == Some("as") {
            *i += 1;
            let start = *i;

            *i += 1;
            let to = ir_ty(&t[start..*i])?;
            node = b.push(format!(
                "::nga::ir::Node::Cast {{ value: {node}, to: {to} }}"
            ));
            continue;
        }
        return Ok(node);
    }
}

fn primary(b: &mut B, t: &[TokenTree], i: &mut usize, want: Option<String>) -> Result<u32, Fail> {
    let span = span_at(t, *i);
    if let Some(inner) = group_at(t, *i, Delimiter::Parenthesis) {
        *i += 1;
        let mut j = 0;
        return expr(b, &inner, &mut j, 0, want);
    }
    if let Some(inner) = group_at(t, *i, Delimiter::Bracket) {
        *i += 1;
        let Some(ty) = want else {
            return Err((
                span,
                "an array literal needs a `let` type annotation".into(),
            ));
        };
        let mut list = Vec::new();
        for element in commas(&inner, false) {
            let mut j = 0;
            list.push(expr(b, &element, &mut j, 0, None)?.to_string());
        }
        return Ok(b.push(format!(
            "::nga::ir::Node::Construct {{ ty: {ty}, args: &[{}] }}",
            list.join(", ")
        )));
    }
    if let Some(TokenTree::Literal(literal)) = t.get(*i) {
        *i += 1;
        return literal_node(b, literal.to_string(), span);
    }
    let Some(name) = ident_at(t, *i) else {
        return Err((span, "expected an expression".into()));
    };
    *i += 1;
    match name.as_str() {
        "true" | "false" => {
            return Ok(b.push(format!(
                "::nga::ir::Node::Const(::nga::ir::Const::Bool({name}))"
            )));
        }
        "discard" => {
            if group_at(t, *i, Delimiter::Parenthesis).is_some() {
                *i += 1;
            }
            return Ok(b.push("::nga::ir::Node::Discard".into()));
        }
        _ => {}
    }

    let mut path = vec![name];
    while punct(t.get(*i), ':') && punct(t.get(*i + 1), ':') {
        let Some(segment) = ident_at(t, *i + 2) else {
            return Err((span, "expected a path segment".into()));
        };
        path.push(segment);
        *i += 3;
    }

    if let Some(args) = group_at(t, *i, Delimiter::Parenthesis) {
        *i += 1;
        let mut list = Vec::new();
        for arg in commas(&args, false) {
            let mut j = 0;
            list.push(expr(b, &arg, &mut j, 0, None)?.to_string());
        }
        let last = path.last().unwrap().clone();

        if path.len() >= 2 && matches!(last.as_str(), "new" | "from_array" | "splat") {
            let ty = named_ty(&path[path.len() - 2], &mut Vec::new());
            return Ok(b.push(format!(
                "::nga::ir::Node::Construct {{ ty: {ty}, args: &[{}] }}",
                list.join(", ")
            )));
        }
        if let Some(op) = intrinsic(&last) {
            return Ok(b.push(format!(
                "::nga::ir::Node::Intrinsic {{ op: ::nga::ir::Intrinsic::{op}, args: &[{}] }}",
                list.join(", ")
            )));
        }
        let mut target = path.clone();
        let n = target.len() - 1;
        target[n] = target[n].to_uppercase();
        let target = target.join("::");
        let func = match b.calls.iter().position(|c| *c == target) {
            Some(index) => index,
            None => {
                b.calls.push(target);
                b.calls.len() - 1
            }
        };
        return Ok(b.push(format!(
            "::nga::ir::Node::Call {{ func: {func}, args: &[{}] }}",
            list.join(", ")
        )));
    }

    if path.len() == 1
        && let Some(node) = b.find(&path[0])
    {
        return Ok(b.push(node));
    }

    if !b.no_struct
        && let Some(fields) = group_at(t, *i, Delimiter::Brace)
    {
        *i += 1;
        let ty = named_ty(&path[path.len() - 1], &mut Vec::new());
        let mut list = Vec::new();
        for field in commas(&fields, false) {
            let Some(name) = ident_at(&field, 0) else {
                return Err((span, "expected `field: value`".into()));
            };
            if !punct(field.get(1), ':') {
                return Err((span, "expected `field: value`".into()));
            }
            let mut j = 2;
            let value = expr(b, &field, &mut j, 0, None)?;
            list.push(format!("({name:?}, {value})"));
        }
        return Ok(b.push(format!(
            "::nga::ir::Node::StructLit {{ ty: {ty}, fields: &[{}] }}",
            list.join(", ")
        )));
    }

    let _ = span;
    Ok(b.push(format!(
        "::nga::ir::Node::Const(::nga::ir::Const::Float({} as f64))",
        path.join("::")
    )))
}

fn literal_node(b: &mut B, text: String, span: Span) -> Result<u32, Fail> {

    const SUFFIXES: &[&str] = &[
        "usize", "isize", "u128", "i128", "f32", "f64", "f16", "u64", "i64", "u32", "i32", "u16",
        "i16", "u8", "i8",
    ];
    let (stem, suffix) = SUFFIXES
        .iter()
        .find_map(|s| text.strip_suffix(s).map(|stem| (stem, *s)))
        .unwrap_or((text.as_str(), ""));
    let stem: String = stem.chars().filter(|&c| c != '_').collect();
    let float = stem.contains('.')
        || stem.contains('e')
        || stem.contains('E')
        || matches!(suffix, "f32" | "f64" | "f16");
    if float {
        let Ok(value) = stem.parse::<f64>() else {
            return Err((span, format!("`{text}` is not a shader literal")));
        };

        return Ok(b.push(format!(
            "::nga::ir::Node::Const(::nga::ir::Const::Float({value:?}))"
        )));
    }
    let Ok(value) = stem.parse::<i64>() else {
        return Err((span, format!("`{text}` is not a shader literal")));
    };
    Ok(b.push(format!(
        "::nga::ir::Node::Const(::nga::ir::Const::Int({value}))"
    )))
}

fn stage_wrapper(attr: &[TokenTree]) -> Option<&'static str> {
    match attr.first() {
        Some(TokenTree::Ident(id)) => match id.to_string().as_str() {
            "vertex" => Some("VertexShader"),
            "fragment" => Some("FragmentShader"),
            "compute" => Some("ComputeShader"),
            "object" => Some("ObjectShader"),
            "mesh" => Some("MeshShader"),
            "tile" => Some("TileShader"),
            _ => None,
        },
        _ => None,
    }
}

fn stage(attr: &[TokenTree]) -> Result<String, Fail> {
    match attr {
        [] => Ok("::nga::ir::Stage::Helper".into()),
        [TokenTree::Ident(id)] => match id.to_string().as_str() {
            "vertex" => Ok("::nga::ir::Stage::Vertex".into()),
            "fragment" => Ok("::nga::ir::Stage::Fragment".into()),
            "object" => Ok("::nga::ir::Stage::Object".into()),
            other => Err((id.span(), format!("`{other}` is not a shader stage"))),
        },
        [TokenTree::Ident(id), TokenTree::Group(g)]
            if id.to_string() == "compute" && g.delimiter() == Delimiter::Parenthesis =>
        {
            let sizes: Vec<String> = commas(&g.stream().into_iter().collect::<Vec<_>>(), false)
                .iter()
                .map(|s| s.iter().map(|t| t.to_string()).collect())
                .collect();
            match sizes.len() {
                1 => Ok(format!(
                    "::nga::ir::Stage::Compute([{}, 1, 1])",
                    sizes[0]
                )),
                2 => Ok(format!(
                    "::nga::ir::Stage::Compute([{}, {}, 1])",
                    sizes[0], sizes[1]
                )),
                3 => Ok(format!(
                    "::nga::ir::Stage::Compute([{}, {}, {}])",
                    sizes[0], sizes[1], sizes[2]
                )),
                _ => Err((g.span(), "`compute` takes one to three group sizes".into())),
            }
        }
        [TokenTree::Ident(id), TokenTree::Group(g)]
            if id.to_string() == "mesh" && g.delimiter() == Delimiter::Parenthesis =>
        {
            let args = commas(&g.stream().into_iter().collect::<Vec<_>>(), false);
            if args.len() != 3 {
                return Err((
                    g.span(),
                    "`mesh` takes `(VertexOutput, max_vertices, max_primitives)`".into(),
                ));
            }
            let vertex = ir_ty(&args[0])?;
            let vertices = args[1].iter().map(|t| t.to_string()).collect::<String>();
            let primitives = args[2].iter().map(|t| t.to_string()).collect::<String>();
            Ok(format!(
                "::nga::ir::Stage::Mesh {{ vertex: {vertex}, max_vertices: {vertices}, max_primitives: {primitives} }}"
            ))
        }
        [TokenTree::Ident(id), TokenTree::Group(g)]
            if id.to_string() == "tile" && g.delimiter() == Delimiter::Parenthesis =>
        {
            let args = commas(&g.stream().into_iter().collect::<Vec<_>>(), false);
            if args.len() != 1 {
                return Err((g.span(), "`tile` takes one imageblock struct type".into()));
            }
            Ok(format!(
                "::nga::ir::Stage::Tile {{ imageblock: {} }}",
                ir_ty(&args[0])?
            ))
        }
        _ => Err((
            attr[0].span(),
            "a shader fn takes nothing, `#[vertex]`, `#[fragment]`, `#[compute(x, y, z)]`, `#[object]`, `#[mesh(Vertex, vertices, primitives)]`, or `#[tile(Imageblock)]`".into(),
        )),
    }
}

fn builtin(name: &str) -> Option<&'static str> {
    Some(match name {
        "thread_id" => "ThreadId",
        "thread_id_in_group" => "ThreadIdInGroup",
        "group_id" => "GroupId",
        "thread_index_in_group" => "ThreadIndexInGroup",
        "vertex_id" => "VertexId",
        "instance_id" => "InstanceId",
        "base_instance" => "BaseInstance",
        "primitive_id" => "PrimitiveId",
        "position" => "Position",
        "render_target_array_index" => "RenderTargetArrayIndex",
        "front_facing" => "FrontFacing",
        _ => return None,
    })
}

pub fn expand(attr: TokenStream, item: TokenStream) -> Result<TokenStream, Fail> {
    let t: Vec<TokenTree> = item.into_iter().collect();
    let attr_tokens: Vec<TokenTree> = attr.into_iter().collect();
    let stage = stage(&attr_tokens)?;

    let Some(keyword) = t
        .iter()
        .position(|token| matches!(token, TokenTree::Ident(id) if id.to_string() == "fn"))
    else {
        return Err((Span::call_site(), "expected a function".into()));
    };
    let Some(TokenTree::Ident(name)) = t.get(keyword + 1) else {
        return Err((span_at(&t, keyword), "expected a function name".into()));
    };
    let Some(args) = group_at(&t, keyword + 2, Delimiter::Parenthesis) else {
        return Err((name.span(), "expected a parameter list".into()));
    };

    let mut ret = "::nga::ir::Ty::Void".to_string();
    let tail = &t[keyword + 3..];
    let Some(TokenTree::Group(body)) = tail.last() else {
        return Err((name.span(), "expected a function body".into()));
    };
    if body.delimiter() != Delimiter::Brace {
        return Err((name.span(), "expected a function body".into()));
    }
    if punct(tail.first(), '-') {
        ret = ir_ty(&tail[2..tail.len() - 1])?;
    }

    let mut b = B {
        nodes: Vec::new(),
        locals: Vec::new(),
        scope: Vec::new(),
        params: Vec::new(),
        calls: Vec::new(),
        no_struct: false,
    };

    let mut params = Vec::new();
    let mut stripped: Vec<String> = Vec::new();
    let mut root_taken = false;
    for arg in commas(&args, true) {
        let mut j = 0;
        let mut binding = None;
        while punct(arg.get(j), '#') {
            let Some(inner) = group_at(&arg, j + 1, Delimiter::Bracket) else {
                return Err((span_at(&arg, j), "expected `#[..]`".into()));
            };
            let Some(name) = ident_at(&inner, 0) else {
                return Err((span_at(&arg, j), "expected an attribute name".into()));
            };
            binding = Some(match (builtin(&name), name.as_str()) {
                (Some(b), _) => format!("::nga::ir::Binding::Builtin(::nga::ir::Builtin::{b})"),
                (None, "root") => "::nga::ir::Binding::Root".into(),
                (None, "constants") => "::nga::ir::Binding::Constants".into(),
                (None, "payload") => "::nga::ir::Binding::Payload".into(),
                (None, "buffer") => {
                    let index = group_at(&inner, 1, Delimiter::Parenthesis)
                        .map(|g| g.iter().map(|t| t.to_string()).collect::<String>())
                        .ok_or_else(|| {
                            (span_at(&arg, j), "`buffer` needs a binding index".into())
                        })?;
                    format!("::nga::ir::Binding::Buffer({index})")
                }
                (None, "location") => {
                    let index = group_at(&inner, 1, Delimiter::Parenthesis)
                        .map(|g| g.iter().map(|t| t.to_string()).collect::<String>())
                        .unwrap_or_else(|| "0".into());
                    format!("::nga::ir::Binding::Location({index})")
                }
                _ => return Err((span_at(&arg, j), format!("`{name}` is not a binding"))),
            });
            j += 2;
        }
        let mutable = ident_at(&arg, j).as_deref() == Some("mut");
        if mutable {
            j += 1;
        }
        let Some(name) = ident_at(&arg, j) else {
            return Err((span_at(&arg, j), "expected `name: Type`".into()));
        };
        if !punct(arg.get(j + 1), ':') {
            return Err((span_at(&arg, j), "expected `name: Type`".into()));
        }
        let ty_tokens = &arg[j + 2..];
        let ty = ir_ty(ty_tokens)?;
        let binding = binding.unwrap_or_else(|| {

            if !root_taken && stage != "::nga::ir::Stage::Helper" && ty.contains("Ty::Ptr") {
                root_taken = true;
                "::nga::ir::Binding::Root".into()
            } else {
                "::nga::ir::Binding::None".into()
            }
        });
        params.push(format!(
            "::nga::ir::Param {{ name: {name:?}, ty: {ty}, binding: {binding} }}"
        ));
        stripped.push(format!(
            "{}{name}: {}",
            if mutable { "mut " } else { "" },
            ty_tokens.iter().map(|t| t.to_string()).collect::<String>()
        ));
        b.params.push(name);
    }

    let body_tokens: Vec<TokenTree> = body.stream().into_iter().collect();
    let root = block(&mut b, &body_tokens, ret != "::nga::ir::Ty::Void")?;

    let konst = if let Some(wrapper) = stage_wrapper(&attr_tokens) {
        format!(
            "const {upper}_FN: ::nga::ir::GpuFn = ::nga::ir::GpuFn {{
                 name: {name:?},
                 stage: {stage},
                 params: &[{params}],
                 ret: {ret},
                 locals: &[{locals}],
                 nodes: &[{nodes}],
                 body: {root},
                 calls: &[{calls}],
             }};
             #[allow(non_upper_case_globals)]
             pub const {upper}: ::nga::{wrapper} = ::nga::{wrapper}(&{upper}_FN);",
            upper = name.to_string().to_uppercase(),
            name = name.to_string(),
            params = params.join(", "),
            locals = b.locals.join(", "),
            nodes = b.nodes.join(", "),
            calls = b
                .calls
                .iter()
                .map(|c| format!("&{c}"))
                .collect::<Vec<_>>()
                .join(", "),
        )
    } else {
        format!(
            "#[allow(non_upper_case_globals)]
             pub const {upper}: ::nga::ir::GpuFn = ::nga::ir::GpuFn {{
                 name: {name:?},
                 stage: {stage},
                 params: &[{params}],
                 ret: {ret},
                 locals: &[{locals}],
                 nodes: &[{nodes}],
                 body: {root},
                 calls: &[{calls}],
             }};",
            upper = name.to_string().to_uppercase(),
            name = name.to_string(),
            params = params.join(", "),
            locals = b.locals.join(", "),
            nodes = b.nodes.join(", "),
            calls = b
                .calls
                .iter()
                .map(|c| format!("&{c}"))
                .collect::<Vec<_>>()
                .join(", "),
        )
    };

    let signature = format!(
        "#[allow(dead_code, unused_variables, unused_mut, unused_parens, unused_assignments, non_snake_case)]
         {prefix} fn {name}({args}){arrow}",
        prefix = t[..keyword]
            .iter()
            .map(|t| t.to_string())
            .collect::<String>(),
        name = name.to_string(),
        args = stripped.join(", "),
        arrow = if punct(tail.first(), '-') {
            tail[..tail.len() - 1]
                .iter()
                .map(|t| t.to_string())
                .collect::<String>()
        } else {
            String::new()
        },
    );

    let mut out: TokenStream = signature
        .parse()
        .map_err(|_| (name.span(), "could not rebuild the signature".into()))?;
    out.extend([TokenTree::Group(body.clone())]);
    out.extend(
        konst
            .parse::<TokenStream>()
            .map_err(|_| (name.span(), format!("could not emit the IR:\n{konst}")))?,
    );
    Ok(out)
}
