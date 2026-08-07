//! per-pixel band math: one output band computed from the input chunk's
//! bands through an expression parsed at construction. window-local on an
//! identity plan, so chunked output equals whole-window output

use crate::caps::{CapsPattern, CapsSet, Constraint, Dtype, FieldMask, RasterPattern, SetField};
use crate::chunk::{Chunk, RasterChunk};
use crate::element::Transform;
use crate::error::{Error, Result};
use crate::window::WindowReq;
use terrano_core::{BandedRaster, Raster};

/// what a true comparison yields, and what `where` reads as its true
/// branch. any other finite value is true to `where` as well
const TRUE_VALUE: f64 = 1.0;
const FALSE_VALUE: f64 = 0.0;

fn from_bool(hit: bool) -> f64 {
    if hit { TRUE_VALUE } else { FALSE_VALUE }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Less,
    LessOrEqual,
    Greater,
    GreaterOrEqual,
    Equal,
    NotEqual,
}

/// spelling of every comparison, two-character ones first so the lexer can
/// take the first match at a position
const COMPARISONS: [(&str, BinOp); 6] = [
    ("<=", BinOp::LessOrEqual),
    (">=", BinOp::GreaterOrEqual),
    ("==", BinOp::Equal),
    ("!=", BinOp::NotEqual),
    ("<", BinOp::Less),
    (">", BinOp::Greater),
];

impl BinOp {
    /// nan in, nan out: every rust comparison against a nan is false, so
    /// nodata would otherwise come out as a confident 0.0, and `!=` as 1.0
    fn eval(self, a: f64, b: f64) -> f64 {
        if a.is_nan() || b.is_nan() {
            return f64::NAN;
        }
        match self {
            BinOp::Add => a + b,
            BinOp::Sub => a - b,
            BinOp::Mul => a * b,
            BinOp::Div => a / b,
            BinOp::Less => from_bool(a < b),
            BinOp::LessOrEqual => from_bool(a <= b),
            BinOp::Greater => from_bool(a > b),
            BinOp::GreaterOrEqual => from_bool(a >= b),
            BinOp::Equal => from_bool(a == b),
            BinOp::NotEqual => from_bool(a != b),
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum Func {
    Sqrt,
    Abs,
    Min,
    Max,
    Pow,
    Log,
    Exp,
    Where,
}

const FUNCS: [(&str, Func, usize); 8] = [
    ("sqrt", Func::Sqrt, 1),
    ("abs", Func::Abs, 1),
    ("min", Func::Min, 2),
    ("max", Func::Max, 2),
    ("pow", Func::Pow, 2),
    ("log", Func::Log, 1),
    ("exp", Func::Exp, 1),
    ("where", Func::Where, 3),
];

/// widest arity in `FUNCS`, the per-pixel argument buffer
const MAX_ARITY: usize = 3;

impl Func {
    fn lookup(name: &str) -> Option<(Func, usize)> {
        FUNCS
            .iter()
            .find(|(n, ..)| *n == name)
            .map(|(_, f, arity)| (*f, *arity))
    }

    /// nan in, nan out for every function: `f64::min` and `f64::max` would
    /// otherwise return the other operand, and `powf` returns 1.0 for a nan
    /// base with a zero exponent
    fn eval(self, args: &[f64]) -> f64 {
        if args.iter().any(|a| a.is_nan()) {
            return f64::NAN;
        }
        match self {
            Func::Sqrt => args[0].sqrt(),
            Func::Abs => args[0].abs(),
            Func::Min => args[0].min(args[1]),
            Func::Max => args[0].max(args[1]),
            Func::Pow => args[0].powf(args[1]),
            Func::Log => args[0].ln(),
            Func::Exp => args[0].exp(),
            Func::Where => {
                if args[0] == FALSE_VALUE {
                    args[2]
                } else {
                    args[1]
                }
            }
        }
    }
}

enum Expr {
    Band(usize),
    Lit(f64),
    Neg(Box<Expr>),
    Bin(BinOp, Box<Expr>, Box<Expr>),
    Call(Func, Vec<Expr>),
}

fn eval(expr: &Expr, bands: &[f64]) -> f64 {
    match expr {
        Expr::Band(i) => bands[*i],
        Expr::Lit(v) => *v,
        Expr::Neg(inner) => -eval(inner, bands),
        Expr::Bin(op, a, b) => op.eval(eval(a, bands), eval(b, bands)),
        Expr::Call(func, args) => {
            let mut vals = [f64::NAN; MAX_ARITY];
            for (slot, arg) in vals.iter_mut().zip(args) {
                *slot = eval(arg, bands);
            }
            func.eval(&vals[..args.len()])
        }
    }
}

#[derive(Clone, PartialEq)]
enum Tok {
    Num(f64),
    Name(String),
    Sym(char),
    Comparison(BinOp),
}

fn comparison_text(op: BinOp) -> &'static str {
    COMPARISONS
        .iter()
        .find(|(_, candidate)| *candidate == op)
        .map(|(text, _)| *text)
        .expect("a comparison token holds a comparison")
}

fn describe(tok: &Tok) -> String {
    match tok {
        Tok::Num(v) => format!("{v}"),
        Tok::Name(n) => n.clone(),
        Tok::Sym(c) => format!("'{c}'"),
        Tok::Comparison(op) => format!("'{}'", comparison_text(*op)),
    }
}

fn parse_err(detail: String) -> Error {
    Error::InvalidGraph(format!("band math: {detail}"))
}

/// `text` is ascii, so its byte length is also its length in `chars`
fn starts_with(chars: &[char], at: usize, text: &str) -> bool {
    text.chars()
        .enumerate()
        .all(|(offset, c)| chars.get(at + offset) == Some(&c))
}

fn lex(src: &str) -> Result<Vec<Tok>> {
    let chars: Vec<char> = src.chars().collect();
    let mut toks = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c.is_whitespace() {
            i += 1;
        } else if c.is_ascii_digit() || c == '.' {
            let start = i;
            while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
                i += 1;
            }
            let text: String = chars[start..i].iter().collect();
            let v = text
                .parse::<f64>()
                .map_err(|_| parse_err(format!("bad number {text}")))?;
            toks.push(Tok::Num(v));
        } else if c.is_ascii_alphabetic() {
            let start = i;
            while i < chars.len() && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            toks.push(Tok::Name(chars[start..i].iter().collect()));
        } else if "+-*/(),".contains(c) {
            toks.push(Tok::Sym(c));
            i += 1;
        } else if let Some((text, op)) = COMPARISONS
            .iter()
            .find(|(text, _)| starts_with(&chars, i, text))
        {
            toks.push(Tok::Comparison(*op));
            i += text.len();
        } else {
            return Err(parse_err(format!("unexpected character '{c}'")));
        }
    }
    Ok(toks)
}

struct Parser<'a> {
    toks: &'a [Tok],
    pos: usize,
    /// highest band index referenced so far, plus one
    bands: usize,
}

impl Parser<'_> {
    fn peek_sym(&self) -> Option<char> {
        match self.toks.get(self.pos) {
            Some(Tok::Sym(c)) => Some(*c),
            _ => None,
        }
    }

    fn eat_sym(&mut self, c: char) -> bool {
        let hit = self.peek_sym() == Some(c);
        if hit {
            self.pos += 1;
        }
        hit
    }

    fn expect_sym(&mut self, c: char) -> Result<()> {
        match self.toks.get(self.pos) {
            Some(Tok::Sym(s)) if *s == c => {
                self.pos += 1;
                Ok(())
            }
            Some(other) => Err(parse_err(format!(
                "expected '{c}', got {}",
                describe(other)
            ))),
            None => Err(parse_err(format!("expected '{c}', expression ends"))),
        }
    }

    fn peek_comparison(&self) -> Option<BinOp> {
        match self.toks.get(self.pos) {
            Some(Tok::Comparison(op)) => Some(*op),
            _ => None,
        }
    }

    /// comparisons bind loosest, so `b0 + 1 < b1` compares the sums
    fn expr(&mut self) -> Result<Expr> {
        let mut left = self.additive()?;
        while let Some(op) = self.peek_comparison() {
            self.pos += 1;
            let right = self.additive()?;
            left = Expr::Bin(op, Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn additive(&mut self) -> Result<Expr> {
        let mut left = self.term()?;
        while let Some(c @ ('+' | '-')) = self.peek_sym() {
            self.pos += 1;
            let right = self.term()?;
            let op = if c == '+' { BinOp::Add } else { BinOp::Sub };
            left = Expr::Bin(op, Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn term(&mut self) -> Result<Expr> {
        let mut left = self.unary()?;
        while let Some(c @ ('*' | '/')) = self.peek_sym() {
            self.pos += 1;
            let right = self.unary()?;
            let op = if c == '*' { BinOp::Mul } else { BinOp::Div };
            left = Expr::Bin(op, Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn unary(&mut self) -> Result<Expr> {
        if self.eat_sym('-') {
            return Ok(Expr::Neg(Box::new(self.unary()?)));
        }
        self.atom()
    }

    fn atom(&mut self) -> Result<Expr> {
        let tok = self
            .toks
            .get(self.pos)
            .cloned()
            .ok_or_else(|| parse_err("expression ends where a value is due".into()))?;
        self.pos += 1;
        match tok {
            Tok::Num(v) => Ok(Expr::Lit(v)),
            Tok::Sym('(') => {
                let inner = self.expr()?;
                self.expect_sym(')')?;
                Ok(inner)
            }
            Tok::Name(name) => self.named(name),
            other => Err(parse_err(format!(
                "expected a value, got {}",
                describe(&other)
            ))),
        }
    }

    fn named(&mut self, name: String) -> Result<Expr> {
        if let Some((func, arity)) = Func::lookup(&name) {
            self.expect_sym('(')?;
            let mut args = vec![self.expr()?];
            while self.eat_sym(',') {
                args.push(self.expr()?);
            }
            self.expect_sym(')')?;
            if args.len() != arity {
                return Err(parse_err(format!(
                    "{name} takes {arity} arguments, got {}",
                    args.len()
                )));
            }
            return Ok(Expr::Call(func, args));
        }
        if let Some(index) = name.strip_prefix('b').and_then(|d| d.parse::<usize>().ok()) {
            self.bands = self.bands.max(index + 1);
            return Ok(Expr::Band(index));
        }
        if self.peek_sym() == Some('(') {
            return Err(parse_err(format!("unknown function {name}")));
        }
        Err(parse_err(format!("unknown name {name}")))
    }
}

/// one output band per pixel from an expression over the input bands.
/// band variables are `b0`, `b1`, ..., with f64 literals, `+ - * /`, unary
/// minus, parentheses, the comparisons `< <= > >= == !=` yielding 1.0 or
/// 0.0, and `sqrt`, `abs`, `min`, `max`, `pow`, `log` (natural), `exp` and
/// `where(cond, a, b)`, which takes `a` where `cond` is nonzero. a nodata
/// cell enters the expression as NaN and NaN propagates through every
/// operator, so the output nodata is NaN
pub struct BandMath {
    root: Expr,
    bands: usize,
}

impl BandMath {
    pub fn new(expr: &str) -> Result<BandMath> {
        let toks = lex(expr)?;
        let mut parser = Parser {
            toks: &toks,
            pos: 0,
            bands: 0,
        };
        let root = parser.expr()?;
        if let Some(tok) = parser.toks.get(parser.pos) {
            return Err(parse_err(format!("trailing {}", describe(tok))));
        }
        Ok(BandMath {
            root,
            bands: parser.bands,
        })
    }
}

impl Transform for BandMath {
    fn constraint(&self) -> Constraint {
        Constraint::Derived {
            input: CapsSet::one(CapsPattern::Raster(RasterPattern {
                // a band index past u16 just demands more than any link carries
                bands: SetField::AtLeast(u16::try_from(self.bands).unwrap_or(u16::MAX)),
                ..RasterPattern::default()
            })),
            passthrough: FieldMask {
                dtype: false,
                bands: false,
                crs: true,
                resolution: true,
                chunk_px: true,
            },
            output: CapsPattern::Raster(RasterPattern {
                dtype: SetField::one(Dtype::F64),
                bands: SetField::one(1),
                ..RasterPattern::default()
            }),
        }
    }

    fn plan(&self, out: &WindowReq) -> WindowReq {
        *out
    }

    fn compute(&self, out: &WindowReq, input: &Chunk) -> Result<Chunk> {
        let input = input.raster()?.crop_to(&out.bbox);
        let planes: Vec<&Raster> = (0..self.bands)
            .map(|b| input.bands.band(b).expect("negotiated bands"))
            .collect();
        let (cols, rows) = (input.width(), input.height());
        let mut values = vec![f64::NAN; self.bands];
        let mut data = Vec::with_capacity(cols * rows);
        for cell in 0..cols * rows {
            for (value, plane) in values.iter_mut().zip(&planes) {
                let raw = plane.data()[cell];
                *value = if plane.is_nodata(raw) { f64::NAN } else { raw };
            }
            data.push(eval(&self.root, &values));
        }
        let band = Raster::from_vec(cols, rows, data, input.resolution, f64::NAN)
            .map_err(Error::Terrano)?;
        Ok(Chunk::Raster(RasterChunk {
            bands: BandedRaster::new(vec![band]).expect("one band"),
            bbox: input.bbox,
            resolution: input.resolution,
            crs: input.crs,
        }))
    }
}
