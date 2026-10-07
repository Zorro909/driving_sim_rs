#!/usr/bin/env python3
"""Translate the restricted C of math/kernels/ucrt.h to Rust (src/math/ucrt.rs).

Requires pycparser 2.23 (wasm/requirements-sim.txt). Run from any directory with
python3 tools/ucrt/gen-math.py; --check compares instead of writing.

The translation keeps every operation and its order: integer arithmetic wraps,
fma becomes mul_add, casts become `as` (the kernels only convert in range), and
each `for` becomes a `while`. The kernels' rules (math/kernels/ucrt.h) are
checked: the operands of a binary operator must have one type, except shift
counts and literals.
"""

import argparse
from pathlib import Path
import re
import subprocess

from pycparser import c_ast as A, c_parser

root = Path(__file__).resolve().parents[2]
KERNELS = root / "math/kernels"
OUTPUT = root / "src/math/ucrt.rs"

TYPES = {
    "double": "f64",
    "float": "f32",
    "uint64_t": "u64",
    "int64_t": "i64",
    "uint32_t": "u32",
    "int32_t": "i32",
    "int": "i32",
    "bool": "bool",
}
INTS = {"u64", "i64", "u32", "i32"}
FLOATS = {"f64", "f32"}
BUILTINS = {"fbits": "u32", "ffrom": "f32", "dbits": "u64", "dfrom": "f64", "fma": "f64"}

# Rust binding strength, higher binds tighter.
PREC = {"*": 11, "/": 11, "%": 11, "+": 10, "-": 10, "<<": 9, ">>": 9, "&": 8, "^": 7, "|": 6}
for op in ["==", "!=", "<", ">", "<=", ">="]:
    PREC[op] = 5
PREC["&&"] = 4
PREC["||"] = 3
ATOM, UNARY, CAST, IF = 14, 13, 12, 1


def source():
    text = (KERNELS / "ucrt.h").read_text()
    text = text.replace('#include "ucrt_tables.h"', (KERNELS / "ucrt_tables.h").read_text())
    text = re.sub(r"//[^\n]*", "", text)
    # Rust has every function; ALTD_MATH_WGSL only fences the WGSL port.
    text = re.sub(r"^\s*#.*$", "", text, flags=re.M)
    text = (
        text.replace("ALTD_MATH_FN", "static")
        .replace("ALTD_MATH_TABLE", "static const")
        .replace("ALTD_MATH_CONST", "static const")
    )
    prelude = "typedef unsigned long long uint64_t; typedef long long int64_t; typedef unsigned int uint32_t; typedef int int32_t; typedef int bool;\n"
    return prelude + text


def ctype(n):
    if isinstance(n, (A.TypeDecl, A.Typename)):
        return ctype(n.type)
    if isinstance(n, A.IdentifierType):
        return TYPES[" ".join(n.names)]
    if isinstance(n, A.ArrayDecl):
        return ("arr", ctype(n.type), int(n.dim.value))
    if isinstance(n, A.FuncDecl):
        return ctype(n.type)
    raise ValueError(("type", n))


def literal_type(n):
    t = n.type
    if t == "double":
        return "f64"
    if t == "float":
        return "f32"
    unsigned = "unsigned" in t
    if "long long" in t:
        return "u64" if unsigned else "i64"
    return "u32" if unsigned else "i32"


class Emitter:
    def __init__(self, ast):
        self.glob = {}
        self.funcs = {}
        for n in ast.ext:
            if isinstance(n, A.FuncDef):
                self.funcs[n.decl.name] = n
            elif isinstance(n, A.Decl) and n.name:
                self.glob[n.name] = ctype(n.type)
        self.scopes = []

    # ---- types ----------------------------------------------------------

    def lookup(self, name):
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        if name in self.glob:
            return self.glob[name]
        raise ValueError(("unknown name", name))

    def kind(self, n):
        if isinstance(n, A.ID):
            return self.lookup(n.name)
        if isinstance(n, A.Constant):
            return literal_type(n)
        if isinstance(n, A.Cast):
            return ctype(n.to_type)
        if isinstance(n, A.ArrayRef):
            return self.kind(n.name)[1]
        if isinstance(n, A.UnaryOp):
            return "bool" if n.op == "!" else self.kind(n.expr)
        if isinstance(n, A.BinaryOp):
            if n.op in ["&&", "||", "<", ">", "<=", ">=", "==", "!="]:
                return "bool"
            if n.op in ["<<", ">>"]:
                return self.kind(n.left)
            return self.common(n.left, n.right)
        if isinstance(n, A.TernaryOp):
            return self.common(n.iftrue, n.iffalse)
        if isinstance(n, A.FuncCall):
            f = n.name.name
            if f in BUILTINS:
                return BUILTINS[f]
            return ctype(self.funcs[f].decl.type.type)
        raise ValueError(("kind", n))

    def is_literal(self, n):
        if isinstance(n, A.UnaryOp) and n.op == "-":
            return self.is_literal(n.expr)
        return isinstance(n, A.Constant)

    def common(self, a, b):
        """The one type of two operands; a literal takes the other's type."""
        if self.is_literal(a) and not self.is_literal(b):
            return self.kind(b)
        if self.is_literal(b) and not self.is_literal(a):
            return self.kind(a)
        ta, tb = self.kind(a), self.kind(b)
        if ta != tb:
            if self.is_literal(a) and self.is_literal(b) and {ta, tb} <= INTS:
                return ta
            raise ValueError(("mixed operand types", ta, tb, a.coord))
        return ta

    # ---- expressions: (text, binding strength) ---------------------------

    def wrap(self, e, prec):
        text, p = e
        return f"({text})" if p < prec else text

    def expr(self, n, want=None):
        if isinstance(n, A.Constant):
            return self.constant(n, want), ATOM
        if isinstance(n, A.ID):
            return n.name, ATOM
        if isinstance(n, A.ArrayRef):
            index = self.wrap(self.expr(n.subscript), CAST + 1)
            return f"{n.name.name}[{index} as usize]", ATOM
        if isinstance(n, A.Cast):
            to = ctype(n.to_type)
            inner = self.expr(n.expr)
            if self.kind(n.expr) == to:
                return inner
            return f"{self.wrap(inner, CAST)} as {to}", CAST
        if isinstance(n, A.UnaryOp):
            t = self.kind(n.expr) if n.op != "!" else "bool"
            if n.op == "-" and isinstance(n.expr, A.Constant):
                return "-" + self.constant(n.expr, want), UNARY
            inner = self.expr(n.expr, want)
            if n.op == "-" and t in INTS:
                return f"{self.wrap(inner, ATOM)}.wrapping_neg()", ATOM
            if n.op in ["-", "!"]:
                return n.op + self.wrap(inner, UNARY), UNARY
            raise ValueError(("unary", n.op))
        if isinstance(n, A.BinaryOp):
            return self.binary(n)
        if isinstance(n, A.TernaryOp):
            t = want or self.kind(n)
            c = self.expr(n.cond)[0]
            a = self.expr(n.iftrue, t)[0]
            b = self.expr(n.iffalse, t)[0]
            return f"if {c} {{ {a} }} else {{ {b} }}", IF
        if isinstance(n, A.FuncCall):
            return self.call(n)
        raise ValueError(("expr", n))

    def constant(self, n, want):
        t = want if want in INTS | FLOATS else literal_type(n)
        v = n.value
        if n.type in ["double", "float"]:
            assert t == literal_type(n), ("float literal type", n.coord)
            x = float(re.sub(r"[fF]$", "", v))
            s = repr(x)
            return s if any(c in s for c in ".e") else s + ".0"
        assert t in INTS, ("integer literal for", t, n.coord)
        digits = re.sub(r"[uUlL]+$", "", v)
        if "x" in digits:
            assert int(digits, 16) < (1 << (64 if t[1:] == "64" else 32)), n.coord
        return digits + t

    def binary(self, n):
        op = n.op
        if op in ["&&", "||"]:
            p = PREC[op]
            a = self.wrap(self.expr(n.left), p)
            b = self.wrap(self.expr(n.right), p + 1)
            return f"{a} {op} {b}", p
        if op in ["<<", ">>"]:
            t = self.kind(n.left)
            assert t in INTS, ("shift of", t)
            # A cast before `<<` reads as generics.
            a = self.wrap(self.expr(n.left, t), CAST + 1)
            count = self.expr(n.right, "u32")
            if self.kind(n.right) != "u32" and not self.is_literal(n.right):
                count = (f"{self.wrap(count, CAST)} as u32", CAST)
            b = self.wrap(count, PREC[op] + 1)
            return f"{a} {op} {b}", PREC[op]
        t = self.common(n.left, n.right)
        p = PREC[op]
        if p == 5:
            # Comparisons do not chain in Rust; a cast before `<` reads as generics.
            a = self.wrap(self.expr(n.left, t), p + 1 if op not in ["<", "<="] else CAST + 1)
            b = self.wrap(self.expr(n.right, t), p + 1)
            return f"{a} {op} {b}", p
        if t in INTS and op in ["+", "-", "*"]:
            method = {"+": "wrapping_add", "-": "wrapping_sub", "*": "wrapping_mul"}[op]
            a = self.wrap(self.expr(n.left, t), ATOM)
            if self.is_literal(n.left):
                a = f"({a})" if a.startswith("-") else a
            b = self.expr(n.right, t)[0]
            return f"{a}.{method}({b})", ATOM
        if t in INTS and op in ["/", "%"]:
            raise ValueError(("integer division", n.coord))
        a = self.wrap(self.expr(n.left, t), p)
        b = self.wrap(self.expr(n.right, t), p + 1)
        return f"{a} {op} {b}", p

    def call(self, n):
        f = n.name.name
        args = n.args.exprs if n.args else []
        if f == "fma":
            a = self.expr(args[0], "f64")
            text = self.wrap(a, ATOM)
            if self.is_literal(args[0]):
                text = f"({text}_f64)" if text.startswith("-") else text + "_f64"
            return f"{text}.mul_add({self.expr(args[1], 'f64')[0]}, {self.expr(args[2], 'f64')[0]})", ATOM
        if f in ["fbits", "dbits"]:
            return f"{self.wrap(self.expr(args[0]), ATOM)}.to_bits()", ATOM
        if f in ["ffrom", "dfrom"]:
            t = "f32" if f == "ffrom" else "f64"
            return f"{t}::from_bits({self.expr(args[0], 'u32' if t == 'f32' else 'u64')[0]})", ATOM
        params = self.funcs[f].decl.type.args.params
        out = [self.expr(a, ctype(p.type))[0] for p, a in zip(params, args)]
        return f"{f}({', '.join(out)})", ATOM

    # ---- statements -----------------------------------------------------

    def function(self, n):
        decl = n.decl
        ret = ctype(decl.type.type)
        self.ret = ret
        params = decl.type.args.params if decl.type.args else []
        self.scopes = [{p.name: ctype(p.type) for p in params}]
        self.mutated = Mutation(self).function(n)
        sig = ", ".join(
            ("mut " if ("param", p.name) in self.mutated else "") + f"{p.name}: {ctype(p.type)}" for p in params
        )
        body = self.compound(n.body, new_scope=False)
        return f"#[inline]\npub fn {decl.name}({sig}) -> {ret} {body}"

    def compound(self, n, new_scope=True):
        if new_scope:
            self.scopes.append({})
        lines = [self.statement(s) for s in n.block_items or []]
        if new_scope:
            self.scopes.pop()
        return "{\n" + "\n".join(lines) + "\n}"

    def block(self, n):
        if isinstance(n, A.Compound):
            return self.compound(n)
        self.scopes.append({})
        s = self.statement(n)
        self.scopes.pop()
        return "{\n" + s + "\n}"

    def statement(self, n):
        if isinstance(n, A.Decl):
            t = ctype(n.type)
            mut = "mut " if id(n) in self.mutated else ""
            if n.init is None:
                self.scopes[-1][n.name] = t
                return f"let {mut}{n.name}: {t};"
            init = self.expr(n.init, t)[0]
            self.scopes[-1][n.name] = t
            return f"let {mut}{n.name}: {t} = {init};"
        if isinstance(n, A.Assignment):
            assert n.op == "=", ("compound assignment", n.coord)
            t = self.kind(n.lvalue)
            return f"{n.lvalue.name} = {self.expr(n.rvalue, t)[0]};"
        if isinstance(n, A.Return):
            return f"return {self.expr(n.expr, self.ret)[0]};"
        if isinstance(n, A.If):
            s = f"if {self.expr(n.cond)[0]} {self.block(n.iftrue)}"
            if n.iffalse is not None:
                if isinstance(n.iffalse, A.If):
                    s += " else " + self.statement(n.iffalse)
                else:
                    s += " else " + self.block(n.iffalse)
            return s
        if isinstance(n, A.For):
            assert not any(isinstance(c, A.Continue) for _, c in n.stmt.children()), "continue in for"
            self.scopes.append({})
            init = "\n".join(self.statement(d) for d in n.init.decls)
            cond = self.expr(n.cond)[0]
            body = self.block(n.stmt)[:-1] + self.statement(n.next) + "\n}"
            self.scopes.pop()
            return "{\n" + init + f"\nwhile {cond} " + body + "\n}"
        raise ValueError(("statement", n))


class Mutation:
    """Which declarations need `mut`: those assigned after they may hold a value."""

    def __init__(self, emitter):
        self.emitter = emitter

    def function(self, n):
        self.mutated = set()
        self.scopes = [{p.name: ("param", p.name) for p in (n.decl.type.args.params if n.decl.type.args else [])}]
        # A parameter or initialized variable is assigned; a declaration alone is not.
        self.walk(n.body, {key: True for key in self.scopes[0].values()})
        return self.mutated

    def resolve(self, name):
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return None

    def walk(self, n, state):
        """Return the state after n, or None if n always returns."""
        if state is None:
            return None
        if isinstance(n, A.Compound):
            self.scopes.append({})
            for s in n.block_items or []:
                state = self.walk(s, state)
            self.scopes.pop()
            return state
        if isinstance(n, A.Decl):
            key = id(n)
            self.scopes[-1][n.name] = key
            state = dict(state)
            state[key] = n.init is not None
            return state
        if isinstance(n, A.Assignment):
            key = self.resolve(n.lvalue.name)
            if state.get(key):
                self.mutated.add(key)
            state = dict(state)
            state[key] = True
            return state
        if isinstance(n, A.Return):
            return None
        if isinstance(n, A.If):
            self.scopes.append({})
            a = self.walk(n.iftrue, state)
            self.scopes.pop()
            self.scopes.append({})
            b = self.walk(n.iffalse, state) if n.iffalse is not None else state
            self.scopes.pop()
            if a is None:
                return b
            if b is None:
                return a
            return {k: a.get(k, False) or b.get(k, False) for k in set(a) | set(b)}
        if isinstance(n, A.For):
            self.scopes.append({})
            for d in n.init.decls:
                state = self.walk(d, state)
            # Twice: the second iteration sees the first's assignments.
            for _ in range(2):
                after = self.walk(n.stmt, state)
                after = self.walk(n.next, after) if after is not None else None
                if after is not None:
                    state = {k: state.get(k, False) or after.get(k, False) for k in set(state) | set(after)}
            self.scopes.pop()
            return state
        raise ValueError(("mutation", n))


def generate():
    ast = c_parser.CParser().parse(source())
    em = Emitter(ast)
    out = [
        "// Generated from math/kernels/ucrt.h by tools/ucrt/gen-math.py. Do not edit.",
        "//! The float math of the Windows UCRT (FMA3 code paths), bit for bit: see math/kernels/ucrt.h.",
        "#![allow(clippy::all, clippy::pedantic)]",
        "",
    ]
    for n in ast.ext:
        if isinstance(n, A.Decl) and n.name:
            t = ctype(n.type)
            if isinstance(t, tuple):
                values = [em.expr(v, t[1])[0] + "," for v in n.init.exprs]
                per = 4 if t[1] == "u64" else 8
                rows = ["    " + " ".join(values[i : i + per]) for i in range(0, len(values), per)]
                out.append("#[rustfmt::skip]")
                out.append(f"pub static {n.name}: [{t[1]}; {t[2]}] = [\n" + "\n".join(rows) + "\n];")
            else:
                out.append(f"pub const {n.name}: {t} = {em.expr(n.init, t)[0]};")
        elif isinstance(n, A.FuncDef):
            out.append(em.function(n))
    text = "\n".join(out) + "\n"
    r = subprocess.run(
        ["rustfmt", "--edition", "2021", "--emit", "stdout", "--config", "max_width=120"],
        input=text, capture_output=True, text=True,
    )
    if r.returncode:
        raise SystemExit(r.stderr)
    return r.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="compare src/math/ucrt.rs without writing")
    args = parser.parse_args()
    text = generate()
    if args.check:
        if OUTPUT.read_text() != text:
            raise SystemExit(f"{OUTPUT.relative_to(root)} is stale: run python3 tools/ucrt/gen-math.py")
        print(f"{OUTPUT.relative_to(root)} is current")
        return
    OUTPUT.write_text(text)
    print(f"wrote {OUTPUT.relative_to(root)}")


if __name__ == "__main__":
    main()
