#!/usr/bin/env python3
"""Generate the exact WGSL port and its world encoder from gpu/sim.

Requires pycparser 2.23. Run from any directory with python3 wasm/gen-sim.py.
HIP source remains the arithmetic and physics specification. Explicit
rewrites adapt lambdas, references, and the browser buffer layout.
"""

import argparse
from pathlib import Path
import re
import struct
import subprocess

from pycparser import c_ast as A, c_parser

root = Path(__file__).resolve().parent.parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--output-dir", type=Path, default=root / "src/wasm/sim",
    help="directory for generated.wgsl and world.rs",
)
parser.add_argument("--check", action="store_true", help="compare outputs without writing files")
args = parser.parse_args()
subprocess.run(["python3", str(root / "gpu/gen_tables.py"), "--check"], check=True)
subprocess.run(["python3", str(root / "tools/ucrt/gen-math.py"), "--check"], check=True)
# Read the shared native GPU implementation.
texts = {
    n: (root / "gpu/sim" / n).read_text()
    for n in [
        "state.h",
        "world.h",
        "double_tables.h",
        "engine_exceptions.h",
        "math.h",
        "vec.h",
        "track.h",
        "rays.h",
        "sensors.h",
        "step.h",
        "stats.h",
    ]
}
# Inline lambdas as a single-iteration loop so lambda return does not return the caller.
# Preserve lambda early exits when the return is inside a nested loop.
texts["step.h"] = texts["step.h"].replace(
    "for (uint32_t j = 0; j < retained; j++)\n            if (car.pairs[j] == index) return;",
    "bool duplicate = false; for (uint32_t j = 0; j < retained; j++) duplicate |= car.pairs[j] == index; if (duplicate) return;",
)
for name in ["track.h", "step.h"]:
    s = texts[name]
    pattern = r"auto (\w+) = \[&\]\(([^)]*)\) \{"
    while m := re.search(pattern, s):
        start = m.end()
        depth = 1
        end = start
        while depth:
            if s[end] == "{":
                depth += 1
            elif s[end] == "}":
                depth -= 1
            end += 1
        body = s[start : end - 1]
        param = m.group(2)
        fn = m.group(1)
        s = s[: m.start()] + s[end + 1 :]
        # All named lambda calls have simple args except expand_rect's balanced expression.
        fs = s.rfind("__device__", 0, m.start())
        fb = s.index("{", fs)
        fend = fb + 1
        fd = 1
        while fd:
            if s[fend] == "{":
                fd += 1
            elif s[fend] == "}":
                fd -= 1
            fend += 1
        tail = s[fend:]
        s = s[:fend]
        pos = m.start()
        while c := re.search(r"\b" + fn + r"\(", s[pos:]):
            a = pos + c.start()
            b = pos + c.end()
            d = 1
            e = b
            while d:
                if s[e] == "(":
                    d += 1
                elif s[e] == ")":
                    d -= 1
                e += 1
            arg = s[b : e - 1]
            pname = param.split()[-1]
            unique = "lambda_" + fn + "_" + pname
            lbody = re.sub(r"\b" + pname + r"\b", unique, body)
            lparam = param[: param.rfind(pname)] + unique
            replacement = (
                "{ "
                + lparam
                + " = "
                + arg
                + "; do { "
                + re.sub(r"\breturn;", "break;", lbody)
                + " } while (false); }"
            )
            s = s[:a] + replacement + s[e:]
            pos = a + len(replacement)
        s += tail
    texts[name] = s
# The Windows math kernels (math/kernels) replace math.h's include of them,
# without the parts WGSL leaves out: log, and the sinf/cosf exceptions, which
# the browser applies on the CPU (sensor offsets) instead. Their specifier
# macros become plain functions and constants.
kernels = root / "math/kernels"
ucrt = (kernels / "ucrt_tables.h").read_text() + (kernels / "ucrt.h").read_text()
ucrt = re.sub(r"#ifndef ALTD_MATH_WGSL\n.*?#endif\n", "", ucrt, flags=re.S)
ucrt = (
    ucrt.replace("ALTD_MATH_FN ", "")
    .replace("ALTD_MATH_TABLE ", "const ")
    .replace("ALTD_MATH_CONST ", "const ")
)
assert "ALTD_MATH" not in re.sub(r"//[^\n]*", "", ucrt)
s = texts["math.h"]
s = re.sub(r"#define ALTD_MATH_\w+[^\n]*\n", "", s)
s = s.replace('#include "../../math/kernels/ucrt.h"', ucrt)
# Ray sensors read their precomputed offsets, so sin and cos keep musl's.
s = re.sub(r"__device__ inline float profile_(?:sin|cos)\(.*?\n\}\n", "", s, flags=re.S)
texts["math.h"] = s
s = texts["step.h"]
# Naga's SPIR-V backend panics on dynamically indexed private pointers passed
# to functions. Keep contact references inside refresh_contact and pass its
# index instead, preserving the original in-place updates without a copy.
s = s.replace(
    "refresh_contact(const World& w, Contact& c, Vec2 position, Vec2 bx, Vec2 by) {",
    "refresh_contact(const World& w, Car& contact_car, uint32_t contact_index, Vec2 position, Vec2 bx, Vec2 by) {\n"
    "    Contact& c = contact_car.contacts[contact_index];",
)
s = s.replace(
    "refresh_contact(w, c, position, bx, by)",
    "refresh_contact(w, car, i, position, bx, by)",
)
s = re.sub(r"template <[^\n]+>\n", "", s)
s = s.replace(
    "const Car& car, const CarFrame& f, uint32_t ci,\n                                         Vec2 push, Add&& add",
    "Car& car, const CarFrame& f, uint32_t ci, Vec2 push",
)
s = s.replace("add(c);", "merge_contact(v, car, c);")
s = s.replace(
    "const StepCarry& k, PushOf&& push_of, bool drive", "const StepCarry& k, bool drive"
)
s = s.replace("Vec2 push = push_of(ci);", "Vec2 push = sat_pair(w, v, car, ci);")
s = re.sub(
    r"pair_contacts\(w, v, car, f, ci, push, \[&\]\(const Contact& c\) \{.*?\}\)",
    "pair_contacts(w, v, car, f, ci, push)",
    s,
    flags=re.S,
)
s = s.replace(
    "step_end(w, v, car, k, [&](uint32_t ci) { return sat_pair(w, v, car, ci); }, drive, eliminate_on_wall)",
    "step_end(w, v, car, k, drive, eliminate_on_wall)",
)
s = s.replace("Vec2 (*out)[2]", "__ref_Pairs out")
s = s.replace('static_assert(MAX_PAIRS <= 32, "pair mask");', "")
texts["step.h"] = s
s = texts["sensors.h"]
s = s.replace(
    "float angle = (s.a - 90.0f) * (PI_F / 180.0f);\n            Vec2 local = Vec2{profile_cos(w.math_profile, angle, err) * s.b, profile_sin(w.math_profile, angle, err) * s.b};",
    "Vec2 local = s.offset;",
)
assert "profile_cos" not in s and "profile_sin" not in s
texts["sensors.h"] = s
s = texts["track.h"].replace("int64_t", "int32_t").replace("sat_i64", "sat_i32")
s = (
    s.replace("const TileCell* tile_at", "int tile_at")
    .replace("return nullptr;", "return -1;")
    .replace("return &w.tiles[y * w.tile_nx + x];", "return y * w.tile_nx + x;")
)
s = (
    s.replace("const TileCell* t = tile_at", "int t = tile_at")
    .replace(
        "return t ? t->surface : w.default_surface",
        "return t >= 0 ? w.tiles[t].surface : w.default_surface",
    )
    .replace("!t || !t->connected", "t < 0 || !w.tiles[t].connected")
    .replace("t->first", "w.tiles[t].first")
    .replace("t->last", "w.tiles[t].last")
)
texts["track.h"] = s
s = texts["vec.h"]
s = s[: s.index("// Rust `as i64`")] + s[s.index("// `godot_math::transform_point`") :]
texts["vec.h"] = (
    s.replace("inline Vec2 operator+", "inline Vec2 vec_add")
    .replace("inline Vec2 operator-", "inline Vec2 vec_sub")
    .replace("inline Vec2 operator*", "inline Vec2 vec_mul")
)
# Overload normalized has a separate name for raw controls.
texts["step.h"] = (
    texts["step.h"]
    .replace("Controls normalized(", "Controls normalized_controls(")
    .replace(
        "Controls control = normalized(raw);",
        "Controls control = normalized_controls(raw);",
    )
)
# Bounds and offsets in exported browser tracks fit signed 32-bit values.
texts["world.h"] = texts["world.h"].replace("int64_t tile_", "int32_t tile_")
# Emit parser-friendly C; references are marked typedefs and resolved by the emitter.
out = "typedef unsigned int uint8_t; typedef unsigned int uint32_t; typedef int int32_t; typedef unsigned long long uint64_t; typedef long long int64_t; typedef unsigned int size_t; typedef int bool;\n"
out += "typedef struct float2 { float x,y; } float2; typedef struct float4 { float x,y,z,w; } float4;\n"
for name, s in texts.items():
    s = re.sub(r"//[^\n]*|/\*.*?\*/", "", s, flags=re.S)
    if name == "step.h":
        s = re.sub(r"#ifdef ALTD_STEP_TIMING.*?#endif", "", s, flags=re.S)
    if name == "math.h":
        s = re.sub(r"#ifdef ALTD_TANH_BRANCHY.*?#else", "", s, flags=re.S)
        s = s.replace("#endif", "")
    s = re.sub(r"^\s*#.*$", "", s, flags=re.M)
    s = s.replace("namespace altd {", "")
    s = re.sub(r"\}\s*$", "", s)
    s = (
        s.replace("__device__", "")
        .replace("__host__", "")
        .replace("constexpr", "const")
        .replace("inline", "")
        .replace("nullptr", "0")
    )
    s = re.sub(r"\bSTEP_(?:MARK|SECTION|COUNT|ADD)\([^)]*\);?", "", s)
    # Explicit binary bit casts.
    s = re.sub(
        r"__builtin_bit_cast\((\w+),\s*((?:\(\w+\))?(?:0x[0-9a-fA-F]+ull|[a-zA-Z_]\w*))\)",
        lambda m: {
            "double": "dfrom",
            "float": "ffrom",
            "uint32_t": "fbits",
            "uint64_t": "dbits",
        }[m[1]]
        + "("
        + m[2]
        + ")",
        s,
    )
    # Drop original bitcast wrappers (provided in WGSL).
    if name == "math.h":
        s = re.sub(
            r"\s*(?:uint32_t|float|uint64_t|double) (?:fbits|ffrom|dbits|dfrom)\([^\n]+\n",
            "\n",
            s,
        )
    s = re.sub(r"const (\w+)&", r"const \1", s)
    s = re.sub(r"(\w+)&", r"__ref_\1 ", s)
    # Compound literals, including nested typed literals.
    s = re.sub(r"\b([A-Z]\w*)\{", r"(\1){", s)
    # Struct tags double as C typedef names.
    s = re.sub(r"struct (\w+)\s*\{", r"typedef struct \1 {", s)
    # Close each top-level struct with its typedef name.
    for m in list(re.finditer(r"typedef struct (\w+) \{", s))[::-1]:
        d = 1
        e = m.end()
        while d:
            if s[e] == "{":
                d += 1
            elif s[e] == "}":
                d -= 1
            e += 1
        s = s[:e] + " " + m[1] + s[e:]
    # Name anonymous enums.
    s = re.sub(r"enum[^\{]*\{", "enum {", s)
    s = s.replace("true", "1").replace("false", "0").replace("{}", "{0}")
    s = re.sub(r"(\w+)\{0\};", r"\1 = {0};", s)
    s = re.sub(r"(0x[\da-fA-F]+)p(-?\d+)", lambda m: str(float.fromhex(m[0])), s)
    if name == "step.h":
        s = "typedef Vec2 Pairs[2][2]; typedef Pairs __ref_Pairs;\n" + s
    refs = set(re.findall(r"\b__ref_(\w+)", s)) - {"Pairs"}
    for t in refs:
        s = "typedef " + t + " __ref_" + t + ";\n" + s
    out += "\n" + s


ast = c_parser.CParser().parse(out)
structs = {}
typedefs = {}
glob = {}
vals = {}
funcs = {}
layouts = {}
BASE = {
    "float": "f32",
    "double": "f64",
    "uint32_t": "u32",
    "uint8_t": "u32",
    "int32_t": "i32",
    "int": "i32",
    "size_t": "u32",
    "uint64_t": "u64",
    "int64_t": "i64",
    "bool": "bool",
    "void": "void",
    "float2": "vec2",
    "float4": "vec4",
    "Vec2": "vec2",
}


def typ(n):
    if isinstance(n, (A.TypeDecl, A.Typename)):
        return typ(n.type)
    if isinstance(n, A.IdentifierType):
        s = " ".join(n.names)
        if s.startswith("__ref_"):
            return ("ref", typ_name(s[6:]))
        return BASE.get(s, typedefs.get(s, s))
    if isinstance(n, A.Struct):
        return BASE.get(n.name, n.name)
    if isinstance(n, A.ArrayDecl):
        return (
            "arr",
            typ(n.type),
            int(n.dim.value) if isinstance(n.dim, A.Constant) else vals[n.dim.name],
        )
    if isinstance(n, A.PtrDecl):
        return ("ptr", typ(n.type))
    if isinstance(n, A.FuncDecl):
        return typ(n.type)
    raise ValueError(("type", n))


def typ_name(n):
    return BASE.get(n, typedefs.get(n, n))


def wg(t):
    if isinstance(t, tuple):
        if t[0] == "arr":
            return f"array<{wg(t[1])}, {t[2]}>"
        if t[0] == "ref":
            return f'ptr<{"private" if t[1]=="Contact" else "function"}, {wg(t[1])}>'
        if t[0] == "ptr":
            return "u32"
    return {
        "f64": "F64",
        "u64": "U64",
        "i64": "U64",
        "vec2": "vec2<f32>",
        "vec4": "vec4<f32>",
    }.get(t, t)


def constint(n):
    if isinstance(n, A.Constant):
        return int(re.sub(r"[uUlL]+$", "", n.value), 0)
    if isinstance(n, A.ID):
        return vals[n.name]
    if isinstance(n, A.BinaryOp):
        return {"<<": lambda a, b: a << b, "|": lambda a, b: a | b}[n.op](
            constint(n.left), constint(n.right)
        )
    raise ValueError(n)


for n in ast.ext:
    if isinstance(n, A.Typedef):
        if isinstance(n.type, A.TypeDecl) and isinstance(n.type.type, A.Struct):
            st = n.type.type
            if st.decls:
                structs[n.name] = [(d.name, typ(d.type)) for d in st.decls]
        else:
            typedefs[n.name] = typ(n.type)
    if isinstance(n, A.Decl):
        if isinstance(n.type, A.Enum):
            value = 0
            for en in n.type.values.enumerators:
                if en.value:
                    value = constint(en.value)
                glob[en.name] = "u32"
                vals[en.name] = value
                value += 1
        elif n.name:
            glob[n.name] = typ(n.type)
            if n.init and glob[n.name] in ["i32", "u32"]:
                vals[n.name] = constint(n.init)
    if isinstance(n, A.FuncDef):
        funcs[n.decl.name] = n


# Scalar/vector raw layout follows repr(C), including C Vec2 alignment of four.
def layout(t):
    if isinstance(t, tuple):
        if t[0] == "ptr":
            return 4, 4
        if t[0] == "arr":
            size, align = layout(t[1])
            return size * t[2], align
    if t in ["f64", "u64", "i64"]:
        return 8, 8
    if t in ["i32", "u32", "f32", "bool"]:
        return 4, 4
    if t == "vec2":
        return 8, 4
    if t == "vec4":
        return 16, 16
    if t in layouts:
        return layouts[t][0:2]
    off = 0
    align = 1
    fields = []
    for name, ft in structs[t]:
        sz, al = layout(ft)
        off = (off + al - 1) // al * al
        fields.append((name, ft, off))
        off += sz
        align = max(align, al)
    size = (off + align - 1) // align * align
    layouts[t] = (size, align, fields)
    return size, align


for st in structs:
    layout(BASE.get(st, st))
ENV = {}
ALIASES = {}
POINTERS = {}
FUN = ""
RETURN = ""
TEMP = 0
reserved = {
    "private",
    "from",
    "workgroup",
    "target",
    "filter",
    "active",
    "input",
    "output",
    "mod",
    "common",
    "shared",
    "partition",
    "patch",
    "resource",
    "sample",
    "typedef",
    "enum",
    "namespace",
    "register",
    "this",
    "template",
    "self",
}


def name(n):
    return "s_" + n if n in reserved else n


def kind(n):
    if isinstance(n, A.ID):
        return ENV.get(
            n.name,
            glob.get(
                n.name,
                {
                    "INFINITY": "f64",
                    "FLT_MAX": "f32",
                    "FLT_EPSILON": "f32",
                    "INT64_MAX": "i64",
                    "INT64_MIN": "i64",
                }.get(n.name, "unknown"),
            ),
        )
    if isinstance(n, A.Constant):
        return (
            "f32"
            if n.type == "float"
            else (
                "f64"
                if n.type == "double"
                else (
                    "u64"
                    if "long" in n.type
                    else "u32" if "unsigned" in n.type else "i32"
                )
            )
        )
    if isinstance(n, A.StructRef):
        t = kind(n.name)
        if isinstance(t, tuple) and t[0] == "ref":
            t = t[1]
        if t in ["vec2", "vec4"]:
            return "f32"
        return dict(structs[t])[n.field.name]
    if isinstance(n, A.ArrayRef):
        t = kind(n.name)
        if isinstance(t, tuple) and t[0] == "ref":
            t = t[1]
        return t[1]
    if isinstance(n, A.Cast):
        return typ(n.to_type)
    if isinstance(n, A.CompoundLiteral):
        return typ(n.type)
    if isinstance(n, A.UnaryOp):
        return "bool" if n.op == "!" else kind(n.expr)
    if isinstance(n, A.BinaryOp):
        if n.op in ["&&", "||", "<", ">", "<=", ">=", "==", "!="]:
            return "bool"
        if n.op in ["<<", ">>"]:
            return unref(kind(n.left))
        a, b = unref(kind(n.left)), unref(kind(n.right))
        if isinstance(a, tuple) and a[0] == "ptr":
            return a
        if isinstance(b, tuple) and b[0] == "ptr":
            return b
        return promote(a, b)
    if isinstance(n, A.TernaryOp):
        return promote(unref(kind(n.iftrue)), unref(kind(n.iffalse)))
    if isinstance(n, A.Assignment):
        return unref(kind(n.lvalue))
    if isinstance(n, A.FuncCall):
        f = n.name.name
        if f in funcs:
            return typ(funcs[f].decl.type.type)
        return {
            "fbits": "u32",
            "ffrom": "f32",
            "dbits": "u64",
            "dfrom": "f64",
            "fabs": "f64",
            "fabsf": "f32",
            "floor": "f64",
            "rint": "f64",
            "round": "f64",
            "sqrtf": "f32",
            "fmodf": "f32",
            "fmax": "f64",
            "fma": "f64",
            "copysignf": "f32",
            "isfinite": "bool",
            "isnan": "bool",
            "isinf": "bool",
            "signbit": "bool",
            "make_float2": "vec2",
            "__double2int_rz": "i32",
            "sat_i32": "i32",
        }.get(
            f,
            (
                kind(n.args.exprs[0])
                if f in ["min", "max", "altd_invalid", "altd_nan_operand"]
                else "unknown"
            ),
        )
    if isinstance(n, A.ExprList):
        return kind(n.exprs[-1])
    raise ValueError(("kind", type(n), n))


def unref(t):
    return t[1] if isinstance(t, tuple) and t[0] == "ref" else t


def promote(a, b):
    if a == b:
        return a
    if a in ["vec2", "vec4"]:
        return a
    if b in ["vec2", "vec4"]:
        return b
    for t in ["f64", "f32", "u64", "i64", "u32", "i32"]:
        if t in [a, b]:
            return t
    if isinstance(a, tuple):
        return a
    raise ValueError(("promote", a, b))


def cast(s, a, b):
    a = unref(a)
    b = unref(b)
    if a == b:
        return s
    if b == "bool":
        if a in ["u64", "i64"]:
            return f"u64_nonzero({s})"
        if a == "f64":
            return f"!f64_eq({s}, F64_ZERO)"
        if a in ["vec2", "vec4"]:
            raise ValueError("vecbool")
        return f"({s} != {wg(a)}(0))"
    if a == "f64":
        if b == "f32":
            return f"f64_to_f32({s})"
        if b == "u32":
            return f"u32(f64_to_i64({s}).x)"
        if b == "i32":
            return f"f64_to_i32({s})"
        if b in ["u64", "i64"]:
            return f"f64_to_{b}({s})"
    if b == "f64":
        return f"f64_from_{a}({s})"
    if a in ["u64", "i64"]:
        if b in ["u64", "i64"]:
            return s
        return f"{wg(b)}({s}.x)"
    if b in ["u64", "i64"]:
        if a == "i32":
            return f"i64_from_i32({s})"
        return f"U64(u32({s}), 0u)"
    return f"{wg(b)}({s})"


def E(n, t=None):
    s = expr(n)
    return cast(s, kind(n), t) if t else s


def ptr(n):
    if isinstance(n, A.ID):
        if n.name in POINTERS:
            return POINTERS[n.name]
        t = unref(kind(n))
        if isinstance(t, tuple) and t[0] == "arr":
            return ("array", name(n.name), "0u", t[1])
    if isinstance(n, A.StructRef):
        return ("data", E(n), "0u", kind(n)[1])
    if isinstance(n, A.BinaryOp) and n.op == "+":
        buf, base, off, t = ptr(n.left)
        return buf, base, f'({off} + {E(n.right,"u32")})', t
    raise ValueError(("pointer", n))


def arrval(n):
    buf, base, off, t = ptr(n.name)
    idx = E(n.subscript, "u32")
    if buf == "array":
        return f"{base}[{off} + {idx}]"
    return load(t, f"{base} + ({off} + {idx}) * {layout(t)[0]//4}u", "world_data")


def expr(n):
    global TEMP
    if isinstance(n, A.ID):
        if n.name in ALIASES:
            return ALIASES[n.name]
        special = {
            "INFINITY": "F64(0u, 0x7ff00000u)",
            "FLT_EPSILON": "bitcast<f32>(0x34000000u)",
            "FLT_MAX": "bitcast<f32>(0x7f7fffffu)",
            "INT64_MAX": "U64(0xffffffffu, 0x7fffffffu)",
            "INT64_MIN": "U64(0u, 0x80000000u)",
        }
        if n.name in special:
            return special[n.name]
        if isinstance(kind(n), tuple) and kind(n)[0] == "ref":
            return f"(*{name(n.name)})"
        return name(n.name)
    if isinstance(n, A.Constant):
        t = kind(n)
        v = (
            re.sub(r"[uUlL]+$", "", n.value)
            if "int" in n.type
            else re.sub(r"[fF]$", "", n.value)
        )
        if t in ["f32", "f64"]:
            x = float.fromhex(v) if "0x" in v else float(v)
            if t == "f32":
                return (
                    f'bitcast<f32>(0x{struct.unpack("<I",struct.pack("<f",x))[0]:08x}u)'
                )
            lo, hi = struct.unpack("<II", struct.pack("<d", x))
            return f"F64(0x{lo:08x}u, 0x{hi:08x}u)"
        x = int(v, 0)
        if t == "u64":
            return f"U64(0x{x&0xffffffff:08x}u, 0x{x>>32:08x}u)"
        return v + ("u" if t == "u32" else "i")
    if isinstance(n, A.StructRef):
        return f"{E(n.name)}.{name(n.field.name)}"
    if isinstance(n, A.ArrayRef):
        t = unref(kind(n.name))
        if t[0] == "ptr" or isinstance(n.name, A.ID) and n.name.name in POINTERS:
            return arrval(n)
        return f'{E(n.name)}[{E(n.subscript,"u32")}]'
    if isinstance(n, A.Cast):
        return E(n.expr, typ(n.to_type))
    if isinstance(n, A.CompoundLiteral):
        return init(n.init, typ(n.type))
    if isinstance(n, A.BinaryOp):
        return binary(n.op, n.left, n.right)
    if isinstance(n, A.UnaryOp):
        op = n.op
        t = unref(kind(n.expr))
        v = E(n.expr)
        if op == "!":
            return f'!({E(n.expr,"bool")})'
        if op in ["-", "+", "~"]:
            if op == "+":
                return v
            if t == "f64":
                return f"f64_neg({v})"
            if t == "f32" and op == "-":
                return f"fp_neg({v})"
            if t in ["u64", "i64"]:
                return f"i64_neg({v})" if op == "-" else f"~({v})"
            if t == "u32" and op == "-":
                return f"(0u - {v})"
            return f"{op}({v})"
        if op in ["p++", "p--", "++", "--"]:
            # Post increment's previous value, including embedded subscripts.
            TEMP += 1
            tmp = f"post_{TEMP}"
            before = f"let {tmp} = {v};"
            inc = arithmetic("+" if "+" in op else "-", v, one(t), t)
            PRE.append(before)
            PRE.append(f"{v} = {inc};")
            return tmp if op[0] == "p" else v
        if op == "&":
            return f"&({v})"
        raise ValueError(("unary", op))
    if isinstance(n, A.TernaryOp):
        t = kind(n)
        a = E(n.iftrue, t)
        b = E(n.iffalse, t)
        c = E(n.cond, "bool")
        if (
            isinstance(t, tuple)
            and t[0] == "arr"
            or t in structs
            and t not in ["vec2", "vec4"]
        ):
            TEMP += 1
            v = f"choice_{TEMP}"
            PRE.append(
                f"var {v}: {wg(t)}; if ({c}) {{ {v} = {a}; }} else {{ {v} = {b}; }}"
            )
            return v
        return f'{"fp_select" if t == "f32" else "select"}({b}, {a}, {c})'
    if isinstance(n, A.FuncCall):
        f = n.name.name
        args = n.args.exprs if n.args else []
        if f in funcs:
            params = funcs[f].decl.type.args.params if funcs[f].decl.type.args else []
            out = []
            for p, a in zip(params, args):
                t = typ(p.type)
                if erase(t):
                    continue
                if isinstance(t, tuple) and t[0] == "ref":
                    if (
                        isinstance(a, A.ID)
                        and a.name not in ALIASES
                        and isinstance(kind(a), tuple)
                        and kind(a)[0] == "ref"
                    ):
                        out.append(name(a.name))
                    else:
                        out.append(f"&({E(a)})")
                elif isinstance(t, tuple) and t[0] == "ptr":
                    if f in ["normalized_controls", "step_begin", "car_step"]:
                        out.append(E(a))
                    elif f == "rectangle_supports":
                        out.append(E(a))
                    else:
                        raise ValueError(("paramptr", f))
                else:
                    out.append(E(a, t))
            return f'{f}({", ".join(out)})'
        t = unref(kind(args[0])) if args else None
        if f in ["fbits", "ffrom"]:
            if f == "ffrom":
                # A constant -0 float can be canonicalized by a driver. Keep
                # explicit bit patterns opaque with the uploaded zero pad.
                return f'bitcast<f32>({E(args[0])} ^ params.pad)'
            return f'bitcast<u32>({E(args[0])})'
        if f in ["dbits", "dfrom"]:
            return E(args[0])
        if f in ["fabs", "fabsf"]:
            return f'{"f64_abs" if t=="f64" else "fp_abs"}({E(args[0])})'
        if f in ["isnan", "isinf", "isfinite", "signbit"]:
            return f'{"f64_" if t=="f64" else "fp_"}{f}({E(args[0])})'
        if f in ["floor", "rint", "round"]:
            return f'f64_{f}({E(args[0],"f64")})'
        if f == "sqrtf":
            return f'fp_sqrt({E(args[0],"f32")})'
        if f == "fmodf":
            return f'fp_mod({E(args[0],"f32")}, {E(args[1],"f32")})'
        if f == "fmax":
            return f'f64_max({E(args[0],"f64")}, {E(args[1],"f64")})'
        if f == "fma":
            return f'f64_fma({E(args[0],"f64")}, {E(args[1],"f64")}, {E(args[2],"f64")})'
        if f == "copysignf":
            return f"fp_copysign({E(args[0])}, {E(args[1])})"
        if f == "altd_invalid":
            return f'{"f64_" if t=="f64" else "fp_"}invalid({E(args[0],t)})'
        if f == "altd_nan_operand":
            return f'{"f64_" if t=="f64" else "fp_"}nan_operand({E(args[0],t)}, {E(args[1],t)})'

        if f == "make_float2":
            return f'vec2<f32>({E(args[0],"f32")}, {E(args[1],"f32")})'
        if f == "__double2int_rz":
            return f"f64_to_i32({E(args[0])})"
        if f == "sat_i32":
            return f"f64_to_i32({E(args[0])})"
        if f in ["min", "max"]:
            return f"{f}({E(args[0],t)}, {E(args[1],t)})"
        raise ValueError(("call", f))
    if isinstance(n, A.ExprList):
        return E(n.exprs[-1])
    raise ValueError(("expr", type(n), n))


def one(t):
    return (
        "F64_ONE"
        if t == "f64"
        else (
            "U64(1u,0u)"
            if t in ["i64", "u64"]
            else "1u" if t == "u32" else "1.0f" if t == "f32" else "1i"
        )
    )


def arithmetic(op, a, b, t):
    if t == "f64":
        return f"f64_{{}}({a}, {b})".format(
            {
                "+": "add",
                "-": "sub",
                "*": "mul",
                "/": "div",
                "<": "lt",
                "<=": "le",
                ">": "gt",
                ">=": "ge",
                "==": "eq",
                "!=": "ne",
            }[op]
        )
    if t in ["u64", "i64"]:
        if op in [
            "+",
            "-",
            "*",
            "/",
            "%",
            "<<",
            ">>",
            "<",
            ">",
            "<=",
            ">=",
            "==",
            "!=",
        ]:
            f = {
                "+": "u64_add",
                "-": "u64_sub",
                "*": "u64_mul_low",
                "/": "u64_div_small",
                "%": "u64_mod_small",
                "<<": "u64_shl",
                ">>": "i64_shr" if t == "i64" else "u64_shr",
                "<": "i64_lt" if t == "i64" else "u64_lt",
                ">": "i64_gt" if t == "i64" else "u64_gt",
                "<=": "i64_le" if t == "i64" else "u64_le",
                ">=": "i64_ge" if t == "i64" else "u64_ge",
                "==": "u64_eq",
                "!=": "u64_ne",
            }[op]
            return f"{f}({a}, {b})"
    if t in ["f32", "vec2", "vec4"] and op in ["+", "-", "*", "/"]:
        return f'{"fp" if t=="f32" else "v2" if t=="vec2" else "v4"}_{ {"+":"add","-":"sub","*":"mul","/":"div"}[op]}({a}, {b})'
    return f"({a} {op} {b})"


def binary(op, a, b):
    if op in ["&&", "||"]:
        return f'({E(a,"bool")} {op} {E(b,"bool")})'
    ta, tb = unref(kind(a)), unref(kind(b))
    t = promote(ta, tb)
    if op in ["<<", ">>"]:
        t = ta
    aa = E(a, t)
    bb = (
        E(b, "u32")
        if op in ["<<", ">>"] or t in ["u64", "i64"] and op in ["/", "%"]
        else E(b, t)
    )
    return arithmetic(op, aa, bb, t)


def init(n, t):
    if not isinstance(n, A.InitList):
        return E(n, t)
    items = n.exprs or []
    if len(items) == 1 and isinstance(items[0], A.Constant) and items[0].value == "0":
        return f"{wg(t)}()"
    if isinstance(t, tuple) and t[0] == "arr":
        types = [t[1]] * t[2]
    elif t in ["vec2", "vec4"]:
        types = ["f32"] * (2 if t == "vec2" else 4)
    else:
        types = [ft for _, ft in structs[t]]
    args = [init(v, ft) for v, ft in zip(items, types)]
    return f'{wg(t)}({", ".join(args)})'


def erase(t):
    return (
        unref(t) in ["Car", "Agent", "World", "VehicleDesc"]
        if not isinstance(unref(t), tuple)
        else False
    )


PRE = []


def declaration(n):
    t = typ(n.type)
    ENV[n.name] = t
    if isinstance(t, tuple) and t[0] == "ref" and n.init:
        ALIASES[n.name] = E(n.init)
        return ""
    if isinstance(t, tuple) and t[0] == "ptr" and n.init:
        POINTERS[n.name] = ptr(n.init)
        return ""
    s = f"var {name(n.name)}: {wg(t)}"
    if n.init:
        s += " = " + init(n.init, t)
    return s + ";"


def statement(n):
    if n is None:
        return ""
    if isinstance(n, A.Compound):
        state = (ENV.copy(), ALIASES.copy(), POINTERS.copy())
        body = "{\n" + "".join(stmt(v) + "\n" for v in n.block_items or []) + "}\n"
        ENV.clear()
        ENV.update(state[0])
        ALIASES.clear()
        ALIASES.update(state[1])
        POINTERS.clear()
        POINTERS.update(state[2])
        return body
    if isinstance(n, A.Decl):
        return declaration(n)
    if isinstance(n, A.DeclList):
        return "\n".join(declaration(d) for d in n.decls)
    if isinstance(n, A.Return):
        return "return" + (" " + E(n.expr, RETURN) if n.expr else "") + ";"
    if isinstance(n, A.Assignment):
        t = unref(kind(n.lvalue))
        a = E(n.lvalue)
        b = E(n.rvalue, "u32" if n.op in ["<<=", ">>="] else t)
        if n.op != "=":
            b = arithmetic(n.op[:-1], a, b, t)
        return f"{a} = {b};"
    if isinstance(n, A.If):
        c = E(n.cond, "bool")
        s = f"if ({c}) " + block(n.iftrue)
        if n.iffalse:
            s += " else " + block(n.iffalse)
        return s
    if isinstance(n, A.For):
        # Loop init remains in a scope, and continuing runs on continue.
        start = stmt(n.init)
        c = E(n.cond, "bool") if n.cond else "true"
        condpre = "\n".join(PRE)
        PRE.clear()
        return (
            "{\n"
            + start
            + "\nloop {\n"
            + condpre
            + f"\nif (!({c})) {{ break; }}\n"
            + stmt(n.stmt)
            + "\ncontinuing {\n"
            + stmt(n.next)
            + "\n}\n}\n}"
        )
    if isinstance(n, A.While):
        c = E(n.cond, "bool")
        condpre = "\n".join(PRE)
        PRE.clear()
        return (
            "loop {\n"
            + condpre
            + f"\nif (!({c})) {{ break; }}\n"
            + stmt(n.stmt)
            + "\n}"
        )
    if isinstance(n, A.DoWhile):
        return "loop " + block(n.stmt).rstrip()[:-1] + "\nbreak;\n}"
    if isinstance(n, A.Break):
        return "break;"
    if isinstance(n, A.Continue):
        return "continue;"
    if isinstance(n, A.Switch):
        body = block(n.stmt)
        if not any(isinstance(v, A.Default) for v in n.stmt.block_items or []):
            body = body.rstrip()[:-1] + "default: {}\n}"
        return f"switch ({E(n.cond)}) " + body
    if isinstance(n, A.Case):
        return (
            f"case {E(n.expr,kind(SWITCH[-1]))}: "
            + "{\n"
            + "".join(stmt(s) + "\n" for s in n.stmts or [])
            + "}"
        )
    if isinstance(n, A.Default):
        return "default: {\n" + "".join(stmt(s) + "\n" for s in n.stmts or []) + "}"
    if isinstance(n, A.EmptyStatement):
        return ""
    if isinstance(n, A.UnaryOp) and n.op in ["p++", "p--", "++", "--"]:
        a = E(n.expr)
        t = unref(kind(n.expr))
        return f'{a} = {arithmetic("+" if "+" in n.op else "-",a,one(t),t)};'
    if isinstance(n, A.ExprList):
        return "\n".join(stmt(s) for s in n.exprs)
    if isinstance(n, A.FuncCall):
        return E(n) + ";"
    raise ValueError(("stmt", type(n), n))


SWITCH = []


def stmt(n):
    global PRE
    before = PRE
    PRE = []
    if isinstance(n, A.Switch):
        SWITCH.append(n.cond)
    s = statement(n)
    if isinstance(n, A.Switch):
        SWITCH.pop()
    p = PRE
    PRE = before
    return "\n".join(p + [s])


def block(n):
    return stmt(n) if isinstance(n, A.Compound) else "{\n" + stmt(n) + "\n}"


def load(t, off, buf):
    if isinstance(t, tuple):
        if t[0] == "ptr":
            return f"{buf}[{off}]"
        if t[0] == "arr":
            return (
                f"{wg(t)}("
                + ", ".join(
                    load(t[1], f"({off} + {i*layout(t[1])[0]//4}u)", buf)
                    for i in range(t[2])
                )
                + ")"
            )
    if t in ["f64", "i64", "u64"]:
        return f"U64({buf}[{off}], {buf}[{off} + 1u])"
    if t == "u32":
        return f"{buf}[{off}]"
    if t == "i32":
        return f"i32({buf}[{off}])"
    if t == "f32":
        return f"bitcast<f32>({buf}[{off}])"
    if t == "bool":
        return f"({buf}[{off}] != 0u)"
    if t in ["vec2", "vec4"]:
        return (
            f"{wg(t)}("
            + ", ".join(
                f"bitcast<f32>({buf}[{off} + {i}u])"
                for i in range(2 if t == "vec2" else 4)
            )
            + ")"
        )
    if buf == "world_data":
        return f"load_{t}({off})"
    return (
        f"{t}("
        + ", ".join(load(ft, f"({off} + {o//4}u)", buf) for _, ft, o in layouts[t][2])
        + ")"
    )


def store(t, off, v, buf):
    if isinstance(t, tuple):
        return "\n".join(
            store(t[1], f"({off} + {i*layout(t[1])[0]//4}u)", f"{v}[{i}u]", buf)
            for i in range(t[2])
        )
    if t in ["f64", "i64", "u64"]:
        return f"{buf}[{off}] = {v}.x; {buf}[{off} + 1u] = {v}.y;"
    if t in ["vec2", "vec4"]:
        return "\n".join(
            f"{buf}[{off} + {i}u] = bitcast<u32>({v}.{c});"
            for i, c in enumerate("xyzw"[: 2 if t == "vec2" else 4])
        )
    if t in ["u32", "i32", "f32", "bool"]:
        s = f"bitcast<u32>({v})" if t == "f32" else f"u32({v})"
        return f"{buf}[{off}] = {s};"
    return "\n".join(
        store(ft, f"({off} + {o//4}u)", f"{v}.{name(n)}", buf)
        for n, ft, o in layouts[t][2]
    )


lines = ["// Generated from gpu/sim by wasm/gen-sim.py. Do not edit this file."]
for st, fields in structs.items():
    if st in ["Vec2", "float2", "float4"]:
        continue
    lines.append(
        "struct "
        + st
        + " {\n"
        + "\n".join(f"    {name(n)}: {wg(t)}," for n, t in fields)
        + "\n}"
    )
lines += [
    "var<private> sim_world: World;",
    "var<private> sim_vehicle: VehicleDesc;",
    "var<private> sim_car: Car;",
    "var<private> sim_agent: Agent;",
]
for n, v in vals.items():
    lines.append(
        f'const {name(n)}: {wg(glob[n])} = {v}{"u" if glob[n]=="u32" else "i"};'
    )
for n in ast.ext:
    if isinstance(n, A.Decl) and n.name and n.init and n.name not in vals:
        ENV = {}
        ALIASES = {}
        lines.append(
            f"const {name(n.name)}: {wg(glob[n.name])} = {init(n.init,glob[n.name])};"
        )
for st in [
    "World",
    "NearGrid",
    "VehicleDesc",
    "WheelDesc",
    "RayNode",
    "TileCell",
    "PathSeg",
    "SurfaceDev",
    "ShapeDev",
    "SensorDesc",
]:
    fields = layouts[st][2]
    lines.append(
        f"fn load_{st}(o: u32) -> {st} {{ return {st}("
        + ", ".join(load(ft, f"(o + {off//4}u)", "world_data") for _, ft, off in fields)
        + "); }"
    )
# Keep state array copies as loops. Expanding 32 contacts into constructor
# expressions causes browser shader compilers to use several gigabytes.
serial_counter = 0


def load_into(t, off, v, buf):
    global serial_counter
    if isinstance(t, tuple) and t[0] == "arr":
        serial_counter += 1
        i = "copy_" + str(serial_counter)
        return (
            f"for (var {i}=0u; {i}<{t[2]}u+params.pad; {i}={i}+1u) {{\n"
            + load_into(t[1], f"({off}+{i}*{layout(t[1])[0]//4}u)", f"{v}[{i}]", buf)
            + "\n}"
        )
    if t in structs and t not in ["vec2", "vec4"]:
        return "\n".join(
            load_into(ft, f"({off}+{o//4}u)", f"{v}.{name(n)}", buf)
            for n, ft, o in layouts[t][2]
        )
    return f"{v} = {load(t,off,buf)};"


def store_loop(t, off, v, buf):
    global serial_counter
    if isinstance(t, tuple) and t[0] == "arr":
        serial_counter += 1
        i = "copy_" + str(serial_counter)
        return (
            f"for (var {i}=0u; {i}<{t[2]}u+params.pad; {i}={i}+1u) {{\n"
            + store_loop(t[1], f"({off}+{i}*{layout(t[1])[0]//4}u)", f"{v}[{i}]", buf)
            + "\n}"
        )
    if t in structs and t not in ["vec2", "vec4"]:
        return "\n".join(
            store_loop(ft, f"({off}+{o//4}u)", f"{v}.{name(n)}", buf)
            for n, ft, o in layouts[t][2]
        )
    return store(t, off, v, buf)


for st, buf in [("Car", "cars"), ("Agent", "agents")]:
    lines.append(
        f"fn load_{st}(o: u32) {{\n"
        + load_into(st, "o", "sim_" + st.lower(), buf)
        + "\n}"
    )
    lines.append(
        f"fn store_{st}(o: u32) {{\n"
        + store_loop(st, "o", "sim_" + st.lower(), buf)
        + "\n}"
    )
lines.append(
    "fn load_sensor_Car(o: u32) {\n"
    + "\n".join(
        load_into(ft, f"(o+{off//4}u)", f"sim_car.{name(n)}", "cars")
        for n, ft, off in layouts["Car"][2]
        if off < 96 or n == "error"
    )
    + "\n}"
)
lines.append(
    "fn load_StepCarry(o: u32) -> StepCarry { return "
    + load("StepCarry", "o", "results")
    + "; }"
)
lines.append(
    "fn store_StepCarry(o: u32, v: StepCarry) {\n"
    + store("StepCarry", "o", "v", "results")
    + "\n}"
)
for f, n in funcs.items():
    if f in ["vec_add", "vec_sub", "vec_mul"]:
        continue
    FUN = f
    RETURN = typ(n.decl.type.type)
    ENV = {}
    ALIASES = {}
    POINTERS = {}
    PRE = []
    params = []
    locals = []
    for p in n.decl.type.args.params if n.decl.type.args else []:
        if p.name is None:
            continue
        t = typ(p.type)
        ENV[p.name] = t
        if erase(t):
            ALIASES[p.name] = {
                "Car": "sim_car",
                "Agent": "sim_agent",
                "World": "sim_world",
                "VehicleDesc": "sim_vehicle",
            }[unref(t)]
            continue
        if isinstance(t, tuple) and t[0] == "ptr":
            count = 4 if f == "rectangle_supports" else 5
            t = ("arr", t[1], count)
            ENV[p.name] = t
        if isinstance(t, tuple) and t[0] == "ref":
            params.append(name(p.name) + ": " + wg(t))
        else:
            params.append(name(p.name) + "_arg: " + wg(t))
            locals.append(f"var {name(p.name)} = {name(p.name)}_arg;")
    try:
        body = stmt(n.body)
    except Exception as e:
        raise RuntimeError(f + " " + str(e)) from e
    lines.append(
        f"fn {f}("
        + ", ".join(params)
        + ")"
        + (" -> " + wg(RETURN) if RETURN != "void" else "")
        + " {\n"
        + "\n".join(locals)
        + "\n"
        + body[1:-2]
        + (
            "\nreturn " + wg(RETURN) + "();"
            if f in ["raycast", "closest_wall_point"]
            else ""
        )
        + "\n}"
    )
lines = [
    (
        re.sub(
            r"bitcast<f32>\(0x([0-9a-f]+)u\)",
            lambda m: repr(struct.unpack("<f", struct.pack("<I", int(m[1], 16)))[0])
            + "f",
            line,
        )
        if line.startswith("const ")
        else line
    )
    for line in lines
]
generated_shader = "\n\n".join(lines)
shader_sections = len(lines)

# Generate the pointer-free browser world layout from the same field offsets.
counts = {
    "ray_nodes": "w.ray_node_count as usize",
    "ray_walls": "w.ray_wall_count as usize",
    "surfaces": "w.surface_count as usize",
    "tiles": '(w.tile_nx as usize).checked_mul(w.tile_ny as usize).ok_or("tile count overflow")?',
    "path_segments": "w.path_points.saturating_sub(1) as usize",
    "path_offsets": "w.path_points as usize",
    "curve_position": "w.curve_points as usize",
    "curve_offsets": "w.curve_points as usize",
    "curve_forwards": "w.curve_points as usize",
    "shapes": "w.shape_count as usize",
    "shape_local": "w.shape_point_count as usize",
    "shape_normals": "w.shape_point_count as usize",
    "path_xy": "w.path_points as usize * 2",
}
lines = [
    "// Generated by wasm/gen-sim.py; offsets match generated.wgsl raw loaders.",
    "use crate::gpu::simulation::WorldDesc;",
    "use super::append_ptr;",
    "pub(super) fn encode_world(w: &WorldDesc) -> Result<Vec<u32>, String> {",
    "let mut data=vec![0u32;64];",
]


def fields(st, prefix, off):
    for n, t, o in layouts[st][2]:
        pos = (off + o) // 4
        v = prefix + "." + n
        if t == "NearGrid":
            lines.append(
                f'let cells=({v}.nx as usize).checked_mul({v}.ny as usize).ok_or("grid count overflow")?;'
            )
            lines.append(
                f"let count_{n}=if cells>0 {{ unsafe {{ *{v}.start.add(cells) as usize }} }} else {{ 0 }};"
            )
            fields("NearGrid", v, off + o)
        elif isinstance(t, tuple) and t[0] == "ptr":
            count = counts.get(
                n,
                (
                    "if cells>0 { cells+1 } else { 0 }"
                    if n == "start"
                    else "count_" + prefix.split(".")[-1]
                ),
            )
            lines.append(
                f"data[{pos}]=unsafe {{ append_ptr(&mut data,{v},{count}) }}?;"
            )
        elif t == "f64":
            lines.append(
                f"let bits={v}.to_bits(); data[{pos}]=bits as u32; data[{pos+1}]=(bits>>32) as u32;"
            )
        elif t == "f32":
            lines.append(f"data[{pos}]={v}.to_bits();")
        elif n.startswith("tile_"):
            lines.append(
                f'data[{pos}]=i32::try_from({v}).map_err(|_| "tile bounds exceed WebGPU range")? as u32;'
            )
        else:
            lines.append(f"data[{pos}]={v};")


fields("World", "w", 0)
lines += ["Ok(data)", "}"]
generated_world = subprocess.run(
    ["rustfmt", "--edition", "2021", "--emit", "stdout"],
    input="\n".join(lines) + "\n",
    text=True,
    capture_output=True,
    check=True,
    cwd=root,
).stdout
outputs = {"generated.wgsl": generated_shader, "world.rs": generated_world}
if args.check:
    stale = [
        str(args.output_dir / name)
        for name, generated in outputs.items()
        if not (args.output_dir / name).is_file()
        or (args.output_dir / name).read_bytes() != generated.encode()
    ]
    if stale:
        parser.exit(1, "Generated output differs or is missing: " + ", ".join(stale) + "\n")
    print(f"Verified {args.output_dir / 'generated.wgsl'} and {args.output_dir / 'world.rs'}")
else:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, generated in outputs.items():
        (args.output_dir / name).write_text(generated)
    print("Generated", shader_sections, "sections", len(generated_shader.encode()), "bytes")
