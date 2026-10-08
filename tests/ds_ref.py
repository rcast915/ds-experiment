"""
Pure-NumPy double-single (DS) reference arithmetic.

Implements the same algorithms as DsTransformPass.cpp — two_sum, two_prod,
Veltkamp split — in scalar NumPy so accuracy tests can compute expected DS
results without GPU hardware or the PJRT plugin.

No JAX or CUDA dependency: runs on any machine.
"""

import numpy as np


# ── Primitives ────────────────────────────────────────────────────────────────

def two_sum(a, b):
    """Error-free add: float(a+b) == s,  true(a+b) == s+e."""
    a = np.float32(a); b = np.float32(b)
    # errstate: NaN/Inf inputs produce NaN/Inf outputs by design.
    with np.errstate(invalid='ignore', over='ignore'):
        s = a + b
        v = s - a
        e = (a - (s - v)) + (b - v)
    return s, np.float32(e)


def veltkamp_split(a):
    """Split f32 into (hi, lo) with non-overlapping 12-bit halves."""
    a = np.float32(a)
    with np.errstate(invalid='ignore', over='ignore'):
        c = np.float32(4097.0) * a
        hi = c - (c - a)
    return hi, a - hi


def two_prod(a, b):
    """Error-free multiply: float(a*b) == p,  true(a*b) == p+e."""
    a = np.float32(a); b = np.float32(b)
    with np.errstate(invalid='ignore', over='ignore'):
        p = a * b
        ah, al = veltkamp_split(a)
        bh, bl = veltkamp_split(b)
        e = ((ah * bh - p) + ah * bl + al * bh) + al * bl
    return p, np.float32(e)


def fast_two_sum(a, b):
    """Dekker's fast form -- REQUIRES |a| >= |b|. Ported from
    double-single-lib's __fast_two_sum. 1 add + 2 subs, vs. two_sum's
    general 1 add + 4 subs."""
    a = np.float32(a); b = np.float32(b)
    with np.errstate(invalid='ignore', over='ignore'):
        h = a + b
        t = h - a
        l = b - t
    return h, np.float32(l)


# ── DS pair arithmetic ────────────────────────────────────────────────────────

def ds_add(ah, al, bh, bl):
    s1, e1 = two_sum(ah, bh)
    s2, e2 = two_sum(al, bl)
    t1, t2 = two_sum(s1, np.float32(s2 + e1))
    return t1, np.float32(t2 + e2)


def ds_sub(ah, al, bh, bl):
    return ds_add(ah, al, np.float32(-bh), np.float32(-bl))


def ds_mul(ah, al, bh, bl):
    p1, e1 = two_prod(ah, bh)
    cross = np.float32(ah * bl + al * bh)
    s, e2 = two_sum(p1, cross)
    return s, np.float32(e1 + e2 + al * bl)


def ds_div(ah, al, bh, bl):
    """Based on double-single-lib's double_binary32_div, WITH ONE
    DELIBERATE CORRECTION to a confirmed bug in the reference.

    __two_mul (the library's FMA-based TwoProduct) is substituted with
    two_prod (Veltkamp-split-based) here too, matching the same
    substitution DsTransformPass.cpp's emitDsDiv makes -- see its comment
    for why a real hardware FMA can't be faithfully reproduced with
    separate ops.

    DEVIATION FROM THE LIBRARY, deliberate: as literally written,
    double_binary32_div computes t7 = t5 + t6, never using t3 (the
    TwoProd/two_mul residual of bh*t1) again after computing it. This is
    a confirmed bug, not an intentional design choice (the library's own
    sibling routine for DS-by-scalar division keeps the equivalent term
    correctly -- see DsTransformPass.cpp's emitDsDiv comment for the full
    derivation and empirical confirmation, including why it's not limited
    to bl != 0 inputs). t7 here includes the `- t3` correction, restoring
    the ~2^-48-class double-word accuracy this algorithm's structure is
    designed to reach (confirmed empirically: worst case ~1.7e-14 over
    500k random trials) instead of the ~2^-23-class (f32-ULP-level)
    accuracy the literal library formula achieves.
    """
    ah = np.float32(ah); al = np.float32(al)
    bh = np.float32(bh); bl = np.float32(bl)
    with np.errstate(invalid='ignore', over='ignore', divide='ignore'):
        t1 = np.float32(ah / bh)
        t2, t3 = two_prod(bh, t1)
        t4 = np.float32(bl * t1)
        t5 = np.float32(ah - t2)   # Sterbenz: exact
        t6 = np.float32(al - t4)
        t7 = np.float32(np.float32(t5 + t6) - t3)   # correction: see docstring
        t8 = np.float32(t7 / bh)
    return fast_two_sum(t1, t8)


def ds_div_by_scalar(ah, al, b):
    """Ported exactly from double-single-lib's
    __double_binary_div_double_by_single -- divides a DS pair by a plain
    scalar. Used only by ds_sqrt's internal refinement step.

    NOT the same as ds_div(ah, al, b, 0.0): unlike double_binary32_div
    (see ds_div's docstring), this routine correctly keeps the TwoProd
    residual (t3) in its correction sum (t5 = al - t3) rather than
    dropping it -- confirmed by reading the library source, this is a
    genuine difference between the two routines, not two equivalent
    formulations. ds_sqrt uses this one because that's what
    double_binary32_sqrt actually calls.
    """
    ah = np.float32(ah); al = np.float32(al); b = np.float32(b)
    with np.errstate(invalid='ignore', over='ignore', divide='ignore'):
        t1 = np.float32(ah / b)
        t2, t3 = two_prod(b, t1)
        t4 = np.float32(ah - t2)   # Sterbenz: exact
        t5 = np.float32(al - t3)
        t6 = np.float32(t4 + t5)
        t7 = np.float32(t6 / b)
    return fast_two_sum(t1, t7)


def ds_sqrt(ah, al):
    """Ported exactly from double-single-lib's double_binary32_sqrt.

    Edge semantics not special-cased, matching the library: ah < 0 gives
    t1 = sqrt(ah) = NaN, propagating to (NaN, NaN). ah == 0 gives t1 = 0,
    which then feeds ds_div_by_scalar as b = 0, whose first division 0/0
    is NaN immediately -- so ds_sqrt(0, 0) is also (NaN, NaN), not a
    clean zero.
    """
    ah = np.float32(ah); al = np.float32(al)
    with np.errstate(invalid='ignore', over='ignore', divide='ignore'):
        t1 = np.float32(np.sqrt(ah))
        t2, t3 = ds_div_by_scalar(ah, al, t1)
        t4, t5 = two_sum(t1, t2)
        t6 = np.float32(t5 + t3)
        t7 = np.float32(0.5 * t4)
        t8 = np.float32(0.5 * t6)
    return fast_two_sum(t7, t8)


# ── exp / log (ported from double-single-libm) ─────────────────────────────────
#
# Mirrors DsTransformPass.cpp's emitDsExp/emitDsLog, which were ported from
# double-single-libm's expds (exp.c) and logds (log.c). The lookup tables are
# parsed from the same headers the pass compiles in
# (stablehlo_pass/libmds/*.h), so there is one copy of the table data.
#
# As with ds_div, the library's FMA-based __libmds_mul_2_11 is substituted
# with two_prod (Veltkamp-split-based); both yield the exact product error,
# so the results are the same wherever neither over/underflows.

import re
from pathlib import Path

_LIBMDS_DIR = Path(__file__).resolve().parent.parent / "stablehlo_pass" / "libmds"
_libmds_tables = {}


def _libmds_table(header, name):
    """Load one `static const __libmds_binary32_t name[N] = {...};` array."""
    if name not in _libmds_tables:
        text = (_LIBMDS_DIR / header).read_text()
        m = re.search(re.escape(name) + r"\[(\d+)\]\s*=\s*\{(.*?)\};", text, re.S)
        vals = [float.fromhex(v) for v in
                re.findall(r"\)\s*(-?0x[0-9a-fA-F.]+p[-+]?\d+)f", m.group(2))]
        assert len(vals) == int(m.group(1)), (name, len(vals))
        _libmds_tables[name] = np.array(vals, dtype=np.float32)
    return _libmds_tables[name]


def _hexf(s):
    return np.float32(float.fromhex(s))


def _i32(v):
    """Wrap a Python int to int32 two's-complement range."""
    v &= 0xffffffff
    return v - (1 << 32) if v & 0x80000000 else v


def _f32_bits(x):
    return int(np.float32(x).view(np.int32))


def _two_pow(k):
    """__libmds_two_pow_normal: 2^k for k in the normal exponent range."""
    return np.int32(_i32((127 + k) << 23)).view(np.float32)


def _add_2_12(a, bh, bl):
    t1h, t1l = two_sum(a, bh)
    return fast_two_sum(t1h, np.float32(t1l + bl))


def _add_2_22(ah, al, bh, bl):
    t1h, t1l = two_sum(ah, bh)
    t2 = np.float32(al + bl)
    return fast_two_sum(t1h, np.float32(t1l + t2))


def _mul_2_12(a, bh, bl):
    t1h, t1l = two_prod(a, bh)
    t2 = np.float32(a * bl)
    return fast_two_sum(t1h, np.float32(t1l + t2))


def _mul_2_22(ah, al, bh, bl):
    t1h, t1l = two_prod(ah, bh)
    t4 = np.float32(np.float32(ah * bl) + np.float32(al * bh))
    return two_sum(t1h, np.float32(t4 + t1l))


_EXP_OMEGA = _hexf("0x1.fffffep127")
_EXP_OVERFLOW_HI = _hexf("0x1.62e43p6")
_EXP_OVERFLOW_LO = _hexf("-0x1.25c612p-22")
_EXP_UNDERFLOW_HI = _hexf("-0x1.9fe36ap6")
_EXP_UNDERFLOW_LO = _hexf("0x1.d32c42p-18")
_EXP_SCALED_RCPR_LOG_TWO = _hexf("0x1.715476p8")
_EXP_SHIFTER = _hexf("0x1.8p23")
_EXP_M_LOG_TWO_CHUNKS = [_hexf(s) for s in (
    "-0x1.62p-9", "-0x1.c8p-18", "-0x1.8p-28",
    "0x1.06p-37", "-0x1.dp-48", "0x1.0cp-57")]
_EXP_POLY = [_hexf(s) for s in (
    "0x1.0p0", "0x1.0p0", "0x1p-1", "0x1.555558p-3", "0x1.55555cp-5")]


def ds_exp(xh, xl):
    """Ported from double-single-libm's expds (exp.c)."""
    xh = np.float32(xh); xl = np.float32(xl)
    with np.errstate(invalid='ignore', over='ignore', under='ignore'):
        s = np.float32(xh + xl)
        if s != s or s > _EXP_OMEGA:
            return s, s                      # NaN, +Inf
        if s < -_EXP_OMEGA:
            return np.float32(0.0), np.float32(0.0)
        if xh > _EXP_OVERFLOW_HI or (xh == _EXP_OVERFLOW_HI and xl > _EXP_OVERFLOW_LO):
            return np.float32(np.inf), np.float32(np.inf)
        if xh < _EXP_UNDERFLOW_HI or (xh == _EXP_UNDERFLOW_HI and xl < _EXP_UNDERFLOW_LO):
            return np.float32(0.0), np.float32(0.0)

        # k = nearestint(xh * 2^8/log(2)) via the shifter trick.
        ssxh = np.float32(np.float32(xh * _EXP_SCALED_RCPR_LOG_TWO) + _EXP_SHIFTER)
        k = np.float32(ssxh - _EXP_SHIFTER)
        kint = _i32(_f32_bits(ssxh) << 14) >> 14
        n = kint >> 8
        idx = kint - (n << 8)

        # rh + rl ~= xh - k * 2^-8 * log(2) + xl
        l2k = [np.float32(k * c) for c in _EXP_M_LOG_TWO_CHUNKS]
        rth = np.float32(xh + l2k[0])        # Sterbenz
        lows = []
        for c in l2k[1:]:
            rth, rtl = two_sum(rth, c)
            lows.append(rtl)
        rt7 = lows[4]
        for rtl in (lows[3], lows[2], lows[1], lows[0], xl):
            rt7 = np.float32(rt7 + rtl)
        rh, rl = two_sum(rth, rt7)

        # ph + pl ~= e^rh
        q = np.float32(rh * np.float32(_EXP_POLY[3] + np.float32(rh * _EXP_POLY[4])))
        pt1h, pt1l = fast_two_sum(_EXP_POLY[2], q)
        pt2h, pt2l = _add_2_12(_EXP_POLY[1], *_mul_2_12(rh, pt1h, pt1l))
        ph, pl = _add_2_12(_EXP_POLY[0], *_mul_2_12(rh, pt2h, pt2l))

        th = _libmds_table("exp_table.h", "__expds_table_hi")[idx]
        tl = _libmds_table("exp_table.h", "__expds_table_lo")[idx]

        # (th + tl) * (ph + pl) * (1 + rl), then scale by 2^n in two steps.
        qh, ql = _mul_2_12(rl, ph, pl)
        zh, zl = _add_2_22(ph, pl, qh, ql)
        wh, wl = _mul_2_22(th, tl, zh, zl)
        n1 = n >> 1
        s1 = _two_pow(n1); s2 = _two_pow(n - n1)
        return np.float32(s1 * np.float32(s2 * wh)), np.float32(s1 * np.float32(s2 * wl))


_LOG_OMEGA = _hexf("0x1.fffffep127")
_LOG_SQRT_TWO = _hexf("0x1.6a09e8p0")
_LOG_MOD_SHIFTER = _hexf("0x1.7ffe96p23")
_LOG_TWO = [_hexf(s) for s in ("0x1.62e4p-1", "0x1.7f7cp-20", "0x1.1cf8p-36")]
_LOG_C1 = (_hexf("0x1p0"), _hexf("-0x1.8p-47"))
_LOG_C2 = (_hexf("-0x1p-1"), _hexf("0x1.95ep-39"))
_LOG_C3 = _hexf("0x1.555556p-2")
_LOG_C4 = _hexf("-0x1.000068p-2")
_LOG_C5 = _hexf("0x1.970e1cp-3")


def _logb(x):
    """__libmds_logb_finite_non_zero: floor(log2(x)), subnormals included."""
    E = ((_f32_bits(x) >> 23) & 0xff) - 127
    G = -E
    G1 = G >> 1
    ssx = np.float32(_two_pow(G - G1) * np.float32(_two_pow(G1) * x))
    return E + ((_f32_bits(ssx) >> 23) & 0xff) - 127


def ds_log(xh, xl):
    """Ported from double-single-libm's logds (log.c), WITH ONE DELIBERATE
    CORRECTION: for xh == 0 the library's comment says "Return -inf" but
    its code computes 1.0f / (0.0f * 0.0f) = +inf. This returns (-inf, 0),
    the mathematically correct log(0), matching emitDsLog.
    """
    xh = np.float32(xh); xl = np.float32(xl)
    with np.errstate(invalid='ignore', over='ignore', under='ignore'):
        s = np.float32(xh + xl)
        if s != s or s > _LOG_OMEGA:
            return s, s                      # NaN, +Inf
        if s < -_LOG_OMEGA or xh < 0.0:
            return np.float32(np.nan), np.float32(np.nan)
        if xh == 0.0:
            return np.float32(-np.inf), np.float32(0.0)

        # Scale so that log(xh + xl) = E*log(2) + log(zh + zl),
        # sqrt(2)/2 < zh < sqrt(2).
        E = _logb(xh)
        F = -E
        F1 = F >> 1
        s1 = _two_pow(F1); s2 = _two_pow(F - F1)
        zh = np.float32(s1 * np.float32(s2 * xh))
        zl = np.float32(s1 * np.float32(s2 * xl))
        if zh >= _LOG_SQRT_TWO:
            zh = np.float32(zh * np.float32(0.5))
            zl = np.float32(zl * np.float32(0.5))
            E += 1

        # i = nearestint(2^8 * zh) - 181, read from the shifted sum's low byte.
        sh = np.float32(np.float32(np.float32(256.0) * zh) + _LOG_MOD_SHIFTER)
        i = _f32_bits(sh) & 0xff

        # rh + rl = w * (zh + zl) - 1 with w ~= 1/zh
        w = _libmds_table("log_table.h", "__logds_table_rcpr_z")[i]
        rh, rl = _add_2_12(np.float32(-1.0), *_mul_2_12(w, zh, zl))
        th = _libmds_table("log_table.h", "__logds_table_m_log_w_hi")[i]
        tl = _libmds_table("log_table.h", "__logds_table_m_log_w_lo")[i]

        # elh + ell = E * log(2)
        e = np.float32(E)
        telh, tell = fast_two_sum(np.float32(e * _LOG_TWO[0]), np.float32(e * _LOG_TWO[1]))
        elh, ell = fast_two_sum(telh, np.float32(tell + np.float32(e * _LOG_TWO[2])))

        # ph + pl ~= log(1 + rh + rl), degree-5 Horner in double-single.
        q4h, q4l = _add_2_12(_LOG_C4, *_mul_2_12(_LOG_C5, rh, rl))
        q3h, q3l = _add_2_12(_LOG_C3, *_mul_2_22(rh, rl, q4h, q4l))
        q2h, q2l = _add_2_22(*_LOG_C2, *_mul_2_22(rh, rl, q3h, q3l))
        q1h, q1l = _add_2_22(*_LOG_C1, *_mul_2_22(rh, rl, q2h, q2l))
        ph, pl = _mul_2_22(rh, rl, q1h, q1l)

        gh, gl = _add_2_22(th, tl, ph, pl)
        return _add_2_22(elh, ell, gh, gl)


# ── Trivial / comparison ops (ported from double-single-lib) ───────────────────
#
# Mirrors DsTransformPass.cpp's emitDsAbs/emitDsCompare, which were ported
# from double_single_ray/llvm-accuracy-analysis-k-test/double-single-lib's
# double_binary32_neg/fabs/compare -- see that library for the authoritative
# reference.

def ds_negate(h, l):
    return np.float32(-np.float32(h)), np.float32(-np.float32(l))


def ds_abs(h, l):
    """hi >= -lo (not sign(hi) alone) -- handles hi==0.0, lo<0 correctly."""
    h = np.float32(h); l = np.float32(l)
    if h >= -l:
        return h, l
    return np.float32(-h), np.float32(-l)


def ds_compare(ah, al, bh, bl):
    """Hi-first, lo-tiebreak lexicographic compare.

    Returns -1 (a<b), 0 (a==b), 1 (a>b), or None (unordered -- NaN present).
    """
    ah = np.float32(ah); al = np.float32(al)
    bh = np.float32(bh); bl = np.float32(bl)
    if not (ah == ah and al == al and bh == bh and bl == bl):
        return None
    if ah > bh or (ah == bh and al > bl):
        return 1
    if ah < bh or (ah == bh and al < bl):
        return -1
    return 0


def ds_max(ah, al, bh, bl):
    order = ds_compare(ah, al, bh, bl)
    if order is None:
        return float('nan'), float('nan')
    return (ah, al) if order >= 0 else (bh, bl)


def ds_min(ah, al, bh, bl):
    order = ds_compare(ah, al, bh, bl)
    if order is None:
        return float('nan'), float('nan')
    return (ah, al) if order <= 0 else (bh, bl)


# ── Higher-level operations ───────────────────────────────────────────────────

def ds_dot(a, b):
    """
    DS dot product via scalar two_prod + DS accumulation.

    Equivalent to what the compiler pass produces for jnp.dot(a * a, b) except
    here it directly computes the true dot product of a and b in DS arithmetic.
    Matches the reduce handler in DsTransformPass.cpp.
    """
    a = np.asarray(a, np.float32).ravel()
    b = np.asarray(b, np.float32).ravel()
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")
    acc_h = np.float32(0.0)
    acc_l = np.float32(0.0)
    for i in range(len(a)):
        ph, pl = two_prod(a[i], b[i])
        acc_h, acc_l = ds_add(acc_h, acc_l, ph, pl)
    return float(acc_h) + float(acc_l)


def ds_sum(a):
    """DS sum of a float32 array (matches reduce with add body)."""
    a = np.asarray(a, np.float32).ravel()
    acc_h = np.float32(0.0)
    acc_l = np.float32(0.0)
    for x in a:
        acc_h, acc_l = ds_add(acc_h, acc_l, x, np.float32(0.0))
    return float(acc_h) + float(acc_l)


def ds_matmul(A, B):
    """
    DS matrix multiply via scalar DS dot products.

    Correct but O(M*K*N) scalar loops — only suitable for small matrices in
    tests. For large matrices use the GPU path with the PJRT plugin.
    """
    A = np.asarray(A, np.float32)
    B = np.asarray(B, np.float32)
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("inputs must be 2-D")
    M, K = A.shape
    K2, N = B.shape
    if K != K2:
        raise ValueError(f"shape mismatch: ({M},{K}) @ ({K2},{N})")
    C = np.empty((M, N), np.float64)
    for i in range(M):
        for j in range(N):
            C[i, j] = ds_dot(A[i], B[:, j])
    return C


# ── Convenience helpers used by test scripts ──────────────────────────────────

def f64_dot(a, b):
    """Ground-truth dot product: cast f32 inputs to f64, dot in f64."""
    return float(np.dot(
        np.asarray(a, np.float32).astype(np.float64),
        np.asarray(b, np.float32).astype(np.float64),
    ))


def f32_dot(a, b):
    return float(np.dot(np.asarray(a, np.float32), np.asarray(b, np.float32)))


def f64_matmul(A, B):
    A = np.asarray(A, np.float32).astype(np.float64)
    B = np.asarray(B, np.float32).astype(np.float64)
    return A @ B


def f32_matmul(A, B):
    return np.asarray(A, np.float32) @ np.asarray(B, np.float32)
