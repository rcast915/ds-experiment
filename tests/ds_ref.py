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
    """Ported exactly from double-single-lib's double_binary32_div.

    __two_mul (the library's FMA-based TwoProduct) is substituted with
    two_prod (Veltkamp-split-based) here too, matching the same
    substitution DsTransformPass.cpp's emitDsDiv makes -- see its comment
    for why a real hardware FMA can't be faithfully reproduced with
    separate ops.

    KNOWN LIBRARY LIMITATION, kept as-is: t3 (the TwoProd/two_mul
    residual of bh*t1) is computed and then never used -- t6 subtracts
    t4 (bl*t1), not t3, matching the library's literal double_binary32_div
    exactly. This is NOT limited to bl != 0 inputs: t3 captures t1's own
    single-precision rounding error, so omitting it costs up to ~2^-23
    relative error (f32-ULP level) on EVERY division through this
    routine, including plain lo=0/lo=0 cases. See DsTransformPass.cpp's
    emitDsDiv comment for the full derivation, confirmed against the
    library's own sibling routine (used by sqrt) which correctly keeps
    the equivalent term.
    """
    ah = np.float32(ah); al = np.float32(al)
    bh = np.float32(bh); bl = np.float32(bl)
    with np.errstate(invalid='ignore', over='ignore', divide='ignore'):
        t1 = np.float32(ah / bh)
        t2, t3 = two_prod(bh, t1)   # t3 intentionally unused -- see docstring
        t4 = np.float32(bl * t1)
        t5 = np.float32(ah - t2)   # Sterbenz: exact
        t6 = np.float32(al - t4)
        t7 = np.float32(t5 + t6)
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
