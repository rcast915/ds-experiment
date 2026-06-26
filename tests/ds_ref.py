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
