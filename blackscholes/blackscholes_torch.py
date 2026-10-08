"""Vectorized PyTorch port of the PARSEC Black-Scholes kernel.

Source: bamos/parsec-benchmark, pkgs/apps/blackscholes/src/blackscholes.c
        (functions CNDF and BlkSchlsEqEuroNoDiv).

Every assignment below mirrors one assignment in the C, in the same order,
with the same constants. Two changes are forced by vectorization:

  * CNDF's `if (InputX < 0.0)` sign branch and the final `if (sign)` are
    replaced by torch.where (both arms are computed, one is selected).
  * BlkSchlsEqEuroNoDiv's `if (otype == 0)` call/put branch is likewise
    replaced by torch.where.

The C's unused `float timet` argument is dropped.

Precision follows the input dtype. PARSEC builds with `#define fptype float`,
but its literals (0.2316419, 1.0, inv_sqrt_2xPI, ...) are doubles, so each C
statement is evaluated in double and rounded to float on assignment. Here
constants take the tensor dtype, so float32 results can differ from the C by a
few ULPs; float64 results match a `fptype double` build.

Constants not exactly representable in float32 are materialized with
torch.full_like (see _c) so that the StableHLO export keeps them at full
float64 precision.
"""

import torch

inv_sqrt_2xPI = 0.39894228040143270286


def _c(like: torch.Tensor, value: float) -> torch.Tensor:
    """Constant `value` with the shape and dtype of `like`.

    torch_xla's StableHLO lowering rounds Python-float operands to float32 even in
    a float64 graph (0.2316419 -> 0.23164190351963043); full_like keeps the exact
    value. It likewise promotes a float32 tensor to float64 for `x < 0.0`, so the
    sign test uses _c too. Other float32-exact literals (0.5, 1.0) stay as floats.
    """
    return torch.full_like(like, value)


def cndf(input_x: torch.Tensor) -> torch.Tensor:
    """Cumulative normal distribution function, elementwise (C: CNDF)."""
    # C: if (InputX < 0.0) { InputX = -InputX; sign = 1; } else sign = 0;
    sign = input_x < _c(input_x, 0.0)
    input_x = torch.where(sign, -input_x, input_x)

    x_input = input_x

    # --- TRANSCENDENTAL: exp ------------------------------------------------
    # C: expValues = exp(-0.5f * InputX * InputX);
    # Gaussian kernel e^{-x^2/2} of the standard normal pdf N'(x).
    exp_values = torch.exp(-0.5 * input_x * input_x)
    x_nprime_of_x = exp_values
    x_nprime_of_x = x_nprime_of_x * _c(x_nprime_of_x, inv_sqrt_2xPI)

    x_k2 = _c(x_input, 0.2316419) * x_input
    x_k2 = 1.0 + x_k2
    # --- RECIPROCAL -----------------------------------------------------------
    # C: xK2 = 1.0 / xK2;   k = 1 / (1 + 0.2316419 x), Abramowitz-Stegun 26.2.17
    x_k2 = torch.reciprocal(x_k2)
    x_k2_2 = x_k2 * x_k2
    x_k2_3 = x_k2_2 * x_k2
    x_k2_4 = x_k2_3 * x_k2
    x_k2_5 = x_k2_4 * x_k2

    x_local_1 = x_k2 * _c(x_k2, 0.319381530)
    x_local_2 = x_k2_2 * _c(x_k2_2, -0.356563782)
    x_local_3 = x_k2_3 * _c(x_k2_3, 1.781477937)
    x_local_2 = x_local_2 + x_local_3
    x_local_3 = x_k2_4 * _c(x_k2_4, -1.821255978)
    x_local_2 = x_local_2 + x_local_3
    x_local_3 = x_k2_5 * _c(x_k2_5, 1.330274429)
    x_local_2 = x_local_2 + x_local_3

    x_local_1 = x_local_2 + x_local_1
    x_local = x_local_1 * x_nprime_of_x
    x_local = 1.0 - x_local

    output_x = x_local

    # C: if (sign) { OutputX = 1.0 - OutputX; }
    output_x = torch.where(sign, 1.0 - output_x, output_x)

    return output_x


def blk_schls_eq_euro_no_div(
    sptprice: torch.Tensor,
    strike: torch.Tensor,
    rate: torch.Tensor,
    volatility: torch.Tensor,
    otime: torch.Tensor,
    otype: torch.Tensor,
) -> torch.Tensor:
    """European option price, no dividends, elementwise (C: BlkSchlsEqEuroNoDiv).

    otype: integer tensor, 0 = call, nonzero = put (PARSEC maps 'P' -> 1).
    """
    x_stock_price = sptprice
    x_strike_price = strike
    x_risk_free_rate = rate
    x_volatility = volatility

    x_time = otime
    # --- TRANSCENDENTAL: sqrt -------------------------------------------------
    # C: xSqrtTime = sqrt(xTime);   sqrt(T) for the sigma*sqrt(T) denominator.
    x_sqrt_time = torch.sqrt(x_time)

    # --- TRANSCENDENTAL: log (with its division) ----------------------------
    # C: logValues = log( sptprice / strike );   ln(S/K) term of d1/d2.
    # Kept as divide-then-log, NOT rewritten as log(S) - log(K).
    log_values = torch.log(sptprice / strike)

    x_log_term = log_values

    x_power_term = x_volatility * x_volatility
    x_power_term = x_power_term * 0.5

    x_d1 = x_risk_free_rate + x_power_term
    x_d1 = x_d1 * x_time
    x_d1 = x_d1 + x_log_term

    x_den = x_volatility * x_sqrt_time
    # --- DIVISION ---------------------------------------------------------------
    # C: xD1 = xD1 / xDen;   d1 = (ln(S/K) + (r + sigma^2/2) T) / (sigma sqrt(T))
    x_d1 = x_d1 / x_den
    x_d2 = x_d1 - x_den

    d1 = x_d1
    d2 = x_d2

    nofx_d1 = cndf(d1)
    nofx_d2 = cndf(d2)

    # --- TRANSCENDENTAL: exp ------------------------------------------------
    # C: FutureValueX = strike * ( exp( -(rate)*(time) ) );
    # Discount factor e^{-rT}; negation applied to rate before the multiply.
    future_value_x = strike * torch.exp(-(rate) * (otime))

    # C: if (otype == 0) { call } else { put }
    call_price = (sptprice * nofx_d1) - (future_value_x * nofx_d2)
    neg_nofx_d1 = 1.0 - nofx_d1
    neg_nofx_d2 = 1.0 - nofx_d2
    put_price = (future_value_x * neg_nofx_d2) - (sptprice * neg_nofx_d1)
    option_price = torch.where(otype == 0, call_price, put_price)

    return option_price


class BlackScholes(torch.nn.Module):
    """Batched PARSEC Black-Scholes; all inputs are 1-D tensors of equal length."""

    def forward(self, sptprice, strike, rate, volatility, otime, otype):
        return blk_schls_eq_euro_no_div(sptprice, strike, rate, volatility, otime, otype)
