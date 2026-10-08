"""Plain-Python scalar translation of PARSEC's CNDF and BlkSchlsEqEuroNoDiv.

A line-by-line transliteration of blackscholes.c for validating the vectorized
port. `F` is applied at every C assignment to emulate the build's fptype:

  * F = as_double: `#define fptype double` (Python floats are IEEE doubles).
  * F = as_float:  `#define fptype float` (PARSEC's default). Double literals
    promote a C expression to double; the result is rounded to float when
    assigned. Expressions that are float-typed in C (float op float) are wrapped
    in F as well, and inputs are passed through F before the call.

libm's exp/log/sqrt take and return double in C, and so does math.* here.
"""

import math
import struct

inv_sqrt_2xPI = 0.39894228040143270286


def as_double(v):
    return v


def as_float(v):
    return struct.unpack("f", struct.pack("f", v))[0]


def CNDF(InputX, F=as_double):
    # Check for negative value of InputX
    if InputX < 0.0:
        InputX = -InputX
        sign = 1
    else:
        sign = 0

    xInput = InputX

    # Compute NPrimeX term common to both four & six decimal accuracy calcs
    expValues = F(math.exp(F(F(-0.5 * InputX) * InputX)))  # -0.5f: float-typed
    xNPrimeofX = expValues
    xNPrimeofX = F(xNPrimeofX * inv_sqrt_2xPI)

    xK2 = F(0.2316419 * xInput)
    xK2 = F(1.0 + xK2)
    xK2 = F(1.0 / xK2)
    xK2_2 = F(xK2 * xK2)
    xK2_3 = F(xK2_2 * xK2)
    xK2_4 = F(xK2_3 * xK2)
    xK2_5 = F(xK2_4 * xK2)

    xLocal_1 = F(xK2 * 0.319381530)
    xLocal_2 = F(xK2_2 * (-0.356563782))
    xLocal_3 = F(xK2_3 * 1.781477937)
    xLocal_2 = F(xLocal_2 + xLocal_3)
    xLocal_3 = F(xK2_4 * (-1.821255978))
    xLocal_2 = F(xLocal_2 + xLocal_3)
    xLocal_3 = F(xK2_5 * 1.330274429)
    xLocal_2 = F(xLocal_2 + xLocal_3)

    xLocal_1 = F(xLocal_2 + xLocal_1)
    xLocal = F(xLocal_1 * xNPrimeofX)
    xLocal = F(1.0 - xLocal)

    OutputX = xLocal

    if sign:
        OutputX = F(1.0 - OutputX)

    return OutputX


def BlkSchlsEqEuroNoDiv(sptprice, strike, rate, volatility, time, otype, F=as_double):
    xStockPrice = sptprice
    xStrikePrice = strike
    xRiskFreeRate = rate
    xVolatility = volatility

    xTime = time
    xSqrtTime = F(math.sqrt(xTime))

    logValues = F(math.log(F(sptprice / strike)))

    xLogTerm = logValues

    xPowerTerm = F(xVolatility * xVolatility)
    xPowerTerm = F(xPowerTerm * 0.5)

    xD1 = F(xRiskFreeRate + xPowerTerm)
    xD1 = F(xD1 * xTime)
    xD1 = F(xD1 + xLogTerm)

    xDen = F(xVolatility * xSqrtTime)
    xD1 = F(xD1 / xDen)
    xD2 = F(xD1 - xDen)

    d1 = xD1
    d2 = xD2

    NofXd1 = CNDF(d1, F)
    NofXd2 = CNDF(d2, F)

    FutureValueX = F(strike * (math.exp(F(-(rate) * (time)))))
    if otype == 0:
        OptionPrice = F(F(sptprice * NofXd1) - F(FutureValueX * NofXd2))
    else:
        NegNofXd1 = F(1.0 - NofXd1)
        NegNofXd2 = F(1.0 - NofXd2)
        OptionPrice = F(F(FutureValueX * NegNofXd2) - F(sptprice * NegNofXd1))

    return OptionPrice
