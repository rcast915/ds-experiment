
#include "double-binary32.h"
#include <stdint.h>

binary32_t sqrtf(binary32_t);

static inline void __fast_two_sum(binary32_t * RESTRICT hi, binary32_t * RESTRICT lo,
				  binary32_t a, binary32_t b) {
  binary32_t h, l, t;

  h = a + b;
  t = h - a;
  l = b - t;
  
  *hi = h;
  *lo = l;
}

static inline void __two_sum(binary32_t * RESTRICT hi, binary32_t * RESTRICT lo,
			     binary32_t a, binary32_t b) {
  binary32_t s, t, ap, bp, da, db;

  s = a + b;
  ap = s - b;
  bp = s - ap;
  da = a - ap;
  db = b - bp;
  t = da + db;
  
  *hi = s;
  *lo = t;
}

static inline void __two_mul(binary32_t * RESTRICT hi, binary32_t * RESTRICT lo,
			     binary32_t a, binary32_t b) {
  binary32_t h, nh, l;

  h = a * b;
  nh = -h;
  l = __builtin_fmaf(a, b, nh);
  
  *hi = h;
  *lo = l;
}

static inline void __double_binary_div_double_by_single(binary32_t * RESTRICT hi, binary32_t * RESTRICT lo,
							binary32_t ah, binary32_t al, binary32_t b) {
  binary32_t t1, t2, t3, t4, t5, t6, t7, h, l;

  t1 = ah / b;
  __two_mul(&t2, &t3, b, t1);
  t4 = ah - t2; /* Sterbenz */
  t5 = al - t3;
  t6 = t4 + t5;
  t7 = t6 / b;
  __fast_two_sum(&h, &l, t1, t7);
  *hi = h;
  *lo = l;
}

static inline void __double_binary32_from_unsigned_long_long_int(double_binary32_t * RESTRICT rop, unsigned long long int op) {
  uint64_t op64, hiunorm64;
  int64_t res;
  binary32_t hiunorm, lounorm, hi, lo;
  
  op64 = (uint64_t) op;
  
  hiunorm = (binary32_t) op;
  if (hiunorm > 18446742974197923840.0f) { /* 18446742974197923840.0f = round(2^64 - 1, SG, RD) */
    hiunorm = 18446742974197923840.0f;
  }
  hiunorm64 = (uint64_t) hiunorm; /* Cannot overflow due to test above */
  if (hiunorm64 < op64) {
    res = (int64_t) (op64 - hiunorm64);
  } else {
    res = -((int64_t) (hiunorm64 - op64)); /* Cannot overflow */
  }
  lounorm = (binary32_t) res;
  
  __fast_two_sum(&hi, &lo, hiunorm, lounorm);
  
  rop->hi = hi;
  rop->lo = lo;
}

static inline void __double_binary32_from_signed_long_long_int(double_binary32_t * RESTRICT rop, signed long long int op) {
  if (op >= ((signed long long int) 0)) {
    __double_binary32_from_unsigned_long_long_int(rop, ((unsigned long long int) op));
    return;
  }

  __double_binary32_from_unsigned_long_long_int(rop,
						(((unsigned long long int) (-(op + ((signed long long int) 1)))) +
						 ((unsigned long long int) 1)));
  rop->hi = -rop->hi;
  rop->lo = -rop->lo;
}

static inline void __double_binary32_to_signed_integer(int64_t * RESTRICT rop,
						       CONST double_binary32_t * RESTRICT op,
						       int64_t m, int64_t M);

static inline void __double_binary32_to_unsigned_integer(uint64_t * RESTRICT rop,
							 CONST double_binary32_t * RESTRICT op,
							 uint64_t m, uint64_t M) {
  uint64_t res, tm, tM, tres, ttt, tt, t;
  binary32_t hi, lo, thi, tlo, treconv;
  volatile binary32_t vtfp;
  volatile uint64_t vt;
  double_binary32_t top;
  int64_t tress, ms, Ms;
  
  __fast_two_sum(&hi, &lo, op->hi, op->lo);

  if (!((hi == hi) && (lo == lo))) {
    /* NaN */
    res = (uint64_t) hi;
  } else {
    if (hi > 340282346638528859811704183484516925440.0f) { /* Maximum finite binary32_t */
      /* +Inf */
      res = M;
    } else {
      if (hi < -340282346638528859811704183484516925440.0f) { /* Minimum finite binary32_t */
	/* -Inf */
	res = m;
      } else {
	/* Not NaN nor +/- Inf */
	if (hi < 0.0f) {
	  /* High part is negative */
	  if (hi < -1.0f) {
	    /* No chance for a unsigned integer */
	    res = m;
	  } else {
	    if (hi == -1.0f) {
	      /* hi = -1.0f. 

		 Therefore -1.0f - 1ulp < hi + lo < -1.0f + 1ulp 

		 If lo <= 0.0, integer(hi + lo) = -1, which we cannot
		 return.

		 If lo > 0.0, integer(hi + lo) is either -1, which we
		 cannot return or 0, which we can return.

	      */
	      if (lo <= 0.0f) {
		res = m;
	      } else {
		vtfp = -0.5f;
		vt = (uint64_t) vtfp;
		if (vt == ((uint64_t) 0)) {
		  if (m > ((uint64_t) 0)) {
		    res = m;
		  } else {
		    res = (uint64_t) 0;
		  }
		} else {
		  res = m;
		}
	      }
	    } else {
	      /* Here 1.0 <= hi + lo < 0.0 

		 Return the same answer as -0.5.

	      */
	      vtfp = -0.5f;
	      vt = (uint64_t) vtfp;
	      if (vt < m) {
		res = m;
	      } else {
		res = vt;
	      }
	    }
	  }
	} else {
	  /* hi + lo >= 0.0 and hi is finite (not Inf nor NaN) 

	     Check if hi <= 18446742974197923840.0f

	     If this condition is satisfied, we know that 

	     integer(hi) <= 2^64 - 1 

	     and

	     abs(lo) <= 549755813888.0f
	     
	     This means that

	     hi + lo <= 18446742974197923840.0 + 549755813888.0 
                      = 18446743523953737728.

	     This means that 

	     integer(hi + lo) <= integer(18446743523953737728) 
	                      <= 2^64 - 1.
	     
	     This means that both hi and hi + lo can be converted
	     to a 64bit unsigned integer without overflow.

	     If hi > 18446742974197923840.0f, we have

	     hi >= 18446744073709551616.0f.

	     There are 2 subcases.

	     If hi = 18446744073709551616.0, we have to 
	     be careful, because integer(hi) = 2^64 > 2^64 - 1.

	     We have two subcases:

	     alpha)  lo >= 0.0. The result overflows.

	     beta)   lo <= 0.0. The result is either 2^64 - 1, which 
	                        we can return or 2^64, which we cannot
				return.
	     
	     If hi > 18446744073709551616.0, we know that 

	     hi + lo > 18446744073709551616.0 and the result overflows.

	  */
	  if (hi > 18446742974197923840.0f) {
	    if (hi == 18446744073709551616.0f) {
	      if (lo < 0.0f) {
		/* In this case 
		   
		   18446743523953737728 < hi + lo < 18446744073709551616 

		   This means

		   0 < (hi - 18446743523953737728) + lo < 549755813888.

		   This also means that 

		   integer(hi + lo) = 18446743523953737728 + integer(thi + tlo)

		   with

		   thi = 18446744073709551616 - 18446743523953737728 = 549755813888 

		   and

		   tlo = lo.

		   We load thi and tlo and call ourselves recursively.

		   We will have

		   integer(hi + lo) = 18446743523953737728 +
		   integer(thi + tlo), which may or may not overflow
		   on 64bit unsigned integers.  If if does overflow,
		   we return M. Otherwise we know the answer.
		   
		*/
		thi = 549755813888.0f;
		tlo = lo;
	       
		top.hi = thi;
		top.lo = tlo;

		tm = (uint64_t) 0;
		tM = tm;
		tM--;
		
		__double_binary32_to_unsigned_integer(&tres, &top, tm, tM);

		tt = (uint64_t) 18446743523953737728ull;
		
		/* Here, our answer is 

		   tt + tres

		   which may or may not overflow.

		*/
		ttt = tt + tres;

		if (ttt < tt) {
		  /* Overflow */
		  res = M;
		} else {
		  /* No overflow */
		  if (ttt < M) {
		    if (ttt < m) {
		      res = m;
		    } else {
		      res = ttt;
		    }
		  } else {
		    res = M;
		  }
		}
	      } else {
		/* Clear overflow */
		res = M;
	      }
	    } else {
	      /* Clear overflow */
	      res = M;
	    }
	  } else {
	    /* Here we know that 

	       0 <= integer(hi + lo) <= 2^64 - 1

	       and

	       0 <= integer(hi) <= 2^64 - 1.

	       We transform this problem into

	       integer(hi + lo) = t + integer((hi - t) + lo) 

	       where

	       t = integer(hi)

	    */
	    t = (uint64_t) hi;

	    /* Now, if lo = 0.0, we know the answer:

	       integer(hi + lo) = integer(hi + 0.0)
                                = integer(hi)
                                = t

	       We know that 

	       0 <= hi <= 18446742974197923840.0f

	       and
	       
	       0 <= t <= 2^64 - 1

	       by all the checks we have done above.

	       So there is no overflow.

	    */
	    if (lo == 0.0f) {
	      if (t < m) {
		res = m;
	      } else {
		if (t > M) {
		  res = M;
		} else {
		  res = t;
		}
	      }
	    } else {
	      /* Here we know that 

		 0 <= hi <= 18446742974197923840.0f,
      
		 0 <= t <= 2^64 - 1
	       
		 and

		 lo != 0.0
	       
		 by all the checks we have done above.

		 If hi >= 2^23, hi is an integer.

		 This means integer(hi) = hi = t.

		 This means we can convert t back to 
		 binary32 exactly.

		 This means that the subtraction

		 hi - t

		 will exactly compute 0.0.

		 If hi < 2^23, we can reason as follows:

		 If t = 0, we can of course convert t 
		 back to binary32 exactly. 
		 
		 The subtraction

		 hi - t 

		 will be exact, too.

		 If t = 1, we can convert t back to binary32.
		 In this case we have 

		 0 < hi < 2.

		 In the subcase when hi >= 1.0, we have

		 1 <= hi < 2 and t = 1. 

		 Therefore 

		 1/2 < 1 < hi/t < 2

		 Thus Sterbenz' lemma applies and the subtraction

		 hi - t 

		 is exact.

		 In the subcase when 0 < hi < 1, however, we need to 
		 be careful. In this case we know that 

		 0 < hi + lo < 1

		 while still 

                 0 < hi      < 1

		 Therefore 

		 integer(hi + lo) = integer(hi) = t. 

		 So, finally for the general case, assume that t >= 2,
		 meaning that hi >= 1.0 or, more specifically,

		 1 <= hi <= 2^23 - 1/2

		 This means that 

		 2 <= t <= 2^23

		 while t is an integer.

		 This means that t is representable
		 in binary32. We can convert t back to 
		 binary32 exactly.

		 Now observe that 

		 -1 < t - hi < 1

		 This means

		 -1/t < 1 - hi/t < 1/t

		 which in turn yields

		 -1/t - 1 < -hi/t < 1/t - 1

		 resp.

		 1 - 1/t < hi/t < 1 + 1/t

		 For 2 <= t <= 2^23, this give

		 1/2 = 1 - 1/2 < hi/t < 1 + 1/2 = 3/2 < 2.

		 This means Sterbenz' lemma applies and

		 hi - t 

		 will be exact.
	    
	      */
	      if ((t == ((uint64_t) 1)) && (hi < 1.0f)) {
		/* We know the answer, which is t. See above for
		   details. 
		*/
		if (t < m) {
		  res = m;
		} else {
		  if (t > M) {
		    res = M;
		  } else {
		    res = t;
		  }
		}
	      } else {
		/* Here, we know that 

		   integer(hi + lo) = t + integer((hi - t) + lo) 

		   t is representable in binary32 

		   and

		   hi - t is exact.

		*/
		treconv = (binary32_t) t;

		thi = hi - treconv;
		tlo = lo;

		/* Here, we have to return 

		   integer(hi + lo) = t + integer(thi + tlo)

		   In order to compute integer(thi + tlo), we call
		   ourselves recursively but on a *signed*
		   conversion. 

		   The recursion stops because thi is less than hi and
		   thi + tlo have less ones in their binary
		   respresentation than hi + lo.
		   
		*/
		top.hi = thi;
		top.lo = tlo;

		tt = (uint64_t) 0;
		tt--;
		tt >>= 1;
		Ms = tt;
		ms = -Ms;

		/* Recursive call */
		__double_binary32_to_signed_integer(&tress, &top, ms, Ms);
		
		/* Here we have

		   integer(hi + lo) = t + integer(thi + tlo)

		   with 

		   tress = integer(thi + tlo)

		   where tress is signed.
		  
		   We need to work according to the sign of tress.
 
		*/
		if (tress < ((int64_t) 0)) {
		  /* tress is negative */
		  tress = -tress; /* No overflow possible due to bound
				     above 
				  */
		  tres = (uint64_t) tress; /* No overflow possible */

		  /* We need to return 

		     t - tres

		  */
		  if (tres <= t) {
		    tt = t - tres;
		    if (tt < m) {
		      res = m;
		    } else {
		      if (tt > M) {
			res = M;
		      } else {
			res = tt;
		      }
		    }
		  } else {
		    /* This case should never happen */
		    res = m;
		  }
		} else {
		  /* tress is positive or zero */
		  tres = (uint64_t) tress; /* No overflow possible */

		  /* We need to return 

		     t + tres 

		  */
		  tt = t + tres;
		  if (tt < t) {
		    /* Integer overflow */
		    res = M;
		  } else {
		    /* Plain result */
		    if (tt < m) {
		      res = m;
		    } else {
		      if (tt > M) {
			res = M;
		      } else {
			res = tt;
		      }
		    }
		  }
		}
	      }
	    }
	  }
	}
      }
    }
  }
  
  if (res < m) {
    res = m;
  }
  if (res > M) {
    res = M;
  }
  *rop = res;
}

static inline void __double_binary32_to_signed_integer(int64_t * RESTRICT rop,
						       CONST double_binary32_t * RESTRICT op,
						       int64_t m, int64_t M) {
  uint64_t m64, M64, nM64, nmm64, MM64, temp;
  int64_t res, t, tt, ttt;
  double_binary32_t nop;

  m64 = (uint64_t) 0;
  M64 = m64;
  M64--;
  M64 >>= 1u;
  t = -((int64_t) M64);
  tt = t;
  tt--;
  if (tt < t) {
    t = tt;
  }
  t++;
  t = -t;
  nM64 = (uint64_t) t;
  nM64++;
  
  MM64 = (uint64_t) M;
  ttt = m;
  ttt++;
  ttt = -ttt;
  nmm64 = (uint64_t) ttt;
  nmm64++;

  if (nmm64 < nM64) {
    nM64 = nmm64;
  }
  if (MM64 < M64) {
    M64 = MM64;
  }

  if ((op->hi + op->lo) >= 0.0f) {
    __double_binary32_to_unsigned_integer(&temp, op, m64, M64);
    res = (int64_t) temp;
  } else {
    nop.hi = -op->hi;
    nop.lo = -op->lo;
    __double_binary32_to_unsigned_integer(&temp, &nop, m64, nM64);
    res = -((int64_t) temp);
  }
  
  if (res < m) {
    res = m;
  }
  if (res > M) {
    res = M;
  }
  *rop = res;
}

void double_binary32_from_binary32(double_binary32_t * RESTRICT rop, binary32_t op) {
  rop->hi = op;
  rop->lo = 0.0f;
}

void double_binary32_from_binary64(double_binary32_t * RESTRICT rop, binary64_t op) {
  binary32_t hi, lo;
  binary64_t hi64, res;

  hi = (binary32_t) op;
  hi64 = (binary64_t) hi;
  res = op - hi64;
  lo = (binary32_t) res;

  rop->hi = hi;
  rop->lo = lo;
}

void double_binary32_from_unsigned_char(double_binary32_t * RESTRICT rop, unsigned char op) {
  __double_binary32_from_unsigned_long_long_int(rop, ((unsigned long long int) op));
}

void double_binary32_from_signed_char(double_binary32_t * RESTRICT rop, signed char op) {
  __double_binary32_from_signed_long_long_int(rop, ((signed long long int) op));
}

void double_binary32_from_unsigned_short(double_binary32_t * RESTRICT rop, unsigned short op) {
  __double_binary32_from_unsigned_long_long_int(rop, ((unsigned long long int) op));
}

void double_binary32_from_signed_short(double_binary32_t * RESTRICT rop, signed short op) {
  __double_binary32_from_signed_long_long_int(rop, ((signed long long int) op));
}

void double_binary32_from_unsigned_int(double_binary32_t * RESTRICT rop, unsigned int op) {
  __double_binary32_from_unsigned_long_long_int(rop, ((unsigned long long int) op));
}

void double_binary32_from_signed_int(double_binary32_t * RESTRICT rop, signed int op) {
  __double_binary32_from_signed_long_long_int(rop, ((signed long long int) op));
}

void double_binary32_from_unsigned_long_int(double_binary32_t * RESTRICT rop, unsigned long int op) {
  __double_binary32_from_unsigned_long_long_int(rop, ((unsigned long long int) op));
}

void double_binary32_from_signed_long_int(double_binary32_t * RESTRICT rop, signed long int op) {
  __double_binary32_from_signed_long_long_int(rop, ((signed long long int) op));
}

void double_binary32_from_unsigned_long_long_int(double_binary32_t * RESTRICT rop, unsigned long long int op) {
  __double_binary32_from_unsigned_long_long_int(rop, op);
}

void double_binary32_from_signed_long_long_int(double_binary32_t * RESTRICT rop, signed long long int op) {
  __double_binary32_from_signed_long_long_int(rop, op);
}

void double_binary32_to_binary32(binary32_t * RESTRICT rop, CONST double_binary32_t * RESTRICT op) {
  *rop = op->hi + op->lo;
}

void double_binary32_to_binary64(binary64_t * RESTRICT rop, CONST double_binary32_t * RESTRICT op) {
  binary64_t h, l;

  h = (binary64_t) op->hi;
  l = (binary64_t) op->lo;
  *rop = h + l;
}

void double_binary32_to_unsigned_char(unsigned char * RESTRICT rop, CONST double_binary32_t * RESTRICT op) {
  unsigned char m, M;
  uint64_t t;

  m = (unsigned char) 0;
  M = m;
  M--;
  __double_binary32_to_unsigned_integer(&t, op, ((uint64_t) m), ((uint64_t) M));
  *rop = (unsigned char) t;
}

void double_binary32_to_signed_char(signed char * RESTRICT rop, CONST double_binary32_t * RESTRICT op) {
  unsigned char tt;
  signed char m, M, mt;
  int64_t t;

  tt = (unsigned char) 0;
  tt--;
  tt >>= 1u;
  M = (signed char) tt;
  m = -M;
  mt = m;
  mt--;
  if (mt < m) {
    m = mt;
  }
  __double_binary32_to_signed_integer(&t, op, ((int64_t) m), ((int64_t) M));
  *rop = (signed char) t;  
}

void double_binary32_to_unsigned_short(unsigned short * RESTRICT rop, CONST double_binary32_t * RESTRICT op) {
  unsigned short m, M;
  uint64_t t;

  m = (unsigned short) 0;
  M = m;
  M--;
  __double_binary32_to_unsigned_integer(&t, op, ((uint64_t) m), ((uint64_t) M));
  *rop = (unsigned short) t;
}

void double_binary32_to_signed_short(signed short * RESTRICT rop, CONST double_binary32_t * RESTRICT op) {
  unsigned short tt;
  signed short m, M, mt;
  int64_t t;

  tt = (unsigned short) 0;
  tt--;
  tt >>= 1u;
  M = (signed short) tt;
  m = -M;
  mt = m;
  mt--;
  if (mt < m) {
    m = mt;
  }
  __double_binary32_to_signed_integer(&t, op, ((int64_t) m), ((int64_t) M));
  *rop = (signed short) t;  
}

void double_binary32_to_unsigned_int(unsigned int * RESTRICT rop, CONST double_binary32_t * RESTRICT op) {
  unsigned int m, M;
  uint64_t t;

  m = (unsigned int) 0;
  M = m;
  M--;
  __double_binary32_to_unsigned_integer(&t, op, ((uint64_t) m), ((uint64_t) M));
  *rop = (unsigned int) t;
}

void double_binary32_to_signed_int(signed int * RESTRICT rop, CONST double_binary32_t * RESTRICT op) {
  unsigned int tt;
  signed int m, M, mt;
  int64_t t;

  tt = (unsigned int) 0;
  tt--;
  tt >>= 1u;
  M = (signed int) tt;
  m = -M;
  mt = m;
  mt--;
  if (mt < m) {
    m = mt;
  }
  __double_binary32_to_signed_integer(&t, op, ((int64_t) m), ((int64_t) M));
  *rop = (signed int) t;
}

void double_binary32_to_unsigned_long_int(unsigned long int * RESTRICT rop, CONST double_binary32_t * RESTRICT op) {
  unsigned long int m, M;
  uint64_t t;

  m = (unsigned long int) 0;
  M = m;
  M--;
  __double_binary32_to_unsigned_integer(&t, op, ((uint64_t) m), ((uint64_t) M));
  *rop = (unsigned long int) t;
}

void double_binary32_to_signed_long_int(signed long int * RESTRICT rop, CONST double_binary32_t * RESTRICT op) {
  unsigned long int tt;
  signed long int m, M, mt;
  int64_t t;

  tt = (unsigned long int) 0;
  tt--;
  tt >>= 1u;
  M = (signed long int) tt;
  m = -M;
  mt = m;
  mt--;
  if (mt < m) {
    m = mt;
  }
  __double_binary32_to_signed_integer(&t, op, ((int64_t) m), ((int64_t) M));
  *rop = (signed long int) t;
}

void double_binary32_to_unsigned_long_long_int(unsigned long long int * RESTRICT rop, CONST double_binary32_t * RESTRICT op) {
  unsigned long long int m, M;
  uint64_t t;

  m = (unsigned long long int) 0;
  M = m;
  M--;
  __double_binary32_to_unsigned_integer(&t, op, ((uint64_t) m), ((uint64_t) M));
  *rop = (unsigned long long int) t;
}

void double_binary32_to_signed_long_long_int(signed long long int * RESTRICT rop, CONST double_binary32_t * RESTRICT op) {
  unsigned long long int tt;
  signed long long int m, M, mt;
  int64_t t;

  tt = (unsigned long long int) 0;
  tt--;
  tt >>= 1u;
  M = (signed long long int) tt;
  m = -M;
  mt = m;
  mt--;
  if (mt < m) {
    m = mt;
  }
  __double_binary32_to_signed_integer(&t, op, ((int64_t) m), ((int64_t) M));
  *rop = (signed long long int) t;
}

void double_binary32_add(double_binary32_t * RESTRICT rop, CONST double_binary32_t * RESTRICT op1, CONST double_binary32_t * RESTRICT op2) {
  binary32_t t1, t2, t3, t4, hi, lo;

  __two_sum(&t1, &t2, op1->hi, op2->hi);
  t3 = op1->lo + op2->lo;
  t4 = t2 + t3;
  __fast_two_sum(&hi, &lo, t1, t4);
  rop->hi = hi;
  rop->lo = lo;
}

void double_binary32_sub(double_binary32_t * RESTRICT rop, CONST double_binary32_t * RESTRICT op1, CONST double_binary32_t * RESTRICT op2) {
  binary32_t t1, t2, t3, t4, hi, lo, op2hi, op2lo;

  op2hi = -op2->hi;
  op2lo = -op2->lo;
  __two_sum(&t1, &t2, op1->hi, op2hi);
  t3 = op1->lo + op2lo;
  t4 = t2 + t3;
  __fast_two_sum(&hi, &lo, t1, t4);
  rop->hi = hi;
  rop->lo = lo;
}

void double_binary32_mul(double_binary32_t * RESTRICT rop, CONST double_binary32_t * RESTRICT op1, CONST double_binary32_t * RESTRICT op2) {
  binary32_t t1, t2, t3, t4, t5, t6, hi, lo;

  __two_mul(&t1, &t2, op1->hi, op2->hi);
  t3 = op1->hi * op2->lo;
  t4 = op1->lo * op2->hi;
  t5 = t3 + t4;
  t6 = t5 + t2;
  __fast_two_sum(&hi, &lo, t1, t6);
  rop->hi = hi;
  rop->lo = lo;  
}

void double_binary32_div(double_binary32_t * RESTRICT rop, CONST double_binary32_t * RESTRICT op1, CONST double_binary32_t * RESTRICT op2) {
  binary32_t t1, t2, t3, t4, t5, t6, t7, t8, hi, lo;

  t1 = op1->hi / op2->hi;
  __two_mul(&t2, &t3, op2->hi, t1);
  t4 = op2->lo * t1;
  t5 = op1->hi - t2; /* Sterbenz */
  t6 = op1->lo - t4;
  t7 = t5 + t6;
  t8 = t7 / op2->hi;
  __fast_two_sum(&hi, &lo, t1, t8);
  rop->hi = hi;
  rop->lo = lo;
}

void double_binary32_neg(double_binary32_t * RESTRICT rop, CONST double_binary32_t * RESTRICT op) {
  rop->hi = -op->hi;
  rop->lo = -op->lo;
}
  
void double_binary32_sqrt(double_binary32_t * RESTRICT rop, CONST double_binary32_t * RESTRICT op) {
  binary32_t t1, t2, t3, t4, t5, t6, t7, t8, c, hi, lo;

  t1 = sqrtf(op->hi);
  __double_binary_div_double_by_single(&t2, &t3, op->hi, op->lo, t1);
  __two_sum(&t4, &t5, t1, t2);
  t6 = t5 + t3;
  c = 0.5f;
  t7 = c * t4;
  t8 = c * t6;
  __fast_two_sum(&hi, &lo, t7, t8);
  rop->hi = hi;
  rop->lo = lo;
}

void double_binary32_fabs(double_binary32_t * RESTRICT rop, CONST double_binary32_t * RESTRICT op) {
  if (op->hi >= -op->lo) {
    rop->hi = op->hi;
    rop->lo = op->lo;
    return;
  }
  rop->hi = -op->hi;
  rop->lo = -op->lo;
}

void double_binary32_compare(double_binary32_order_t * RESTRICT rop, CONST double_binary32_t * RESTRICT op1, CONST double_binary32_t * RESTRICT op2) {

  if (!((op1->hi == op1->hi) &&
	(op2->hi == op2->hi) &&
	(op1->lo == op1->lo) &&
	(op2->lo == op2->lo))) {
    /* Unordered */
    *rop = DOUBLE_BINARY32_UNORDERED;
  }

  if ((op1->hi > op2->hi) ||
      ((op1->hi == op2->hi) &&
       (op1->lo > op2->lo))) {
    /* Greater */
    *rop = DOUBLE_BINARY32_GREATER;
  }

  if ((op1->hi < op2->hi) ||
      ((op1->hi == op2->hi) &&
       (op1->lo < op2->lo))) {
    /* Less */
    *rop = DOUBLE_BINARY32_LESS;
  }

  if ((op1->hi == op2->hi) &&
      (op1->lo == op2->lo)) {
    /* Equal */
    *rop = DOUBLE_BINARY32_EQUAL;
  }

  /* Should never happen, but who knows */
  // Without a condition or else... it will always happen.
  // *rop = DOUBLE_BINARY32_UNORDERED;
}

