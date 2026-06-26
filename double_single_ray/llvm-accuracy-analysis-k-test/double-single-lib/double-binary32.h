#ifndef DOUBLE_BINARY32_H
#define DOUBLE_BINARY32_H

#define CONST const
#define RESTRICT

typedef float binary32_t;
typedef double binary64_t;

typedef struct {
  binary32_t hi;
  binary32_t lo;
} double_binary32_t;

typedef enum {
  DOUBLE_BINARY32_LESS,
  DOUBLE_BINARY32_EQUAL,
  DOUBLE_BINARY32_GREATER,
  DOUBLE_BINARY32_UNORDERED
} double_binary32_order_t;

void double_binary32_from_binary32(double_binary32_t * RESTRICT rop, binary32_t op);
void double_binary32_from_binary64(double_binary32_t * RESTRICT rop, binary64_t op);
void double_binary32_from_unsigned_char(double_binary32_t * RESTRICT rop, unsigned char op);
void double_binary32_from_signed_char(double_binary32_t * RESTRICT rop, signed char op);
void double_binary32_from_unsigned_short(double_binary32_t * RESTRICT rop, unsigned short op);
void double_binary32_from_signed_short(double_binary32_t * RESTRICT rop, signed short op);
void double_binary32_from_unsigned_int(double_binary32_t * RESTRICT rop, unsigned int op);
void double_binary32_from_signed_int(double_binary32_t * RESTRICT rop, signed int op);
void double_binary32_from_unsigned_long_int(double_binary32_t * RESTRICT rop, unsigned long int op);
void double_binary32_from_signed_long_int(double_binary32_t * RESTRICT rop, signed long int op);
void double_binary32_from_unsigned_long_long_int(double_binary32_t * RESTRICT rop, unsigned long long int op);
void double_binary32_from_signed_long_long_int(double_binary32_t * RESTRICT rop, signed long long int op);

void double_binary32_to_binary32(binary32_t * RESTRICT rop, CONST double_binary32_t * RESTRICT op);
void double_binary32_to_binary64(binary64_t * RESTRICT rop, CONST double_binary32_t * RESTRICT op);
void double_binary32_to_unsigned_char(unsigned char * RESTRICT rop, CONST double_binary32_t * RESTRICT op);
void double_binary32_to_signed_char(signed char * RESTRICT rop, CONST double_binary32_t * RESTRICT op);
void double_binary32_to_unsigned_short(unsigned short * RESTRICT rop, CONST double_binary32_t * RESTRICT op);
void double_binary32_to_signed_short(signed short * RESTRICT rop, CONST double_binary32_t * RESTRICT op);
void double_binary32_to_unsigned_int(unsigned int * RESTRICT rop, CONST double_binary32_t * RESTRICT op);
void double_binary32_to_signed_int(signed int * RESTRICT rop, CONST double_binary32_t * RESTRICT op);
void double_binary32_to_unsigned_long_int(unsigned long int * RESTRICT rop, CONST double_binary32_t * RESTRICT op);
void double_binary32_to_signed_long_int(signed long int * RESTRICT rop, CONST double_binary32_t * RESTRICT op);
void double_binary32_to_unsigned_long_long_int(unsigned long long int * RESTRICT rop, CONST double_binary32_t * RESTRICT op);
void double_binary32_to_signed_long_long_int(signed long long int * RESTRICT rop, CONST double_binary32_t * RESTRICT op);

void double_binary32_add(double_binary32_t * RESTRICT rop, CONST double_binary32_t * RESTRICT op1, CONST double_binary32_t * RESTRICT op2);
void double_binary32_sub(double_binary32_t * RESTRICT rop, CONST double_binary32_t * RESTRICT op1, CONST double_binary32_t * RESTRICT op2);
void double_binary32_mul(double_binary32_t * RESTRICT rop, CONST double_binary32_t * RESTRICT op1, CONST double_binary32_t * RESTRICT op2);
void double_binary32_div(double_binary32_t * RESTRICT rop, CONST double_binary32_t * RESTRICT op1, CONST double_binary32_t * RESTRICT op2);
void double_binary32_neg(double_binary32_t * RESTRICT rop, CONST double_binary32_t * RESTRICT op);
void double_binary32_sqrt(double_binary32_t * RESTRICT rop, CONST double_binary32_t * RESTRICT op);

void double_binary32_fabs(double_binary32_t * RESTRICT rop, CONST double_binary32_t * RESTRICT op);

void double_binary32_compare(double_binary32_order_t * RESTRICT rop, CONST double_binary32_t * RESTRICT op1, CONST double_binary32_t * RESTRICT op2);

#endif // DOUBLE_BINARY32_H
