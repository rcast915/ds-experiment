
#pragma once

#ifndef FLOATINGPOINTIMPLEM_HPP
#define FLOATINGPOINTIMPLEM_HPP

#include <cstddef>
#include <cmath>

#include <iomanip>
#include <ios>
#include <sstream>
#include <string>
#include <iostream>

#include <typeindex>

extern "C" {
#include <double-binary32.h>
}

namespace floatingpointimplem {
  class fp {
  private:
    double_binary32_t value;

    fp(const double_binary32_t &v) {
      value = v;
    }
    
  public:
    
    fp() {
      const double_binary32_t val = { .hi = 0.0f, .lo = 0.0f };
      value = val;
    }

    fp(const signed char i) {
      double_binary32_from_signed_char(&value, i);
    }
    
    fp(const unsigned char i) {
      double_binary32_from_unsigned_char(&value, i);
    }

    fp(const signed short i) {
      double_binary32_from_signed_short(&value, i);
    }
    
    fp(const unsigned short i) {
      double_binary32_from_unsigned_short(&value, i);
    }
    
    fp(const signed int i) {
      double_binary32_from_signed_int(&value, i);
    }
    
    fp(const unsigned int i) {
      double_binary32_from_unsigned_int(&value, i);
    }

    fp(const signed long int i) {
      double_binary32_from_signed_long_int(&value, i);
    }
    
    fp(const unsigned long int i) {
      double_binary32_from_unsigned_long_int(&value, i);
    }

    fp(const signed long long int i) {
      double_binary32_from_signed_long_long_int(&value, i);
    }
    
    fp(const unsigned long long int i) {
      double_binary32_from_unsigned_long_long_int(&value, i);
    }
    
    fp(const float f) {
      double_binary32_from_binary32(&value, f);
    }
    
    fp(const double d) {
      double_binary32_from_binary64(&value, d);
    }
    
    fp(const fp &other) {
      value = other.value;
    }

    fp(fp &&other) {
      value = std::move(other.value);
    }
        
    friend bool operator== (const fp &a, const fp &b) {
      double_binary32_order_t order;
      double_binary32_compare(&order, &a.value, &b.value);
      const bool res = (order == DOUBLE_BINARY32_EQUAL);
      return res;
    }

    friend bool operator!= (const fp &a, const fp &b) {
      double_binary32_order_t order;
      double_binary32_compare(&order, &a.value, &b.value);
      const bool res = (order != DOUBLE_BINARY32_EQUAL);
      return res;
    }

    friend bool operator< (const fp &a, const fp &b) {
      double_binary32_order_t order;
      double_binary32_compare(&order, &a.value, &b.value);
      const bool res = (order == DOUBLE_BINARY32_LESS);
      return res;
    }

    friend bool operator<= (const fp &a, const fp &b) {
      double_binary32_order_t order;
      double_binary32_compare(&order, &a.value, &b.value);
      const bool res = ((order == DOUBLE_BINARY32_LESS) ||
			(order == DOUBLE_BINARY32_EQUAL));
      return res;
    }

    friend bool operator> (const fp &a, const fp &b) {
      double_binary32_order_t order;
      double_binary32_compare(&order, &a.value, &b.value);
      const bool res = (order == DOUBLE_BINARY32_GREATER);
      return res;
    }

    friend bool operator>= (const fp &a, const fp &b) {
      double_binary32_order_t order;
      double_binary32_compare(&order, &a.value, &b.value);
      const bool res = ((order == DOUBLE_BINARY32_GREATER) ||
			(order == DOUBLE_BINARY32_EQUAL));
      return res;
    }

    friend fp operator+ (const fp &a, const fp &b) {
      double_binary32_t res;
      double_binary32_add(&res, &a.value, &b.value);
      return fp(res);
    }
    
    friend fp operator- (const fp &a, const fp &b) {
      double_binary32_t res;
      double_binary32_sub(&res, &a.value, &b.value);
      return fp(res);
    }

    friend fp operator* (const fp &a, const fp &b) {
      double_binary32_t res;
      double_binary32_mul(&res, &a.value, &b.value);
      return fp(res);
    }

    friend fp operator/ (const fp &a, const fp &b) {
      double_binary32_t res;
      double_binary32_div(&res, &a.value, &b.value);
      return fp(res);
    }

    friend fp operator-(const fp &a) {
      double_binary32_t res;
      double_binary32_neg(&res, &a.value);
      return fp(res);
    }

    fp &operator= (const fp &other) {
      if (this == &other) return *this;
      value = other.value;
      return *this;
    }

    fp &operator= (fp &&other) {
      if (this == &other) return *this;
      value = std::move(other.value);
      return *this;
    }

    operator signed char() const {
      signed char res;
      double_binary32_to_signed_char(&res, &value);
      return res;
    }

    operator unsigned char() const {
      unsigned char res;
      double_binary32_to_unsigned_char(&res, &value);
      return res;
    }

    operator signed short() const {
      signed short res;
      double_binary32_to_signed_short(&res, &value);
      return res;
    }

    operator unsigned short() const {
      unsigned short res;
      double_binary32_to_unsigned_short(&res, &value);
      return res;
    }
    
    operator signed int() const {
      signed int res;
      double_binary32_to_signed_int(&res, &value);
      return res;
    }

    operator unsigned int() const {
      unsigned int res;
      double_binary32_to_unsigned_int(&res, &value);
      return res;
    }

    operator signed long int() const {
      signed long int res;
      double_binary32_to_signed_long_int(&res, &value);
      return res;
    }

    operator unsigned long int() const {
      unsigned long int res;
      double_binary32_to_unsigned_long_int(&res, &value);
      return res;
    }

    operator signed long long int() const {
      signed long long int res;
      double_binary32_to_signed_long_long_int(&res, &value);
      return res;
    }

    operator unsigned long long int() const {
      unsigned long long int res;
      double_binary32_to_unsigned_long_long_int(&res, &value);
      return res;
    }
    
    operator float() const {
      binary32_t res;
      double_binary32_to_binary32(&res, &value);
      return (float) res;
    }
    
    operator double() const {
      binary64_t res;
      double_binary32_to_binary64(&res, &value);
      return (double) res;
    }

    friend fp sqrt(const fp &x) {
      double_binary32_t res;
      double_binary32_sqrt(&res, &x.value);
      return fp(res);
    }

    friend fp fabs(const fp &x) {
      double_binary32_t res;
      double_binary32_fabs(&res, &x.value);
      return fp(res);
    }
    
    friend std::ostream& operator<< (std::ostream &out, const fp &x) {
      binary64_t val;
      
      double_binary32_to_binary64(&val, &x.value);
      return (out << val);
    }
    
  };
}

#endif

