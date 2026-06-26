
#pragma once

#ifndef FLOATINGPOINT_HPP
#define FLOATINGPOINT_HPP

#include <floatingpointimplem.hpp>

#include <cstddef>
#include <cmath>

#include <iomanip>
#include <ios>
#include <sstream>
#include <string>

#include <typeindex> 
 
namespace floatingpoint {
  class floatingpoint {
  private:
    floatingpointimplem::fp value;

    floatingpoint(const floatingpointimplem::fp &a) {
      value = a;
    }

  public:
    floatingpoint() {
      value = 0.0;
    }

    floatingpoint(const signed char i) {
      value = i;
    }
    
    floatingpoint(const unsigned char i) {
      value = i;
    }

    floatingpoint(const signed short i) {
      value = i;
    }
    
    floatingpoint(const unsigned short i) {
      value = i;
    }
    
    floatingpoint(const signed int i) {
      value = i;
    }
    
    floatingpoint(const unsigned int i) {
      value = i;
    }

    floatingpoint(const signed long int i) {
      value = i;
    }
    
    floatingpoint(const unsigned long int i) {
      value = i;
    }

    floatingpoint(const signed long long int i) {
      value = i;
    }
    
    floatingpoint(const unsigned long long int i) {
      value = i;
    }
        
    floatingpoint(const float f) {
      value = f;
    }
    
    floatingpoint(const double d) {
      value = d;
    }
    
    floatingpoint(const floatingpoint &other) {
      value = other.value;
    }

    floatingpoint(floatingpoint &&other) {
      value = std::move(other.value);
    }

    friend bool operator== (const floatingpoint &a, const floatingpoint &b) {
      return (a.value == b.value);
    }

    friend bool operator!= (const floatingpoint &a, const floatingpoint &b) {
      return (a.value != b.value);
    }

    friend bool operator< (const floatingpoint &a, const floatingpoint &b) {
      return (a.value < b.value);
    }

    friend bool operator<= (const floatingpoint &a, const floatingpoint &b) {
      return (a.value <= b.value);
    }

    friend bool operator> (const floatingpoint &a, const floatingpoint &b) {
      return (a.value > b.value);
    }

    friend bool operator>= (const floatingpoint &a, const floatingpoint &b) {
      return (a.value >= b.value);
    }

    friend bool operator== (signed char a, const floatingpoint &b) {
      return (floatingpoint(a) == b);
    }

    friend bool operator!= (signed char a, const floatingpoint &b) {
      return (floatingpoint(a) != b);
    }

    friend bool operator< (signed char a, const floatingpoint &b) {
      return (floatingpoint(a) < b);
    }

    friend bool operator<= (signed char a, const floatingpoint &b) {
      return (floatingpoint(a) <= b);
    }

    friend bool operator> (signed char a, const floatingpoint &b) {
      return (floatingpoint(a) > b);
    }

    friend bool operator>= (signed char a, const floatingpoint &b) {
      return (floatingpoint(a) >= b);
    }

    friend bool operator== (unsigned char a, const floatingpoint &b) {
      return (floatingpoint(a) == b);
    }

    friend bool operator!= (unsigned char a, const floatingpoint &b) {
      return (floatingpoint(a) != b);
    }

    friend bool operator< (unsigned char a, const floatingpoint &b) {
      return (floatingpoint(a) < b);
    }

    friend bool operator<= (unsigned char a, const floatingpoint &b) {
      return (floatingpoint(a) <= b);
    }

    friend bool operator> (unsigned char a, const floatingpoint &b) {
      return (floatingpoint(a) > b);
    }

    friend bool operator>= (unsigned char a, const floatingpoint &b) {
      return (floatingpoint(a) >= b);
    }

    friend bool operator== (signed short a, const floatingpoint &b) {
      return (floatingpoint(a) == b);
    }

    friend bool operator!= (signed short a, const floatingpoint &b) {
      return (floatingpoint(a) != b);
    }

    friend bool operator< (signed short a, const floatingpoint &b) {
      return (floatingpoint(a) < b);
    }

    friend bool operator<= (signed short a, const floatingpoint &b) {
      return (floatingpoint(a) <= b);
    }

    friend bool operator> (signed short a, const floatingpoint &b) {
      return (floatingpoint(a) > b);
    }

    friend bool operator>= (signed short a, const floatingpoint &b) {
      return (floatingpoint(a) >= b);
    }

    friend bool operator== (unsigned short a, const floatingpoint &b) {
      return (floatingpoint(a) == b);
    }

    friend bool operator!= (unsigned short a, const floatingpoint &b) {
      return (floatingpoint(a) != b);
    }

    friend bool operator< (unsigned short a, const floatingpoint &b) {
      return (floatingpoint(a) < b);
    }

    friend bool operator<= (unsigned short a, const floatingpoint &b) {
      return (floatingpoint(a) <= b);
    }

    friend bool operator> (unsigned short a, const floatingpoint &b) {
      return (floatingpoint(a) > b);
    }

    friend bool operator>= (unsigned short a, const floatingpoint &b) {
      return (floatingpoint(a) >= b);
    }
    
    friend bool operator== (signed int a, const floatingpoint &b) {
      return (floatingpoint(a) == b);
    }

    friend bool operator!= (signed int a, const floatingpoint &b) {
      return (floatingpoint(a) != b);
    }

    friend bool operator< (signed int a, const floatingpoint &b) {
      return (floatingpoint(a) < b);
    }

    friend bool operator<= (signed int a, const floatingpoint &b) {
      return (floatingpoint(a) <= b);
    }

    friend bool operator> (signed int a, const floatingpoint &b) {
      return (floatingpoint(a) > b);
    }

    friend bool operator>= (signed int a, const floatingpoint &b) {
      return (floatingpoint(a) >= b);
    }

    friend bool operator== (unsigned int a, const floatingpoint &b) {
      return (floatingpoint(a) == b);
    }

    friend bool operator!= (unsigned int a, const floatingpoint &b) {
      return (floatingpoint(a) != b);
    }

    friend bool operator< (unsigned int a, const floatingpoint &b) {
      return (floatingpoint(a) < b);
    }

    friend bool operator<= (unsigned int a, const floatingpoint &b) {
      return (floatingpoint(a) <= b);
    }

    friend bool operator> (unsigned int a, const floatingpoint &b) {
      return (floatingpoint(a) > b);
    }

    friend bool operator>= (unsigned int a, const floatingpoint &b) {
      return (floatingpoint(a) >= b);
    }

    friend bool operator== (signed long int a, const floatingpoint &b) {
      return (floatingpoint(a) == b);
    }

    friend bool operator!= (signed long int a, const floatingpoint &b) {
      return (floatingpoint(a) != b);
    }

    friend bool operator< (signed long int a, const floatingpoint &b) {
      return (floatingpoint(a) < b);
    }

    friend bool operator<= (signed long int a, const floatingpoint &b) {
      return (floatingpoint(a) <= b);
    }

    friend bool operator> (signed long int a, const floatingpoint &b) {
      return (floatingpoint(a) > b);
    }

    friend bool operator>= (signed long int a, const floatingpoint &b) {
      return (floatingpoint(a) >= b);
    }

    friend bool operator== (unsigned long int a, const floatingpoint &b) {
      return (floatingpoint(a) == b);
    }

    friend bool operator!= (unsigned long int a, const floatingpoint &b) {
      return (floatingpoint(a) != b);
    }

    friend bool operator< (unsigned long int a, const floatingpoint &b) {
      return (floatingpoint(a) < b);
    }

    friend bool operator<= (unsigned long int a, const floatingpoint &b) {
      return (floatingpoint(a) <= b);
    }

    friend bool operator> (unsigned long int a, const floatingpoint &b) {
      return (floatingpoint(a) > b);
    }

    friend bool operator>= (unsigned long int a, const floatingpoint &b) {
      return (floatingpoint(a) >= b);
    }

    friend bool operator== (signed long long int a, const floatingpoint &b) {
      return (floatingpoint(a) == b);
    }

    friend bool operator!= (signed long long int a, const floatingpoint &b) {
      return (floatingpoint(a) != b);
    }

    friend bool operator< (signed long long int a, const floatingpoint &b) {
      return (floatingpoint(a) < b);
    }

    friend bool operator<= (signed long long int a, const floatingpoint &b) {
      return (floatingpoint(a) <= b);
    }

    friend bool operator> (signed long long int a, const floatingpoint &b) {
      return (floatingpoint(a) > b);
    }

    friend bool operator>= (signed long long int a, const floatingpoint &b) {
      return (floatingpoint(a) >= b);
    }

    friend bool operator== (unsigned long long int a, const floatingpoint &b) {
      return (floatingpoint(a) == b);
    }

    friend bool operator!= (unsigned long long int a, const floatingpoint &b) {
      return (floatingpoint(a) != b);
    }

    friend bool operator< (unsigned long long int a, const floatingpoint &b) {
      return (floatingpoint(a) < b);
    }

    friend bool operator<= (unsigned long long int a, const floatingpoint &b) {
      return (floatingpoint(a) <= b);
    }

    friend bool operator> (unsigned long long int a, const floatingpoint &b) {
      return (floatingpoint(a) > b);
    }

    friend bool operator>= (unsigned long long int a, const floatingpoint &b) {
      return (floatingpoint(a) >= b);
    }
    
    friend bool operator== (float a, const floatingpoint &b) {
      return (floatingpoint(a) == b);
    }

    friend bool operator!= (float a, const floatingpoint &b) {
      return (floatingpoint(a) != b);
    }

    friend bool operator< (float a, const floatingpoint &b) {
      return (floatingpoint(a) < b);
    }

    friend bool operator<= (float a, const floatingpoint &b) {
      return (floatingpoint(a) <= b);
    }

    friend bool operator> (float a, const floatingpoint &b) {
      return (floatingpoint(a) > b);
    }

    friend bool operator>= (float a, const floatingpoint &b) {
      return (floatingpoint(a) >= b);
    }

    friend bool operator== (double a, const floatingpoint &b) {
      return (floatingpoint(a) == b);
    }

    friend bool operator!= (double a, const floatingpoint &b) {
      return (floatingpoint(a) != b);
    }

    friend bool operator< (double a, const floatingpoint &b) {
      return (floatingpoint(a) < b);
    }

    friend bool operator<= (double a, const floatingpoint &b) {
      return (floatingpoint(a) <= b);
    }

    friend bool operator> (double a, const floatingpoint &b) {
      return (floatingpoint(a) > b);
    }

    friend bool operator>= (double a, const floatingpoint &b) {
      return (floatingpoint(a) >= b);
    }

    friend bool operator== (const floatingpoint &a, signed char b) {
      return (a == floatingpoint(b));
    }

    friend bool operator!= (const floatingpoint &a, signed char b) {
      return (a != floatingpoint(b));
    }

    friend bool operator< (const floatingpoint &a, signed char b) {
      return (a < floatingpoint(b));
    }

    friend bool operator<= (const floatingpoint &a, signed char b) {
      return (a <= floatingpoint(b));
    }

    friend bool operator> (const floatingpoint &a, signed char b) {
      return (a > floatingpoint(b));
    }

    friend bool operator>= (const floatingpoint &a, signed char b) {
      return (a >= floatingpoint(b));
    }

    friend bool operator== (const floatingpoint &a, unsigned char b) {
      return (a == floatingpoint(b));
    }

    friend bool operator!= (const floatingpoint &a, unsigned char b) {
      return (a != floatingpoint(b));
    }

    friend bool operator< (const floatingpoint &a, unsigned char b) {
      return (a < floatingpoint(b));
    }

    friend bool operator<= (const floatingpoint &a, unsigned char b) {
      return (a <= floatingpoint(b));
    }

    friend bool operator> (const floatingpoint &a, unsigned char b) {
      return (a > floatingpoint(b));
    }

    friend bool operator>= (const floatingpoint &a, unsigned char b) {
      return (a >= floatingpoint(b));
    }
    
    friend bool operator== (const floatingpoint &a, signed short b) {
      return (a == floatingpoint(b));
    }

    friend bool operator!= (const floatingpoint &a, signed short b) {
      return (a != floatingpoint(b));
    }

    friend bool operator< (const floatingpoint &a, signed short b) {
      return (a < floatingpoint(b));
    }

    friend bool operator<= (const floatingpoint &a, signed short b) {
      return (a <= floatingpoint(b));
    }

    friend bool operator> (const floatingpoint &a, signed short b) {
      return (a > floatingpoint(b));
    }

    friend bool operator>= (const floatingpoint &a, signed short b) {
      return (a >= floatingpoint(b));
    }

    friend bool operator== (const floatingpoint &a, unsigned short b) {
      return (a == floatingpoint(b));
    }

    friend bool operator!= (const floatingpoint &a, unsigned short b) {
      return (a != floatingpoint(b));
    }

    friend bool operator< (const floatingpoint &a, unsigned short b) {
      return (a < floatingpoint(b));
    }

    friend bool operator<= (const floatingpoint &a, unsigned short b) {
      return (a <= floatingpoint(b));
    }

    friend bool operator> (const floatingpoint &a, unsigned short b) {
      return (a > floatingpoint(b));
    }

    friend bool operator>= (const floatingpoint &a, unsigned short b) {
      return (a >= floatingpoint(b));
    }

    friend bool operator== (const floatingpoint &a, signed int b) {
      return (a == floatingpoint(b));
    }

    friend bool operator!= (const floatingpoint &a, signed int b) {
      return (a != floatingpoint(b));
    }

    friend bool operator< (const floatingpoint &a, signed int b) {
      return (a < floatingpoint(b));
    }

    friend bool operator<= (const floatingpoint &a, signed int b) {
      return (a <= floatingpoint(b));
    }

    friend bool operator> (const floatingpoint &a, signed int b) {
      return (a > floatingpoint(b));
    }

    friend bool operator>= (const floatingpoint &a, signed int b) {
      return (a >= floatingpoint(b));
    }

    friend bool operator== (const floatingpoint &a, unsigned int b) {
      return (a == floatingpoint(b));
    }

    friend bool operator!= (const floatingpoint &a, unsigned int b) {
      return (a != floatingpoint(b));
    }

    friend bool operator< (const floatingpoint &a, unsigned int b) {
      return (a < floatingpoint(b));
    }

    friend bool operator<= (const floatingpoint &a, unsigned int b) {
      return (a <= floatingpoint(b));
    }

    friend bool operator> (const floatingpoint &a, unsigned int b) {
      return (a > floatingpoint(b));
    }

    friend bool operator>= (const floatingpoint &a, unsigned int b) {
      return (a >= floatingpoint(b));
    }

    friend bool operator== (const floatingpoint &a, signed long int b) {
      return (a == floatingpoint(b));
    }

    friend bool operator!= (const floatingpoint &a, signed long int b) {
      return (a != floatingpoint(b));
    }

    friend bool operator< (const floatingpoint &a, signed long int b) {
      return (a < floatingpoint(b));
    }

    friend bool operator<= (const floatingpoint &a, signed long int b) {
      return (a <= floatingpoint(b));
    }

    friend bool operator> (const floatingpoint &a, signed long int b) {
      return (a > floatingpoint(b));
    }

    friend bool operator>= (const floatingpoint &a, signed long int b) {
      return (a >= floatingpoint(b));
    }

    friend bool operator== (const floatingpoint &a, unsigned long int b) {
      return (a == floatingpoint(b));
    }

    friend bool operator!= (const floatingpoint &a, unsigned long int b) {
      return (a != floatingpoint(b));
    }

    friend bool operator< (const floatingpoint &a, unsigned long int b) {
      return (a < floatingpoint(b));
    }

    friend bool operator<= (const floatingpoint &a, unsigned long int b) {
      return (a <= floatingpoint(b));
    }

    friend bool operator> (const floatingpoint &a, unsigned long int b) {
      return (a > floatingpoint(b));
    }

    friend bool operator>= (const floatingpoint &a, unsigned long int b) {
      return (a >= floatingpoint(b));
    }

    friend bool operator== (const floatingpoint &a, signed long long int b) {
      return (a == floatingpoint(b));
    }

    friend bool operator!= (const floatingpoint &a, signed long long int b) {
      return (a != floatingpoint(b));
    }

    friend bool operator< (const floatingpoint &a, signed long long int b) {
      return (a < floatingpoint(b));
    }

    friend bool operator<= (const floatingpoint &a, signed long long int b) {
      return (a <= floatingpoint(b));
    }

    friend bool operator> (const floatingpoint &a, signed long long int b) {
      return (a > floatingpoint(b));
    }

    friend bool operator>= (const floatingpoint &a, signed long long int b) {
      return (a >= floatingpoint(b));
    }

    friend bool operator== (const floatingpoint &a, unsigned long long int b) {
      return (a == floatingpoint(b));
    }

    friend bool operator!= (const floatingpoint &a, unsigned long long int b) {
      return (a != floatingpoint(b));
    }

    friend bool operator< (const floatingpoint &a, unsigned long long int b) {
      return (a < floatingpoint(b));
    }

    friend bool operator<= (const floatingpoint &a, unsigned long long int b) {
      return (a <= floatingpoint(b));
    }

    friend bool operator> (const floatingpoint &a, unsigned long long int b) {
      return (a > floatingpoint(b));
    }

    friend bool operator>= (const floatingpoint &a, unsigned long long int b) {
      return (a >= floatingpoint(b));
    }

    friend bool operator== (const floatingpoint &a, float b) {
      return (a == floatingpoint(b));
    }

    friend bool operator!= (const floatingpoint &a, float b) {
      return (a != floatingpoint(b));
    }

    friend bool operator< (const floatingpoint &a, float b) {
      return (a < floatingpoint(b));
    }

    friend bool operator<= (const floatingpoint &a, float b) {
      return (a <= floatingpoint(b));
    }

    friend bool operator> (const floatingpoint &a, float b) {
      return (a > floatingpoint(b));
    }

    friend bool operator>= (const floatingpoint &a, float b) {
      return (a >= floatingpoint(b));
    }

    friend bool operator== (const floatingpoint &a, double b) {
      return (a == floatingpoint(b));
    }

    friend bool operator!= (const floatingpoint &a, double b) {
      return (a != floatingpoint(b));
    }

    friend bool operator< (const floatingpoint &a, double b) {
      return (a < floatingpoint(b));
    }

    friend bool operator<= (const floatingpoint &a, double b) {
      return (a <= floatingpoint(b));
    }

    friend bool operator> (const floatingpoint &a, double b) {
      return (a > floatingpoint(b));
    }

    friend bool operator>= (const floatingpoint &a, double b) {
      return (a >= floatingpoint(b));
    }

    friend floatingpoint operator+ (const floatingpoint &a, const floatingpoint &b) {
      return floatingpoint(a.value + b.value);
    }
    
    friend floatingpoint operator- (const floatingpoint &a, const floatingpoint &b) {
      return floatingpoint(a.value - b.value);
    }

    friend floatingpoint operator* (const floatingpoint &a, const floatingpoint &b) {
      return floatingpoint(a.value * b.value);
    }

    friend floatingpoint operator/ (const floatingpoint &a, const floatingpoint &b) {
      return floatingpoint(a.value / b.value);
    }
    
    friend floatingpoint operator+ (signed char a, const floatingpoint &b) {
      return floatingpoint(a) + b;
    }
    
    friend floatingpoint operator- (signed char a, const floatingpoint &b) {
      return floatingpoint(a) - b;
    }

    friend floatingpoint operator* (signed char a, const floatingpoint &b) {
      return floatingpoint(a) * b;
    }

    friend floatingpoint operator/ (signed char a, const floatingpoint &b) {
      return floatingpoint(a) / b;
    }

    friend floatingpoint operator+ (unsigned char a, const floatingpoint &b) {
      return floatingpoint(a) + b;
    }
    
    friend floatingpoint operator- (unsigned char a, const floatingpoint &b) {
      return floatingpoint(a) - b;
    }

    friend floatingpoint operator* (unsigned char a, const floatingpoint &b) {
      return floatingpoint(a) * b;
    }

    friend floatingpoint operator/ (unsigned char a, const floatingpoint &b) {
      return floatingpoint(a) / b;
    }

    friend floatingpoint operator+ (signed short a, const floatingpoint &b) {
      return floatingpoint(a) + b;
    }
    
    friend floatingpoint operator- (signed short a, const floatingpoint &b) {
      return floatingpoint(a) - b;
    }

    friend floatingpoint operator* (signed short a, const floatingpoint &b) {
      return floatingpoint(a) * b;
    }

    friend floatingpoint operator/ (signed short a, const floatingpoint &b) {
      return floatingpoint(a) / b;
    }

    friend floatingpoint operator+ (unsigned short a, const floatingpoint &b) {
      return floatingpoint(a) + b;
    }
    
    friend floatingpoint operator- (unsigned short a, const floatingpoint &b) {
      return floatingpoint(a) - b;
    }

    friend floatingpoint operator* (unsigned short a, const floatingpoint &b) {
      return floatingpoint(a) * b;
    }

    friend floatingpoint operator/ (unsigned short a, const floatingpoint &b) {
      return floatingpoint(a) / b;
    }

    friend floatingpoint operator+ (signed int a, const floatingpoint &b) {
      return floatingpoint(a) + b;
    }
    
    friend floatingpoint operator- (signed int a, const floatingpoint &b) {
      return floatingpoint(a) - b;
    }

    friend floatingpoint operator* (signed int a, const floatingpoint &b) {
      return floatingpoint(a) * b;
    }

    friend floatingpoint operator/ (signed int a, const floatingpoint &b) {
      return floatingpoint(a) / b;
    }

    friend floatingpoint operator+ (unsigned int a, const floatingpoint &b) {
      return floatingpoint(a) + b;
    }
    
    friend floatingpoint operator- (unsigned int a, const floatingpoint &b) {
      return floatingpoint(a) - b;
    }

    friend floatingpoint operator* (unsigned int a, const floatingpoint &b) {
      return floatingpoint(a) * b;
    }

    friend floatingpoint operator/ (unsigned int a, const floatingpoint &b) {
      return floatingpoint(a) / b;
    }
    
    friend floatingpoint operator+ (signed long int a, const floatingpoint &b) {
      return floatingpoint(a) + b;
    }
    
    friend floatingpoint operator- (signed long int a, const floatingpoint &b) {
      return floatingpoint(a) - b;
    }

    friend floatingpoint operator* (signed long int a, const floatingpoint &b) {
      return floatingpoint(a) * b;
    }

    friend floatingpoint operator/ (signed long int a, const floatingpoint &b) {
      return floatingpoint(a) / b;
    }

    friend floatingpoint operator+ (unsigned long int a, const floatingpoint &b) {
      return floatingpoint(a) + b;
    }
    
    friend floatingpoint operator- (unsigned long int a, const floatingpoint &b) {
      return floatingpoint(a) - b;
    }

    friend floatingpoint operator* (unsigned long int a, const floatingpoint &b) {
      return floatingpoint(a) * b;
    }

    friend floatingpoint operator/ (unsigned long int a, const floatingpoint &b) {
      return floatingpoint(a) / b;
    }
    
    friend floatingpoint operator+ (signed long long int a, const floatingpoint &b) {
      return floatingpoint(a) + b;
    }
    
    friend floatingpoint operator- (signed long long int a, const floatingpoint &b) {
      return floatingpoint(a) - b;
    }

    friend floatingpoint operator* (signed long long int a, const floatingpoint &b) {
      return floatingpoint(a) * b;
    }

    friend floatingpoint operator/ (signed long long int a, const floatingpoint &b) {
      return floatingpoint(a) / b;
    }

    friend floatingpoint operator+ (unsigned long long int a, const floatingpoint &b) {
      return floatingpoint(a) + b;
    }
    
    friend floatingpoint operator- (unsigned long long int a, const floatingpoint &b) {
      return floatingpoint(a) - b;
    }

    friend floatingpoint operator* (unsigned long long int a, const floatingpoint &b) {
      return floatingpoint(a) * b;
    }

    friend floatingpoint operator/ (unsigned long long int a, const floatingpoint &b) {
      return floatingpoint(a) / b;
    }
    
    friend floatingpoint operator+ (float a, const floatingpoint &b) {
      return floatingpoint(a) + b;
    }
    
    friend floatingpoint operator- (float a, const floatingpoint &b) {
      return floatingpoint(a) - b;
    }

    friend floatingpoint operator* (float a, const floatingpoint &b) {
      return floatingpoint(a) * b;
    }

    friend floatingpoint operator/ (float a, const floatingpoint &b) {
      return floatingpoint(a) / b;
    }
    
    friend floatingpoint operator+ (double a, const floatingpoint &b) {
      return floatingpoint(a) + b;
    }
    
    friend floatingpoint operator- (double a, const floatingpoint &b) {
      return floatingpoint(a) - b;
    }

    friend floatingpoint operator* (double a, const floatingpoint &b) {
      return floatingpoint(a) * b;
    }

    friend floatingpoint operator/ (double a, const floatingpoint &b) {
      return floatingpoint(a) / b;
    }
    
    friend floatingpoint operator+ (const floatingpoint &a, signed char b) {
      return a + floatingpoint(b);
    }

    friend floatingpoint operator- (const floatingpoint &a, signed char b) {
      return a - floatingpoint(b);
    }

    friend floatingpoint operator* (const floatingpoint &a, signed char b) {
      return a * floatingpoint(b);
    }

    friend floatingpoint operator/ (const floatingpoint &a, signed char b) {
      return a / floatingpoint(b);
    }

    friend floatingpoint operator+ (const floatingpoint &a, unsigned char b) {
      return a + floatingpoint(b);
    }

    friend floatingpoint operator- (const floatingpoint &a, unsigned char b) {
      return a - floatingpoint(b);
    }

    friend floatingpoint operator* (const floatingpoint &a, unsigned char b) {
      return a * floatingpoint(b);
    }

    friend floatingpoint operator/ (const floatingpoint &a, unsigned char b) {
      return a / floatingpoint(b);
    }

    friend floatingpoint operator+ (const floatingpoint &a, signed short b) {
      return a + floatingpoint(b);
    }

    friend floatingpoint operator- (const floatingpoint &a, signed short b) {
      return a - floatingpoint(b);
    }

    friend floatingpoint operator* (const floatingpoint &a, signed short b) {
      return a * floatingpoint(b);
    }

    friend floatingpoint operator/ (const floatingpoint &a, signed short b) {
      return a / floatingpoint(b);
    }

    friend floatingpoint operator+ (const floatingpoint &a, unsigned short b) {
      return a + floatingpoint(b);
    }

    friend floatingpoint operator- (const floatingpoint &a, unsigned short b) {
      return a - floatingpoint(b);
    }

    friend floatingpoint operator* (const floatingpoint &a, unsigned short b) {
      return a * floatingpoint(b);
    }

    friend floatingpoint operator/ (const floatingpoint &a, unsigned short b) {
      return a / floatingpoint(b);
    }

    friend floatingpoint operator+ (const floatingpoint &a, signed int b) {
      return a + floatingpoint(b);
    }

    friend floatingpoint operator- (const floatingpoint &a, signed int b) {
      return a - floatingpoint(b);
    }

    friend floatingpoint operator* (const floatingpoint &a, signed int b) {
      return a * floatingpoint(b);
    }

    friend floatingpoint operator/ (const floatingpoint &a, signed int b) {
      return a / floatingpoint(b);
    }

    friend floatingpoint operator+ (const floatingpoint &a, unsigned int b) {
      return a + floatingpoint(b);
    }

    friend floatingpoint operator- (const floatingpoint &a, unsigned int b) {
      return a - floatingpoint(b);
    }

    friend floatingpoint operator* (const floatingpoint &a, unsigned int b) {
      return a * floatingpoint(b);
    }

    friend floatingpoint operator/ (const floatingpoint &a, unsigned int b) {
      return a / floatingpoint(b);
    }

    friend floatingpoint operator+ (const floatingpoint &a, signed long int b) {
      return a + floatingpoint(b);
    }

    friend floatingpoint operator- (const floatingpoint &a, signed long int b) {
      return a - floatingpoint(b);
    }

    friend floatingpoint operator* (const floatingpoint &a, signed long int b) {
      return a * floatingpoint(b);
    }

    friend floatingpoint operator/ (const floatingpoint &a, signed long int b) {
      return a / floatingpoint(b);
    }

    friend floatingpoint operator+ (const floatingpoint &a, unsigned long int b) {
      return a + floatingpoint(b);
    }

    friend floatingpoint operator- (const floatingpoint &a, unsigned long int b) {
      return a - floatingpoint(b);
    }

    friend floatingpoint operator* (const floatingpoint &a, unsigned long int b) {
      return a * floatingpoint(b);
    }

    friend floatingpoint operator/ (const floatingpoint &a, unsigned long int b) {
      return a / floatingpoint(b);
    }

    friend floatingpoint operator+ (const floatingpoint &a, signed long long int b) {
      return a + floatingpoint(b);
    }

    friend floatingpoint operator- (const floatingpoint &a, signed long long int b) {
      return a - floatingpoint(b);
    }

    friend floatingpoint operator* (const floatingpoint &a, signed long long int b) {
      return a * floatingpoint(b);
    }

    friend floatingpoint operator/ (const floatingpoint &a, signed long long int b) {
      return a / floatingpoint(b);
    }

    friend floatingpoint operator+ (const floatingpoint &a, unsigned long long int b) {
      return a + floatingpoint(b);
    }

    friend floatingpoint operator- (const floatingpoint &a, unsigned long long int b) {
      return a - floatingpoint(b);
    }

    friend floatingpoint operator* (const floatingpoint &a, unsigned long long int b) {
      return a * floatingpoint(b);
    }

    friend floatingpoint operator/ (const floatingpoint &a, unsigned long long int b) {
      return a / floatingpoint(b);
    }
    
    friend floatingpoint operator+ (const floatingpoint &a, float b) {
      return a + floatingpoint(b);
    }

    friend floatingpoint operator- (const floatingpoint &a, float b) {
      return a - floatingpoint(b);
    }

    friend floatingpoint operator* (const floatingpoint &a, float b) {
      return a * floatingpoint(b);
    }

    friend floatingpoint operator/ (const floatingpoint &a, float b) {
      return a / floatingpoint(b);
    }

    friend floatingpoint operator+ (const floatingpoint &a, double b) {
      return a + floatingpoint(b);
    }

    friend floatingpoint operator- (const floatingpoint &a, double b) {
      return a - floatingpoint(b);
    }

    friend floatingpoint operator* (const floatingpoint &a, double b) {
      return a * floatingpoint(b);
    }

    friend floatingpoint operator/ (const floatingpoint &a, double b) {
      return a / floatingpoint(b);
    }
    
    friend floatingpoint operator-(const floatingpoint &a) {
      return floatingpoint(-a.value);
    }

    floatingpoint &operator= (const floatingpoint &other) {
      if (this == &other) return *this;
      value = other.value;
      return *this;
    }

    floatingpoint &operator= (floatingpoint &&other) {
      if (this == &other) return *this;
      value = std::move(other.value);
      return *this;
    }

    floatingpoint &operator+= (const floatingpoint &other) {
      value = value + other.value;
      return *this;
    }

    floatingpoint &operator-= (const floatingpoint &other) {
      value = value - other.value;
      return *this;
    }

    floatingpoint &operator*= (const floatingpoint &other) {
      value = value * other.value;
      return *this;
    }

    floatingpoint &operator/= (const floatingpoint &other) {
      value = value / other.value;
      return *this;
    }

    operator signed char() const {
      return ((signed char) value);
    }

    operator unsigned char() const {
      return ((unsigned char) value);
    }

    operator signed short() const {
      return ((signed char) value);
    }

    operator unsigned short() const {
      return ((unsigned char) value);
    }
    
    operator signed int() const {
      return ((signed int) value);
    }

    operator unsigned int() const {
      return ((unsigned int) value);
    }

    operator signed long int() const {
      return ((signed long int) value);
    }

    operator unsigned long int() const {
      return ((unsigned long int) value);
    }

    operator signed long long int() const {
      return ((signed long long int) value);
    }

    operator unsigned long long int() const {
      return ((unsigned long long int) value);
    }
        
    operator float() const {
      return ((float) value);
    }
    
    operator double() const {
      return ((double) value);
    }

    friend floatingpoint sqrt(const floatingpoint &x) {
      return floatingpoint(sqrt(x.value));
    }

    friend floatingpoint fabs(const floatingpoint &x) {
      return floatingpoint(fabs(x.value));
    }
    
    friend std::ostream& operator<< (std::ostream &out, const floatingpoint & x) {
      return (out << (x.value));
    }
    
  };
}

#endif

