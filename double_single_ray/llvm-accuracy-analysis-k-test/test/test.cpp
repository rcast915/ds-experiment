#include <floatingpoint.hpp>
#include <iostream>
#include <cmath>

using namespace std;

constexpr float delta = 1e-10;

void print_passed() {
    std::cout << "\033[32m✔ Test Passed\033[0m\n"; // ANSI Codes
}

void print_failed(floatingpoint::floatingpoint expected, floatingpoint::floatingpoint obtained) {
    std::cout << "\033[31m✘ Test Failed\033[0m\n";
    std::cout << "Expected: " << expected << ", Obtained: " << obtained << "\n";
}

// Function to test addition
void testAddition(floatingpoint::floatingpoint x, floatingpoint::floatingpoint y, floatingpoint::floatingpoint expected_add) {
    floatingpoint::floatingpoint z = x + y;
    std::cout << "Addition: " << x << " + " << y << std::endl;
    std::cout << "Expected: " << expected_add << ", Result: " << z << std::endl;

    if (fabs(z - expected_add) <= delta) {
        print_passed();
    } else {
        print_failed(expected_add, z);
    }
    std::cout << std::endl;
}

// Function to test subtraction
void testSubtraction(floatingpoint::floatingpoint x, floatingpoint::floatingpoint y, floatingpoint::floatingpoint expected_sub) {
    floatingpoint::floatingpoint z = y - x;
    std::cout << "Subtraction: " << y << " - " << x << std::endl;
    std::cout << "Expected: " << expected_sub << ", Result: " << z << std::endl;

    if (fabs(z - expected_sub) <= delta) {
        print_passed();
    } else {
        print_failed(expected_sub, z);
    }
    std::cout << std::endl;
}

// Function to test multiplication
void testMultiplication(floatingpoint::floatingpoint x, floatingpoint::floatingpoint y, floatingpoint::floatingpoint expected_mul) {
    floatingpoint::floatingpoint z = x * y;
    std::cout << "Multiplication: " << x << " * " << y << std::endl;
    std::cout << "Expected: " << expected_mul << ", Result: " << z << std::endl;

    if (fabs(z - expected_mul) <= delta) {
        print_passed();
    } else {
        print_failed(expected_mul, z);
    }
    std::cout << std::endl;
}

// Function to test division
void testDivision(floatingpoint::floatingpoint x, floatingpoint::floatingpoint y, floatingpoint::floatingpoint expected_div) {
    floatingpoint::floatingpoint z = y / x;
    std::cout << "Division: " << y << " / " << x << std::endl;
    std::cout << "Expected: " << expected_div << ", Result: " << z << std::endl;

    if (fabs(z - expected_div) <= delta) {
        print_passed();
    } else {
        print_failed(expected_div, z);
    }
    std::cout << std::endl;
}




// Function to test square root operation
void test_square_root() {
    floatingpoint::floatingpoint x = 9.000000000000001;
    floatingpoint::floatingpoint result = sqrt(x);
    floatingpoint::floatingpoint expected_sqrt = 3.0000000000000001;

    if (fabs(result - expected_sqrt) < delta) {
        print_passed();
    } else {
        print_failed(expected_sqrt, result);
    }
}

// Function to test special cases (0, infinity, NaN)
void test_special_cases() {
    // Test for infinity
    floatingpoint::floatingpoint x = 0.0;
    floatingpoint::floatingpoint inf = 1.0 / x;

    if (isinf(inf)) {
        print_passed();
    } else {
        std::cout << "\033[31m✘ Infinity Test Failed\033[0m\n";
    }

    // Test for NaN
    floatingpoint::floatingpoint nan = sqrt(-1.0);
    if (isnan(nan)) {
        print_passed();
    } else {
        std::cout << "\033[31m✘ NaN Test Failed\033[0m\n";
    }
}

// Function to test underflow and overflow
void test_underflow_overflow() {
    // Test for underflow
    floatingpoint::floatingpoint small = delta;
    floatingpoint::floatingpoint smaller = small / 1e10;

    floatingpoint::floatingpoint expected_underflow = 0.0;
    if (smaller == expected_underflow) {
        print_passed();
    } else {
        print_failed(expected_underflow, smaller);
    }

    // Test for overflow
    floatingpoint::floatingpoint large = 1e10;
    floatingpoint::floatingpoint larger = large * 1e10;

    if (isinf(larger)) {
        print_passed();
    } else {
        std::cout << "\033[31m✘ Overflow Test Failed\033[0m\n";
    }
}

// Function to test precision with high decimal values
void test_precision() {
    floatingpoint::floatingpoint x = 0.1234567890123456789;
    floatingpoint::floatingpoint y = 0.0000000000000000012;
    floatingpoint::floatingpoint z = x + y;

    floatingpoint::floatingpoint expected_precision = 0.1234567890123456801;
    if (fabs(z - expected_precision) < delta) {
        print_passed();
    } else {
        print_failed(expected_precision, z);
    }
}

int main(int argc, char **argv) {
    floatingpoint::floatingpoint x = 1.0000000000000001;
    floatingpoint::floatingpoint y = 2.0000000000000001;

    testAddition(x, y, 3.0000000000000002);
    testSubtraction(x, y, 1.0000000000000000);
    testMultiplication(x, y, 2.0000000000000004);
    testDivision(x, y, 2.0000000000000000);

  return 0;
}
