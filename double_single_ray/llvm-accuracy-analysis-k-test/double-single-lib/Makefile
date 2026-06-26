
all: test

test: test.o double-binary32.o
	g++ -I. -Wall -g -O0 -o $@ $^

test.o: test.cpp floatingpoint.hpp floatingpointimplem.hpp
	g++ -I. -Wall -g -O0 -c $<

double-binary32.o: double-binary32.c double-binary32.h
	gcc -I. -Wall -g -O0 -c $<

clean:
	rm -f test.o
	rm -f double-binary32.o
	rm -f test

.PHONY: all clean


