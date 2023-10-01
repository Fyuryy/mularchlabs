
CC = gcc
CFLAGS = -std=gnu99 -O3 -fopenmp -Wall -lpthread

all: pi pi_pthreads integral test test2

pi: pi.c utility.h
	$(CC) $(CFLAGS) $< -o $@

pi_pthreads: pi_pthreads.c utility.h
	$(CC) $(CFLAGS) $< -o $@

integral: integral.c function.o utility.h
	$(CC) $(CFLAGS) $< -o $@

test: omp_hello.c
	$(CC) $(CFLAGS) $< -o $@

test2: pthreads_hello.c
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -f pi pi_pthreads integral function.o test test2
