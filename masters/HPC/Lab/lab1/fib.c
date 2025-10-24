#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "utils.h"

#define F_30 832040LL
#define F_40 102334155LL
#define F_50 12586269025LL
#define F_60 1548008755920LL

static int N;
static int CUTOFF;

#define SEPARATOR "------------------------------------\n"

// Parse command line arguments to set solver parameters
void parse_arguments(int argc, char *argv[]);

// Fibonacci Golden Model - DO NOT CHANGE!
unsigned long long fibonacci_g(unsigned long long n)
{
    if (n < 2)
        return n;
    return fibonacci_g(n - 2) + fibonacci_g(n - 1);
}

// Run the Fibonacci
unsigned long long fib(unsigned long long n)
{
    if (n < 2)
        return n;
    return fib(n - 2) + fib(n - 1);
}

int main(int argc, char *argv[])
{
    parse_arguments(argc, argv);

    printf(SEPARATOR);
    printf("Number:       %d\n", N);
    printf("Cutoff:       %d\n", CUTOFF);
    printf(SEPARATOR);

    // Run Jacobi solver
    start_timer();
    unsigned long long f_n = fib(N);
    stop_timer();

    // Check error of final solution
    unsigned long long g_n;
    if (N == 30)
        g_n = F_30;
    else if (N == 40)
        g_n = F_40;
    else if (N == 50)
        g_n = F_50;
    else if (N == 60)
        g_n = F_60;
    else
        g_n = fibonacci_g(N);

    unsigned long long err = f_n - g_n;

    printf(SEPARATOR);
    printf("F(%d) = %llu\n", N, f_n);
    printf("Error = %llu\n", err);
    printf("Runtime = %lf ms\n", elapsed_ns() / 1E6);
    printf(SEPARATOR);

    return 0;
}

int parse_int(const char *str)
{
    char *next;
    int value = strtoul(str, &next, 10);
    return strlen(next) ? -1 : value;
}

double parse_double(const char *str)
{
    char *next;
    double value = strtod(str, &next);
    return strlen(next) ? -1 : value;
}

void parse_arguments(int argc, char *argv[])
{
    // Set default values
    N = 30;
    CUTOFF = 20;

    for (int i = 1; i < argc; i++)
    {
        if (!strcmp(argv[i], "--number") || !strcmp(argv[i], "-n"))
        {
            if (++i >= argc || (N = parse_int(argv[i])) < 0)
            {
                printf("Invalid matrix order\n");
                exit(1);
            }
        }
        else if (!strcmp(argv[i], "--cutoff") || !strcmp(argv[i], "-c"))
        {
            if (++i >= argc || (CUTOFF = parse_int(argv[i])) < 0)
            {
                printf("Invalid seed\n");
                exit(1);
            }
        }
        else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h"))
        {
            printf("\n");
            printf("Usage: ./jacobi [OPTIONS]\n\n");
            printf("Options:\n");
            printf("  -h  --help               Print this message\n");
            printf("  -c  --cutoff       C     Set task cutoff\n");
            printf("  -n  --number       N     Set the Fibonacci number\n");
            printf("\n");
            exit(0);
        }
        else
        {
            printf("Unrecognized argument '%s' (try '--help')\n", argv[i]);
            exit(1);
        }
    }
}