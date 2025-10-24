/**
 * @file exercise7.c
 * @author Alessandro Capotondi
 * @date 27 Mar 2020
 * @brief Exercise 8
 *
 * Pi calculation
 * @see https://dolly.fim.unimore.it/2019/course/view.php?id=152
 */

#include <stdio.h>
#include <omp.h>

#include "utils.h"

/**
 * @brief EX 8- Pi Calculation
 *
 * This program computes pi as
 * \pi = 4 arctan(1)
 *     = 4 \int _0 ^1 \frac{1} {1 + x^2} dx
 *
 * @return void
 */
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "utils.h"

#if !defined(ITERS)
#define ITERS (4)
#endif

#define NSTEPS 134217728

void exercise()
{
    long i;
    double dx = 1.0 / NSTEPS;
    double pi = 0.0;

    double start_time = omp_get_wtime();

// #pragma omp parallel for share(pi) -> kills performance
#pragma omp parallel for reduction(+ : pi)
    for (i = 0; i < NSTEPS; i++)
    {
        double x = (i + 0.5) * dx;
        pi += 1.0 / (1.0 + x * x);
    }
    pi *= 4.0 * dx;

    double run_time = omp_get_wtime() - start_time;
    double ref_pi = 4.0 * atan(1.0);
    printf("pi with %d steps is %.10f in %.6f seconds (error=%e)\n",
           NSTEPS, pi, run_time, fabs(ref_pi - pi));
}

int main(int argc, char **argv)
{
    for (int i = 0; i < ITERS; i++)
    {
        printf("\n\n");
        printf("============================\n");
        printf("Test - Iteration %d...\n", i);
        printf("============================\n");
        start_stats();
        exercise();
        collect_stats();
    }

    printf("\n\n");
    printf("============================\n");
    printf("Statistics\n");
    printf("============================\n");
    print_stats();
    return 0;
}