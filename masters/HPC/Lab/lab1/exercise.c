/**
 * @file exercise2.c
 * @author Alessandro Capotondi
 * @brief Exercise 2
 *
 */

#include <stdio.h>
#include <omp.h>

#include "utils.h"

/**
 * @brief EX 2 - Data parallelism: parallel loop ** 1 to 16 THREADS **
 *
 * a) Parallelize loop w/static scheduling
 * b) Parallelize loop w/dynamic scheduling for chunks = NITER/NTHR (same as static) and 1 (finest granularity)
 * c) Same as 2a + 2b, with 1<<20 loop iterations and work(10)
 *
 * @return void
 */
void exercise()
{
#pragma omp parallel for schedule(static) num_threads(NTHREADS)
    for (int i = 0; i < 1 << 10; i++)
    {
        DEBUG_PRINT("%hu: I am executing iteration %hu!\n", omp_get_thread_num(), i);
        work(1 << 10);
        DEBUG_PRINT("%hu: Done with iteration %hu!\n", omp_get_thread_num(), i);
    }
}