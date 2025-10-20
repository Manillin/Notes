/**
 * @file exercise3.c
 * @author Alessandro Capotondi
 * @brief Exercise 3
 * 
 */

#include <stdio.h>
#include <omp.h>

#include "utils.h"

/**
 * @brief EX 3 - Data parallelism: unbalanced parallel loop ** 4 THREADS **
 * 
 * a) Parallelize loop w/static scheduling
 * b) Parallelize loop w/dynamic scheduling, for chunks of 32, 16, 8, 4 ,1 (128 iter)
 * 
 * @return void
 */
void exercise()
{
    for (int i = 0; i < 128; i++)
    {
        DEBUG_PRINT("%hu: I am executing iteration %hu!\n", omp_get_thread_num(), i);
        work((i >> 2) * 1 << 20);
    }
}


