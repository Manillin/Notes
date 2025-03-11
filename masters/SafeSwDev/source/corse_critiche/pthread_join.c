/*
 * pthread_join.c - attesa di un thread
 *
 * Si illustra il funzionamento della funzione
 * di libreria pthread_join(), che consente di
 * attendere il termine dell'esecuzione di un
 * thread.
 * Un processo padre crea due thread che stampano
 * una stringa (passata come argomento) e ritornano
 * la lunghezza della stessa. Successivamente, il
 * processo padre attende i due thread e stampa il
 * loro valore di ritorno.
 */

#include <stdio.h>	/* perror() */
#include <stdlib.h>	/* exit(), malloc(), free() */
#include <pthread.h>	/* pthread_create(), pthread_join() */
#include <unistd.h>	/* sleep() */
#include <string.h>	/* strlen() */

void *print_message(void *arg) {

	char *s = (char *)arg;

	printf("%s", s);

	return (void *)strlen(s);
}

int main(void) {

	pthread_t t1, t2;
	void *res_t1, *res_t2;
	int s;

	/* Il processo crea due thread. */
	s = pthread_create(&t1, NULL, print_message, (void *)"Hello from thread 1!\n"); 
	if (s != 0) {
		perror("pthread_create(&t1, ...)");
		exit(EXIT_FAILURE);
	}
	s = pthread_create(&t2, NULL, print_message, (void *)"Ciao da thread 2!\n");
	if (s != 0) {
		perror("pthread_create(&t2, ...)");
		exit(EXIT_FAILURE);
	}

	/* Il processo attende i due thread. */
	s = pthread_join(t1, &res_t1);
	if (s != 0) {
		perror("pthread_join(t1, ...)");
		exit(EXIT_FAILURE);
	}
	s = pthread_join(t2, &res_t2);
	if (s != 0) {
		perror("pthread_join(t2, ...)");
		exit(EXIT_FAILURE);
	}

	/* Qui i due thread sono già terminati. */
	printf("Valore di ritorno di thread 1: %ld\n", (long) res_t1);
	printf("Valore di ritorno di thread 2: %ld\n", (long) res_t2);

	exit(EXIT_SUCCESS);
}
