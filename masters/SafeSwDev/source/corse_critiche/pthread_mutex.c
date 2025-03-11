/*
 * pthread_mutex.c - accesso in mutua esclusione
 *		     ad un'area di memoria condivisa
 *
 * Si illustra il funzionamento delle funzioni di
 * libreria pthread_mutex_{create,destroy,lock,unlock}(),
 * che consentono di impostare l'accesso in mutua
 * esclusione ad un'area di memoria condivisa.
 */

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#define NTHREADS 2

/* Variabili globali (condivise tra i thread) */
pthread_mutex_t lock;
int variabile_contesa = 0;

void *do_work (void *thread_num)
{
	int *my_num = (int *)thread_num;
	printf("[%d] inizio funzione specifica\n", *my_num);

	printf("[%d] il contenuto della variabile è %d\n",
		*my_num, variabile_contesa);

	printf("[%d] acquisizione lock\n", *my_num);
	if (pthread_mutex_lock(&lock) != 0) {
		perror("pthread_mutex_lock()");
		pthread_exit(NULL);
	}

	variabile_contesa++;

	printf("[%d] rilascio lock\n", *my_num);
	if (pthread_mutex_unlock(&lock) != 0) {
		perror("pthread_mutex_unlock()");
		pthread_exit(NULL);
	}

	printf("[%d] dopo la modifica, il contenuto della variabile è %d\n",
		*my_num, variabile_contesa);

	pthread_exit(NULL);
}

int main (void)
{
	int i;
	pthread_t threads[NTHREADS];
	/*
	 * Utilizziamo la variabile args per mantenere il numero
	 * del thread creato e differenziare le tracce di
	 * esecuzione. Il thread creato per primo riceverà, come
	 * argomento, il contenuto di args[0], quindi il numero "0",
	 * appunto. Similmente, il thread creato per secondo
	 * riceverà, come argomento, il contenuto di args[1], quindi
	 * il numero "1".
	 */
	int args[NTHREADS] = { 0, 1 };

	/* Inizializzazione del mutex */
	pthread_mutex_init(&lock, NULL);

	/* Creazione dei thread, che eseguono la medesima funzione */
	for (i = 0; i < NTHREADS; i++) {
		if (pthread_create(&threads[i], NULL, &do_work, &args[i]) != 0) {
			perror("pthread_create()");
			exit(EXIT_FAILURE);
		}
	}

	/* La traccia di esecuzione principale attende i due thread */
	for (i = 0; i < NTHREADS; i++) {
		if (pthread_join(threads[i], NULL) != 0) {
			perror("pthread_join()");
			exit(EXIT_FAILURE);
		}
	}

	if (pthread_mutex_destroy(&lock) != 0) {
		pthread_mutex_unlock(&lock);
		perror("pthread_mutex_destroy()");
		exit(EXIT_FAILURE);
	}

	exit(EXIT_SUCCESS);
}
