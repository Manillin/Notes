/*
 * producer_consumer_sem.c - soluzione del problema produttore-consumatore
 *                             mediante mutex
 *
 * Un processo padre definisce NUM_PRODUCERS thread produttori
 * e NUM_CONSUMERS thread consumatori.
 * - Un thread produttore genera un numero casuale e lo
 *   inserisce in mutua esclusione in un array circolare.
 * - Un thread consumatore estrae un numero dall'array
 *   in mutua esclusione e lo consuma, stampandolo.
 */
#include <pthread.h>	/* pthread_* */
#include <semaphore.h>	/* sem_* */
#include <stdio.h>	/* printf() */
#include <stdlib.h>	/* exit(), srand(), rand() */
#include <time.h>	/* time */

#define NUM_ELEMENTS 	5
#define NUM_PRODUCERS	10
#define NUM_CONSUMERS	10

/* l'array condiviso */
int elements[NUM_ELEMENTS];

/* puntatore alla prima cella libera */
int in = 0;

/* puntatore alla prima cella occupata */
int out = 0;

/*
 * Numero di elementi nell'array
 * counter = 0: array vuoto
 * counter = NUM_ELEMENTS: array pieno
 */
int counter = 0;

/* mutex proteggente le variabili condivise */
pthread_mutex_t lock;

/* 
 * semafori di sincronizzazione
 * - full: numero di elementi pieni nell'array
 * - empty: numero di elementi vuoti nell'array
 */
sem_t full, empty;

/* produttore */
void *producer(void *arg) {

	int thread_num = *((int *)arg);
	int elem;

	while(1) {

		/* produzione di un elemento */
		elem = rand();
		printf("P%d: produced %d\n", thread_num, elem);

		/*
		 * Attesa dello svuotamento di almeno un
		 * elemento dell'array. A differenza di
		 * prima, ora l'attesa è passiva.
		 */
		if (sem_wait(&empty) != 0) {
			perror("sem_wait()");
			exit(EXIT_FAILURE);
		}
		/*
		 * Richiesta accesso in mutua esclusione
		 * alle strutture dati condivise
		 * dell'elemento nell'array
		 */
		if (pthread_mutex_lock(&lock) != 0) {
			perror("pthread_mutex_lock()");
			exit(EXIT_FAILURE);
		}

		/* Inserimento dell'elemento nell'array */
		elements[in] = elem;

		/* Aggiornamento dei contatori */
		in = (in + 1) % NUM_ELEMENTS;
		counter++;

		/* Rilascio del lock */
		if (pthread_mutex_unlock(&lock) != 0) {
			perror("pthread_mutex_unlock()");
			pthread_exit(NULL);
		}

		/*
		 * Segnalazione del riempimento di
		 * un elemento dell'array. Ciò
		 * risveglia un consumer.
		 */
		if (sem_post(&full) != 0) {
			perror("sem_post()");
			exit(EXIT_FAILURE);
		}
	}
}

/* consumatore */
void *consumer(void *arg) {

	int thread_num = *((int *)arg);
	int elem;

	while(1) {

		/*
		 * Attesa del riempimento di almeno un
		 * elemento dell'array. A differenza di
		 * prima, ora l'attesa è passiva.
		 */
		if (sem_wait(&full) != 0) {
			perror("sem_wait()");
			exit(EXIT_FAILURE);
		}
		/*
		 * Richiesta accesso in mutua esclusione
		 * alle strutture dati condivise
		 * dell'elemento nell'array
		 */
		if (pthread_mutex_lock(&lock) != 0) {
			perror("pthread_mutex_lock()");
			pthread_exit(NULL);
		}

		/* Recupero dell'elemento dall'array */
		elem = elements[out];

		/* Aggiornamento dei contatori */
		out = (out + 1) % NUM_ELEMENTS;
		counter--;

		/* Rilascio del lock */
		if (pthread_mutex_unlock(&lock) != 0) {
			perror("pthread_mutex_unlock()");
			pthread_exit(NULL);
		}

		/*
		 * Segnalazione dello svuotamento di
		 * un elemento dell'array. Ciò
		 * risveglia un producer.
		 */
		if (sem_post(&empty) != 0) {
			perror("sem_post()");
			exit(EXIT_FAILURE);
		}

		/* consumo dell'elemento */
		printf("C%d: consumed %d\n", thread_num, elem);
	}
}

int main(void) {

	pthread_t producers[NUM_PRODUCERS];
	pthread_t consumers[NUM_CONSUMERS];
	int i;

	/* inizializzazione del generatore di numeri casuali */
	srand(time(0));

	/* Inizializzazione del mutex */
	pthread_mutex_init(&lock, NULL);

	/*
	 * Inizializzazione dei semafori
	 * - empty = NUM_ELEMENTS 
	 * - full = 0
	 */
	if (sem_init(&empty, 0, NUM_ELEMENTS) != 0) {
		perror("sem_init()");
		exit(EXIT_FAILURE);
	}
	if (sem_init(&full, 0, 0) != 0) {
		perror("sem_init()");
		exit(EXIT_FAILURE);
	}

	/* Creazione dei thread produttori */
	for (i = 0; i < NUM_PRODUCERS; i++) {
		if (pthread_create(&producers[i], NULL, &producer, (void *)&i) != 0) {
			perror("pthread_create()");
			exit(EXIT_FAILURE);
		}
	}

	/* Creazione dei thread consumatori */
	for (i = 0; i < NUM_CONSUMERS; i++) {
		if (pthread_create(&consumers[i], NULL, &consumer, (void *)&i) != 0) {
			perror("pthread_create()");
			exit(EXIT_FAILURE);
		}
	}

	/* Join dei thread */
	for (i = 0; i < NUM_PRODUCERS; i++) {
		if (pthread_join(producers[i], (void **)0) != 0) {
			perror("pthread_join()");
			exit(EXIT_FAILURE);
		}
	}
	for (i = 0; i < NUM_CONSUMERS; i++) {
		if (pthread_join(consumers[i], (void **)0) != 0) {
			perror("pthread_join()");
			exit(EXIT_FAILURE);
		}
	}

	pthread_exit(NULL);
}
