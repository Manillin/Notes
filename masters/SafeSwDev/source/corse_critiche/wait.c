/*
 * wait.c - sincronizzazione con i processi figli 
 *
 * Si illustra il funzionamento della chiamata
 * di sistema wait(), che consente al processo
 * padre di attendere il termine dell'esecuzione
 * di un processo figlio.
 */

#include <stdio.h>	/* perror() */
#include <stdlib.h>	/* exit() */
#include <unistd.h>	/* fork(), getpid(), sleep() */
#include <sys/wait.h>   /* wait() */

#define	SLEEPINTVL	5

int main (void)
{
	pid_t pid, child_pid;

	/* Il padre genera un processo figlio */
	if ((pid = fork()) < 0) {
		perror("fork()");
		exit(EXIT_FAILURE);
	}

	if (pid > 0) {
		/* Processo padre: si attende l'uscita del figlio */
		printf("Processo padre: attendo il figlio.\n");

		/*
		 * In questi primi esempi non interessa indagare sui
		 * motivi del cambio di stato. Pertanto, si può
		 * passare NULL come argomento a wait().
		 */
		if ((child_pid = wait(NULL)) == -1) {

			perror("wait()");
			exit(EXIT_FAILURE);
		} else
			/* Si stampa il PID del processo figlio uscito */
			printf("Processo padre: il figlio con PID %d è uscito.\n", child_pid);
	} else if (pid == 0) {
		/* Processo figlio: dorme per alcuni secondi ed esce. */		
		printf("Processo figlio: ho PID %d e aspetto %d secondi.\n",
			getpid(), SLEEPINTVL);
		sleep(SLEEPINTVL);
		printf("Processo figlio: esco.\n");
	}

	exit(EXIT_SUCCESS);
}
