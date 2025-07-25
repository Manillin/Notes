CC = gcc
CFLAGS = -Wall -O0 -g -m32 -static

all:	shellcode.o shellcode_proof

shellcode_proof:	shellcode_proof.c
		$(CC) $(CFLAGS) -o $@ $<

shellcode.o:	shellcode.s
		$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -rf shellcode_proof{,.o} shellcode.o
