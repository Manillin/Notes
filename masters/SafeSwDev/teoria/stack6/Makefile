CC = gcc
CFLAGS = -Wall -O0 -g -m32

all:	system

execl:	system.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -rf system{,.o}
