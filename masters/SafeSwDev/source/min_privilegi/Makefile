CC = gcc
CFLAGS = -Wall -D_GNU_SOURCE

all:	flag01 flag01-drop flag01-noperm flag01-env

flag01:	level1.c
	$(CC) $(CFLAGS) -o $@ $<
flag01-drop:	level1-drop.c
	$(CC) $(CFLAGS) -o $@ $<
flag01-noperm:	level1-noperm.c
	$(CC) $(CFLAGS) -o $@ $<
flag01-env:	level1-env.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -rf flag01{,.o} flag01-drop{,.o} flag01-noperm{,.o} flag01-env{,.o}
