CC = gcc
CFLAGS = -Wall -D_GNU_SOURCE

all:	level02 level02-nodebug

level02:	level2.c
	$(CC) $(CFLAGS) -o $@ $<
level02-nodebug:	level2-nodebug.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -rf level02{,.o} level02-nodebug{,.o}
