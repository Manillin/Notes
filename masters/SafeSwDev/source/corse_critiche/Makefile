CC = gcc
CFLAGS = -Wall -pthread

WAIT_SRC = wait.c
WAIT_OBJ = wait.o
WAIT_BIN = wait
PTHREAD_JOIN_SRC = pthread_join.c
PTHREAD_JOIN_OBJ = pthread_join.o
PTHREAD_JOIN_BIN = pthread_join
PTHREAD_MUTEX_SRC = pthread_mutex.c
PTHREAD_MUTEX_OBJ = pthread_mutex.o
PTHREAD_MUTEX_BIN = pthread_mutex
PRODUCER_CONSUMER_SEM_SRC = producer_consumer_sem.c
PRODUCER_CONSUMER_SEM_OBJ = producer_consumer_sem.o
PRODUCER_CONSUMER_SEM_BIN = producer_consumer_sem

% : %.o
	@$(CC) $(CFLAGS) -o $@ $<
%.o : %.c
	@$(CC) $(CFLAGS) -c $<

build: $(WAIT_BIN) \
	$(PTHREAD_JOIN_BIN) $(PTHREAD_MUTEX_BIN) \
	$(PRODUCER_CONSUMER_SEM_BIN)

.PHONY:	clean

clean:
	rm -rf $(WAIT_BIN) $(WAIT_OBJ) $(WAIT_BIN) \
		$(PTHREAD_JOIN_BIN) $(PTHREAD_JOIN_OBJ) $(PTHREAD_MUTEX_BIN) $(PTHREAD_MUTEX_OBJ) \
		$(PRODUCER_CONSUMER_SEM_BIN) $(PRODUCER_CONSUMER_SEM_OBJ)
