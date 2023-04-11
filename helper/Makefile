OPTS = -O2 -Wall -Wc++-compat -pedantic -Wextra
LIBS = -lrt -lpthread
CC = gcc

main:
	$(CC) -o shm_reader.out shm-reader.c $(LIBS) $(OPTS)

.PHONY: clean
clean:
	rm *.out