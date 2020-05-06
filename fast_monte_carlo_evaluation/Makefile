all:	podds podds-so

podds:	podds.c poker.c xorshift.c
	gcc -O3 -pthread -o podds podds.c poker.c xorshift.c

podds-so:  podds.c poker.c xorshift.c
	gcc -fPIC -shared -O3 -pthread -o podds.so podds.c poker.c xorshift.c

clean:
	rm podds podds-perftest
