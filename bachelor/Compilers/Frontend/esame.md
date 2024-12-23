## domande esame orale 

- scrivere un espressione regolare per le costranti in c 
- cosa si intende per analisi semantica 
    - quale aspetto piu importante nell'analisi semantica (contesto di programmazione)  
    Forzare la coerenza dei tipi (cosa che esula dall'analisi sintattica)


### secondo orale 

- automa deterministico per le costanti 
- cosa sono gli i item (parser LR) - occhio a confonderlo con le collezioni
    - e cosa sono gli insieme di item ? perchè sono importanti 



### terzo orale 

- generare stringhe del tipo a^nb^bc^2n con una gramamtica ( = deve essere context free)
- ruolo del lexer, che ruoli svolge
- var expre -> risoluzione di una variabile 


### quarto orale

- floor di -1.5 
- produzione su idsec
- data una grammatica fatta dal prof dire se la grammatica è LL(1)
- automa non deterministico - subset construction per passare da AFND a AFD


### quinto orale 



sqrt3: callsqrt.o sqrt3.o
	clang++ -o sqrt3 callsqrt.o sqrt3.o

sqrt3.o:	sqrt3.k
	../kcomp sqrt3.k 2> sqrt3.ll
	./tobinary sqrt3.ll