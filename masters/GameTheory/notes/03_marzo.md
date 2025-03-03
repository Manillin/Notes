# lezione del 3 marzo 

giochi sequenziali - i giocatori fanno turni per fare le loro mosse, e quando fanno le loro mosse sanno cosa hanno fatto prima gli altri giocatori. (tipo scacchi)  

modo di ragionare: un giocatore deve cercare di prevedere il futuro, fare la propria scelta in base a cosa potrebbe fare l'altro giocatore.  

**Game Trees**  
Descriviamo il gioco sequenziale come un albero particolare.  
L'albero rappresenta tutta la dinamica del gioco, le mosse possibili sono modellate dai rami dell'albero.  
I numeri sui nodi terminali (foglie) rappresentano la fine del gioco e hanno i payoffs dei vari giocatori (stabilito un ordine di lettura).  

**strategia** - definire un insieme di mosse per avere un piano (come libertto di istruzioni) su ogni nodo in cui mi trovo.  

**mossa** - singola decisione che si prende su un nodo.  


se abbiamo un gioco e i giocatori sono razionali ci aspettiamo un rollback induction.  

Tutti i giochi sequenziali possono essere risolti con la rollback induction