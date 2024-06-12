
# Intro Matlab

Programma che facilita la gestione e la programmazione per calcolo scientifico.  
Matlab ha un gestore della memoria proprio, chiamato `Workspace` quindi non è responsabilità del programmatore.  
Tale workspace è visualizzabile (`whos` o dalle opzioni della gui)


### Nozioni:

- il `;` serve per eliminare lo stout di ogni istruzione, se omesso vedremo l'output di ogni riga.
- **caratteristica** di matblab $\rightarrow$ ogni variabile è una matrice, es: gli scalari sono visti come matrice 1x1.  
- `ans` è la variabile interna dove vengono memorizzate le operazioni non assegnate a nessuna variabile esplicita.  
- `clc` equivalente di fare ctrl+l sul terminale
- `clear all` cancella tutto il contenuto del workspace

### Operatori Relazionali:

- `~` $\rightarrow$ NOT
- `&` $\rightarrow$ AND
- `|` $\rightarrow$ OR
- `xor` $\rightarrow$ XOR

- `TRUE` $\rightarrow$ 1
- `FALSE` $\rightarrow$ 0 

### Costanti predefinte 

- `pi `$\rightarrow$ $\pi$
- `i,j` $\rightarrow$ $i,j$
- `Inf`$\rightarrow$ $\infty$
- `eps` $\rightarrow$ $NaN$
- `eps` $\rightarrow$ $\varepsilon$ (precisione di macchina)
- `realmax` $\rightarrow$ massimo reale rappresentabile 
- `realmin` $\rightarrow$ minimo reale rappresentabile




### Creazione di matrici:

Le matrici vere e proprie (di dimensioni arbitraria) si generano con le parentesi quadre $\rightarrow$ `[ ]`

- Vettore riga: `[1 2 3 4]` o mettendo le virgole tra ogni numero
- Vettore colonna: `[1;2;3;4;]` o andando a capo dopo ogni numero
- Matrice $n\space x \space n$: `[1, 2, ...; 5, 6, ...; ...]`

La creazione di matrice fallisce se si prova a creare una matrice con $i$ righe e $j$ colonne se $i\neq j$ .  
