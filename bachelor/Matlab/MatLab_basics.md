
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

### Indicizzazione:
In MatLab l'indicizzazione dei vettore parte da 1 e non da zero
es:
```matlab
a = [1,2,3,4]
a(1) %stampa '1' e non '2'
a(3) %stampa '3' e non '4' 

M = [1,2,3
    4,5,6]

M(1,1) %stampa '1' e non '5'

```
Si possono accedere e modificare gli elementi accedendo algi indici con la notazione `( )`.  

### Comodities: 
L'operatore `:` serve per creare intervalli equispaziali, o con un passo specificato.  

```matlab
k = 1:5 %crea vettore: [1,2,3,4,5]

I = 2.5:10 %crea vettore: [2.5, 3.5, 4.5, ..., 8.5, 9.5]

k2 = 1:2:10 %crea vettore: [1, 3, 5, 7, 9] con passo 2 
```

Questo operatore è molto utile per creare sottomatrice, tipo splicing in python.  

```matlab
M = [1,2,6,8; 2,5,8,1]

C = M(2,1:3) % crea [2,5,8]

F = M(1:2, 1:3) % prende le righe 1:2 e colonne 1:3
F = M([1,2], [1,3]) % analogo a sopra 
```


- `tril()` $\rightarrow$ prende il triangolo inferiore di una matrice
- `triu()` $\rightarrow$ ritaglia il triangolo superiore
- `B = tril(A,n)` $\rightarrow$ il risultato è una matrice di ordine pari ad A ma di tipo triangolare inferiore; il parametro n indica da quale diagonale iniziare, se omesso parte da 0, ossia la diagonale principale.  


### Operazioni:
Le operazioni sono pensate per matrici, quindi ad esempio se proviamo la moltiplicazione verrà applicata quella riga per colonna, stessa cosa vale per l'elevamento a potenza di una matrice, sarà considerato come M x M, non l'elevazione di ogni suo singolo elemento alla potenza desiderata.  

Si usa la notazione **puntata** delle operazioni nel caso si volesse usare l'operazione default e non quella matematica

```matlab

A^2 % equivale a fare A*A come moltiplicazione righe per colonne 
A.^2 % equivale ad elevare al quadrato tutti gli elementi di A

```


### Operazioni e Funzioni utili per lavorare con le matrici:
- `eye(n)` $\rightarrow$ matrice identità nxn
- `zeros(m,n)` $\rightarrow$ matrice mxn con tutti elementi uguali a zero
- `ones(m,n)` $\rightarrow$ matrice mxn con tutti elementi uguali a uno
- `size(x)` $\rightarrow$ restituisce la dimensione della matrice (es: mxn)
- `length(v)` $\rightarrow$ restituisce il numero di componenti del vettore o la dimensione maggiore della matrice



Alcune operazioni su matrici sono semplificate, in particolare quelle che involgono l'uso di funzione prestabilite. 
Questa cosa mi consente di semplificare il codice ed evitare il dover applicare le funzioni che desidero attraverso cicli, che rallentano molto l'esecuzione.  

es:
```matlab
x = [1,2,3;
     5,6,7]

sin(x) % applica la funzione sin() a tutti gli elementi della matrice e la restituisce.  

```



### Aiuto inline
Se non ci ricordiamo come usare un comando possiamo dalla command window chiamare l'helper, baseterà `help <comando/funzione>` e verrà visualizzata una breve spiegazione di applicazione del comando.  
Se servono spiegazioni più approfondite si può usare `doc` al posto di help, che aprirà la pagina della documentazione relativa al comando richiesto. 



## Istruzioni Condizionali:


```matlab
if <espressione logica1>
    <istruzione1>

elseif <espressione logica2>
    <istruzione2>

else
    <istruzione3>

end

```

si usa una sintassi intuitiva per i cicli for (costosi a livello computazionale).  


### Scrittura di Script e Funzioni:

Ogni volta che si vuole fare uno script matlab inizializzalo con: `clear all`.  
Evita di mescolare le variabili con quelle esistenti già nel workspace. 


```matlab

function [a] = fibonacci_fun(n)

a = zeros(n,1);
a(2) = 1;
for k = 3:n
    a(k) = a(k-1) + a(k-2);
end

% il risultato della function verrà salvato in $a$, mnentre il parametro verra usato al posto della $n$

```


--- 


# Creazione di Grafici in Matlab

L'istruzione che permette di creare grafici è `plot(x,y)` 
 - apre una finestra grafica detta figure
 - disegna la spezzata che congiunge i punti $(x_i,y_i)$
 - opzione di tratteggio: `plot(x,y,':m')` $\rightarrow$ la `m` sta per magenta, ci sono altre opzioni

Ogni nuova istruzione `plot` sovrascrive il contenuto della precedente
- `hold on` permette di visualizzare più grafici in una stessa finestra 
- `figure` permette di aprire più finestre grafiche
- `subplot` divide un finestra in più aree grafiche 


Quando i valori da rappresentare hanno una grande varaibilità la scala lineare potrebbe non essere 
adeguata
- `loglog()` - scala logaritmica su asse x e y 
- `semilogy()` - scala logaritmica su asse y
- `semilogx()` - scala logaritmica su asse x 
 


--- 

## Grafici di funzioni di 2 variabili 

Occorre fornire le ascisse e le ordinate dei punti del grafico 


l'idea base è di discretizzare il dominio bidimensionale della funzione, quindi individuare 
un isieme di punti su cui valutare la funzione, e con un istruzione di grafica chiedere a matlab di 
collegare i valori che la funzione assume sui punti del dominio.  


**discretizzare il dominio**  

con due vettori si forniscono i valori della discretizzazione del dominio, ossia le ascisse e le 
ordinate dei punti del piano che discretizzano il dominio.  

x = [1 2 3]; y = [4 5 6 7]; (3 valori per le x e 4 valori per asse y)

Matlab permette di usare un comando ossia `[X,Y] = meshgrid(x,y)` che mi permettono di far valutare 
la funzione sui punti del dominio in maniera efficinete e senza dover usare cicli annidati.

tale comando porta le info contenute nei vettori in due matrici costituite da un numero di righe pari 
alla lunghezza di `y` e un numero di colonne pari alla lunghezza di `x`.  

```matlab
X = 
    1 2 3
    1 2 3
    1 2 3
    1 2 3

Y = 
    4 4 4 
    5 5 5
    6 6 6 
    7 7 7
```

A questo punto le coordinate dei punti sulla griglia sono esprimibili usando le due matrici prodotte
es: (3,2) trovo (ascissa:2 , ordinata:6)  


Esistono varie opzioni per cambiare la colorbar delle figure:
- summer, autumn, winter, pink $\rightarrow$ altre opzioni 

`countour(X,Y,Z)` $\rightarrow$ Crea curve di livello, servono le informazioni sui punti della griglia (X,Y) e i valori che la funzione assume su tali punti (Y)  

`[c,h] = countour(X,Y,Z); clabel(c,h);` $\rightarrow$ permette di associare i valori delle costanti che la funzione assume sulle curve di livello.  


