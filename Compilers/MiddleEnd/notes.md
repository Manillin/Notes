## Esempio di dead code

```c++
main {
    int a = 4;
    int b = 2; i
    nt c = 1;
    int d = a + b;
    std::cout<<d;
}
```

Notiamo che `int c = 1` non viene mai utilizzato e non altera lo stato del sistema senza avere side effects, di conseguenza possiamo cancellarla.

Definiamo come _dead code_ le istruzioni prive di **side effects** che definiscono una variabile che non viene mai utilizzata.

## Algoritmo per la DCE

1. Itero su tutte le istruzioni e aggiungo tutti i RHS (operandi) delle istruzioni a un metadato array

2. Per ogni istruzioni del BB controllo i LHS (destinazione dell'istruzione) e se tale variabile non è presente nell'array allora rimuovo l'istruzione.

es:
`int a = x + y ` $\rightarrow$ `int a`: LHS(destinazione); `x, y` RHS(operandi).

instruction -> dest + operandi , single line instruction
instruction.args -> operandi
instruction.dest -> destinazione

ridondanza -> passare lo stesso valore attraverso più variabili

# Code motion

### Condizioni di code motion:

- correttezza: lo spostamento del codice non altera la semantica del programma
- performance: l'esecuzione del codice non rallenta

_nota:_ Queste condizioni sono valide per tutte le trasformazioni

- **Idea di base**: l'istruzione candidata per la code motion definisce la varia ile nel suo LHS una volta e per tutte nel loop.

bisogna sempre verificare la condizione di dominanaza delle uscite, altrimenti si rischia di alterare la semantica del programma es: iteratore che potrebbe avere un valore superiore alla variabile di controllo non si entra nel ciclo e quindi la definizione interna non viene eseguita -> spostarla fuori rompererebbe la semantica.

-> Dobbiamo determinare il dominance tree

# Lezione 13 maggio:

Se la variabile interna al ciclo possiamo spostarla nel caso si tratti di dead code anche se tale blocco non domina tutte le uscite del loop, questo è l'unico caso però in quanto se non fosse codice dead, potrebbe capitare di non entrare mai nel ciclo e quindi di non eseguire tale istruzione.

in llvm un istruzione è dead, se dal punto di interessse all'uscita non ci sono più usi.  
-> controlliamo quindi gli usi (per verificare se è deadcode) e verifichiamo in che blocco si trovano. se gli usi sono confinati al loop allora è dead code una volta usciti.

# Lezione 17 Maggio:

### Ottimizzazioni sulla memoria

Le cache servono a nascondere il dislivello di latenza tra dram e performance cpu.

dram considerata lenta in quanto si basa sul principio della conservazione di carica di un condensatore (1 -> carica | 0 -> no carica). La carica è volatile e i capacitori nel tempo perdono la carica, per questo la dram ha bisogno di un refresh periodico per ripristinare i dati.

Una tecnologia molto diversa da quella linare della cpu, per questo c'è tale discrepanza di performance.

Metriche di performance di una memoria:

- latenza -> tempo che ci vuole per portare su o giu dati
- banda -> quantità di dati che riesco a 'portare su' in un unica transizione

i bit in dram sono organizzati come array, e ci sono le dram layer, attivare una riga comporta un costo temporale e per ammortizzare questo costo una tecnica è quella di portare su tutta la row.

### SRAM

ha la caratteristica di essere fatta di trnsistor, cioè di un circuito logico, piu simile alla cpu. puo essere integrato dentro un chip -> si usa per le gerarchie di cache.

È estremamente veloce, comparabile con la velocità di operazione di una cpu (dipende dalla distanza della sram dall'unità Load Store unit -> per questo gerarchie di cache in base alla distanza).

Perche non fare tuttu con sram -> per i due pattern ricorrenti di accesso in memoria (array,...)

**Cache misses:**

- Compulsory: cold cache miss, non esistono dati nella cache
- Confilt: Quando uso cache direct mapped (un posto univoco per dato) e capita quando due indirizzi in DRAM sono mappati sulla stessa cache line

Ottimizzazioni della memoria:

Tipi di dato da considerare: scalari (variabili semplici), Strutture e puntatori (struct e ptr), Array.

Scalari: variabili locali, globali e argomenti di procedure.

```c++
int x;
double y; // variabili globali
foo(int a){
    int i; // variabili locali
    ...
    x = a * i;  // la 'a' è procedure argument
    ...
}
```

quando voglio usare variabili e ho esaurito i registri uso lo stack! (non serve solo per la calling procedure delle funzioni).
