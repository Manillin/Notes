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
