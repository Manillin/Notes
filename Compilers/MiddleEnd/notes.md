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
