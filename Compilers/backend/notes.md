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
