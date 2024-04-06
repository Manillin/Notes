# Lab 2: Introduzione ai passi di trasformazione

Ricordiamo il fatto che il _middle end_ sia organizzato come una sequenza di passi, in particolare passi di analisi e di trasformazione.  
In questo laboratorio ci concentreremo nei passi di **trasformazione**.

## IR llvm

La IR llvm segue canonicamente la forma **SSA** (single static assigment) che prevede che una variabile non possa essere definita più di una volta.

# Relazione User - Use - Value in llvm:

## Value $\rightarrow$ valore

La classe `Value` è fondamentale per la struttura llvm, un oggetto di questo tipo rappresenta un valore o il risultato di un operazione, infatti avremo che **variabili, costanti e istruzioni** sono tutte istanze di Value.

## User $\rightarrow$ istruzione

La classe `User` è chiave per llvm, essa eredita dalla classe `Value` e rappresenta un **istruzione che utilizza uno o più Value** come _operandi_.
Serve per rappresentare un entita che dipende da altri valori o istruzioni per funzionare correttamente.

> Un oggetto User rappresneta un istruzione che utilizzano altri Value come operandi.

Questa classe è responsabile di gestire gli effetti collaterali che un cambiamento nel codice potrebbe causare $\rightarrow$ se un istruzione viene modificata o cancellata, l'istruzione `User` che ne dipendeva verrà gestita per mantenere la corenza nel codice.

## Use $\rightarrow$ instr:value

La classe `Use` rappresenta l'associazione tra un'istruzione `User` e un valore `Value` suo operando.  
Un oggetto Use è usato quindi per tracciare e gestire le dipendenze tra un'istruzione e i valori che essa utilizza.  
Ogni istanza di Use rappresenat un collgamento tra un'istruzione User e uno dei sue operandi Value.

## Interazione tra le classi:

La relazione User-Use-Value in llvm mappa le dipendenze tra le istruzioni ed è **fondamentale** per gestire correttamente le **trasformazioni** di codice a seguito di ottimizzazioni al fine di assicurarsi che il progrmam risultante rimanga coerente e corretto.

### Dipendenza tra istruzioni:

Le istruzioni llvm sono costruite a partire della forma SSA ed in modo che ogni istruzione dipenda da altre per funzionare correttamente, questo tipo di dipendenza è rappresentato dalla classe `User`.

Es:

```
x = y + t
```

In questo esempio:

- `y`, `t` ed `x` sono istanze di `Value` in quanto rappresentano valori
- `x` è anche istanza di `User` in quanto rappresenta un istruzione che usa altri Value
- in `x` avremo due istanze `Use`, una x:y e una x:t
