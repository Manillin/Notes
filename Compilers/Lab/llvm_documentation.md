# Key Points per LLVM

## Scrittura di un nuovo passo:

Per scrivere un nuovo llvm pass bisogna seguire i seguenti passaggi:

1. Creare una classe `NewPass` che implementi il passo
2. ...
3. ...

</br>

**Reminder**: I passi prendono in ingresso codice IR, per trasformare del codice in C in codice IR usiamo il seguente comando:

```bash
clang -O2 -emit-llvm -S -c TEST/file.c -o TEST/file.ll
```

Maggiori dettagli seguono sotto:

### 1. Creazione della classe

Ogni passo da aggiungere deve essere strutturato sotto forma di Classe C++ per scelte progettuali e architetturali di LLVM.  
Ogni classe che rappresenta un llvm pass, **dovrà** ereditare dalla classe `PassInfoMixin <NomePassaggio>` e implementare la funzione `run`.

La classe `PassInfoMixin<>` fornisce una struttura di base e funzionalità comuni per semplificare l'implementazione di nuovi passaggi.
Riduce la neccessità di scrivere codice boilerplate e mette a disposizione il meccanismo (( forse il metodo run() )) per registrare il passaggio nel sistema dei passaggi llvm.

```c++
class NewPass: public PassInfoMixin<NewPass>{
    PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
}
```
