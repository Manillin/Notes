# Primo Laboratorio: Scrivere un passo LLVM

# Test Pass

Aggiungiamo un passo alla toolchain che stampi tutte le funzioni locali al file analizzato!

Abbiamo due file sorgente scritti in C (serve trasformarli in IR), iniziamo dal file `Loop.c`

```bash
clang -O2 -emit-llvm -S -c TEST/Loop.c -o TEST/Loop.ll
```

## Scrittura del passo:

1. Avere il codice IR a disposzione
2. Preparare lo scheletro (la build) per un nuovo passo

## 2. Preparare lo scheletro

Creiamo un file e inseriamo il codice relativo al passo che vogliamo inserire in:

```bash
llvm/lib/Transforms/Utils/TestPass.cpp
```

Una volta creato aggiungiamo il nome di tale passo al file `CmakeLists.txt`:

```bash
llvm/lib/Transforms/utils/CmakeLists.txt
```

Ora che la build è settata per un nuovo passo non ci resta che scrivere il codice relativo al passo!

## Scrittura del codice per il passo:

Ricordiamo che il passo si compone di un file .cpp (creato nello step precedente) e dal corrispettivo file header .h

Il file header dovrà contenere il seguente codice boilerplate:

```c++
#ifndef LLVM_TRANSFORMS_TESTPASS_H
#define LLVM_TRANSFORMS_TESTPASS_H

#include "llvm/IR/PassManager.h"
namespace llvm{
class TestPass: public PassInfoMixin<TestPass>{
    public:
        PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};
} // namespace llvm

#endif // LLVM_TRANSFORMS_TESTPASS_H
```

Questo file header crea la classe per il passo e dichiara il metodo `run()`, ereditato da `PassInfoMixin< - >` che setta altro boilerplate neccessario per il funzionamento del nostro passo. Questo passo è **obbligatorio**, ogni passo deve esistere come classe ed ereditare dal PassInfoMixin.

Ora possiamo scrivere il codice vero e proprio nel file che abbiamo creato in precedenza, e includiamo il file header e il namespace definiti nel precedente step:

```c++
// llvm/lib/Transforms/Utils/TestPass.cpp

#include "llvm/Transforms/Utils/TestPass.h"
using namespace llvm;

//definiamo run()
PreservedAnalyses Testpass::run(Function &F, FunctionAnalysisManager &AM) {
    errs()<< F.getName() << "\n";
    return PreservedAnalyses::all()
}
```
