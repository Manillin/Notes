# Assignment 3: LICM

A partire dal codice della precedente esercitazione implementare un passo di LICM (loop invariant code motion).

Il codice di base e della precedente esercitazione:

```c++

#include "llvm/Transforms/Utils/LoopPasses.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include <llvm/IR/Constants.h>

using namespace llvm;

PreservedAnalyses LoopPasses::run(Loop &L, LoopAnalysisManager &LAM,
                                  LoopStandardAnalysisResults &LAR,
                                  LPMUpdater &LU) {

  outs() << "Starting loop programm: \n\n";

  if (!L.isLoopSimplifyForm()) {
    outs() << "\n il Loop non è in forma normale \n";
    return PreservedAnalyses::all();
  }

  outs() << "\n Il loop è in forma Normale si può continuare "
            "nell'ottimizazione... \n";
  BasicBlock *head = L.getHeader();

  // recuperiamo l'handle alla funzione che contiene il Loop
  Function *F = head->getParent();

  // stampo il CFG
  outs() << "-----CFG------ \n";
  int cont = 0;
  for (auto iter = F->begin(); iter != F->end(); ++iter) {
    outs() << "Basic Block(" << cont++ << ") : " << "\n";
    BasicBlock &BB = *iter;
    outs() << BB << "\n";
  }

  outs() << " ---- fine ----- ";

  // Stampo il Loop
  outs() << "\n\n  ---- IL LOOP ------  \n";
  cont = 0;

  for (auto BI = L.block_begin(); BI != L.block_end(); ++BI) {

    outs() << "Basic Block(" << cont++ << ") : " << "\n";
    BasicBlock *BB = *BI;
    outs() << *BB << "\n";
    return PreservedAnalyses::all();
  }
}
```
