# Primo Assignment:

> Implementare tre passi LLVM (dentro lo stesso passo LocalOpts già scritto durante il LAB 2) che realizzano le seguenti ottimizzazioni locali:

1. Algebric Identity

   $$
   x + 0 = 0 +x \space \rightarrow \space x
   $$

   $$
   x * 1 = 1 * x \space \rightarrow \space x
   $$

2. Strength Reduction

   $$
   15 * x = x * 15 \space \rightarrow \space(x<<4)-x
   $$

   $$
   y = x \div 8 \space \rightarrow \space y = x >> 3
   $$

3. Multi-Instruction Optimization
   $$
   a=b+1, c=a-1 \space \rightarrow \space c = b+1 -1 \space \rightarrow \space c=b
   $$

## Strength Reduction + helper function:

```c++
unsigned int getBestShiftValue(uint64_t constVal)
{

    // verifica se potenza di 2:
    APInt apInt(32, constVal);
    if (apInt.isPowerOf2())
    {
        return apInt.logBase2();
    }
    // verifica se potenza di due con offset +1
    APInt apIntPlusOne(32, constVal + 1);
    if (apIntPlusOne.isPowerOf2())
    {
        return apIntPlusOne.logBase2();
    }
    // verifica se potenza di due con differenza di -1
    APInt apIntMinusOne(32, constVal - 1);
    if (apIntMinusOne.isPowerOf2())
    {
        return apIntMinusOne.logBase2();
    }

    outs() << "Nessuna potenza valida è stata trovata!\n";
    return 0;
}

bool strengthReduction(Instruction &I)
{
    if (auto *BinOp = dyn_cast<BinaryOperator>(&I))
    {
        auto OpCode = BinOp->getOpcode();
        Value *Op1 = I.getOperand(0);
        Value *Op2 = I.getOperand(1);

        if (ConstantInt *constInt = dyn_cast<ConstantInt>(Op1))
            std::swap(Op1, Op2);

        if (!isa<ConstantInt>(Op2))
        {
            // Op2 non contiene una costante -> errore
            return false;
        }

        // DBUG: outs() << "Assegnamento variabili corretto!\n";
        ConstantInt *constInt = dyn_cast<ConstantInt>(Op2);
        // calcolo shift value

        unsigned int shiftValue = getBestShiftValue(constInt->getZExtValue());
        ConstantInt *shift = ConstantInt::get(constInt->getType(), shiftValue);

        // DBUG: outs() << "shiftValue: " << shiftValue << "\n";

        Instruction *newInstruction = nullptr;
        if (OpCode == BinaryOperator::Mul)
        {
            if (!Op1)
            {
                outs() << "OP1 riferimento nullo\n";
                return false;
            }
            Instruction *shiftLeft =
                BinaryOperator::Create(Instruction::Shl, Op1, shift);

            // Verifica che shiftLeft sia stato creato correttamente
            if (!shiftLeft)
            {
                outs() << "Errore: impossibile creare shiftLeft\n";
                return false;
            }
            // DBUG: outs() << "shiftLeft:  -> " << *shiftLeft << "\n";
            shiftLeft->insertAfter(&I);

            // calcolo del resto
            int64_t operationRest =
                static_cast<int64_t>(constInt->getZExtValue()) - (1 << shiftValue);
            outs() << "Triggered Strenght Reduction on " << I << "\n";

            // analisi del resto
            if (operationRest == 0)
            {
                newInstruction = shiftLeft;
            }
            else if (operationRest == 1)
            {
                newInstruction =
                    BinaryOperator::Create(BinaryOperator::Add, shiftLeft, Op1);
                outs() << "newInstruction: " << *newInstruction << "\n";
                newInstruction->insertAfter(shiftLeft);
            }
            else if (operationRest == -1)
            {
                newInstruction =
                    BinaryOperator::Create(BinaryOperator::Sub, shiftLeft, Op1);
                outs() << "newInstruction: " << *newInstruction << "\n";
                newInstruction->insertAfter(shiftLeft);
            }
            else
            {
                // Il resto non è 0, 1 o -1, quindi non eseguire la strength reduction
                return false;
            }
        }

        else if (OpCode == BinaryOperator::UDiv)
        {
            if (constInt->getValue().isPowerOf2())
            {
                newInstruction = BinaryOperator::Create(Instruction::LShr, Op1, shift);
                outs() << "newInstruction: " << *newInstruction << "\n";
                newInstruction->insertAfter(&I);
            }
        }
        if (newInstruction)
            I.replaceAllUsesWith(newInstruction);

        return newInstruction;
    }
    else
    {
        return false;
    }
}
```

Which gets invoked in this manner:

```c++
bool runOnBasicBlock(BasicBlock &B)
{
    // ottimizzatore punto 2
    std::vector<Instruction *> toRemove;

    for (auto &I : B)
    {
        if (strengthReduction(I))
        {
            toRemove.push_back(&I);
        }
    }

    // Rimuovi le istruzioni dopo aver completato il ciclo
    for (auto *I : toRemove)
    {
        I->eraseFromParent();
    }
    return true;
}

bool runOnFunction(Function &F)
{
    bool Transformed = false;

    for (auto Iter = F.begin(); Iter != F.end(); ++Iter)
    {
        if (runOnBasicBlock(*Iter))
        {
            Transformed = true;
        }
    }

    return Transformed;
}

PreservedAnalyses LocalOpts::run(Module &M, ModuleAnalysisManager &AM)
{
    for (auto Fiter = M.begin(); Fiter != M.end(); ++Fiter)
        if (runOnFunction(*Fiter))
            return PreservedAnalyses::none();

    return PreservedAnalyses::all();
}
```
