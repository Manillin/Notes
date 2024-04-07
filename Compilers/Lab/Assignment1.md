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

## Strength Reduction:

Declaring the first func

```c++
bool isAlmostPow2(Value *op, ConstantInt *&CI)
{
    if ((CI = dyn_cast<ConstantInt>(op)))
    {
        APInt value = CI->getValue();
        if (value.isPowerOf2())
            return true;
        value++;
        if (value.isPowerOf2())
            return true;
        value -= 2;
        if (value.isPowerOf2())
            return true;
    }

    return false;
}
```

```c++
bool strengthReduction(BasicBlock &BB)
{
    bool loc_changed = false;
    for (auto &I : BB)
    {
        if (auto *BinOp = dyn_cast<BinaryOperator>(&I))
        {
            auto OpCode = BinOp->getOpcode();
            if (OpCode != Instruction::Mul && OpCode != Instruction::SDiv)
                continue;

            // Retrieve the operands
            Value *Op1 = BinOp->getOperand(0);
            Value *Op2 = BinOp->getOperand(1);
            ConstantInt *constInt;

            // Check which operands holds the Constant intger value
            // Then check if is power of two (offset w sum or sub accepted)
            if (!isAlmostPow2(Op1, constInt) && !isAlmostPow2(Op2, constInt))
                continue;

            // Ensures to have the (almost) power of two operand in the Op2 variable
            if (isAlmostPow2(Op1, constInt))
                std::swap(Op1, Op2);

            // Extract the shift value
            unsigned ShiftVal;
            if (constInt->getValue().isPowerOf2())
            {
                ShiftVal = constInt->getValue().logBase2();
            }
            else if (constInt->getValue() + 1 == APInt(constInt->getBitWidth(), 1) << ShiftVal)
            {
                ShiftVal = constInt->getValue().logBase2() + 1;
            }
            else
            {
                ShiftVal = constInt->getValue().logBase2();
            }

            IRBuilder<> Builder(BinOp);
            // Create shift and adjustment instructions
            Value *ShiftInst;
            if (OpCode == Instruction::Mul)
            {
                ShiftInst = Builder.CreateShl(Op1, ShiftVal, "shift");
            }
            else
            {
                ShiftInst = Builder.CreateLShr(Op1, ShiftVal, "shift");
            }

            Value *AdjInst = nullptr;
            if (constInt->getValue() != APInt(constInt->getBitWidth(), 1) << ShiftVal)
            {
                if (OpCode == Instruction::Mul)
                {
                    AdjInst = Builder.CreateSub(ShiftInst, Op1);
                }
                else
                { // OpCode == Instruction::SDiv
                    AdjInst = Builder.CreateAdd(ShiftInst, ConstantInt::get(constInt->getType(), 1));
                }
            }
            else
            {
                AdjInst = ShiftInst;
            }
            BinOp->replaceAllUsesWith(AdjInst);
            BinOp->eraseFromParent();
            loc_changed = true;
        }
    }
    return loc_changed;
}

```

spiegazione del codie a seguire (TBD)
