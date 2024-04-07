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
