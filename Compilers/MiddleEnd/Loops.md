# Loops e UD-DU Chains:

Ricordiamo che i `Loop` sono fondamentali pre i passi di ottimizzazione MiddleEnd in quanto i programmi spendono la maggior parte del loro tempo dentro ad essi.

**Obiettivo:** $\rightarrow$ Definire un Loop in termini di `GraphTheory` (CFG)

## Definizione di Loop:

Singolo ENTRY Point e gli archi devono formare almeno un ciclo

### Formalmente:

- **Dominator:**
  Un nodo $d$ domina un nodo $n$ in un grafo $(d dom n)$ se **ogni** percoros dall'ENTRY node a $n$ passa per $d$.

- **Dominator Tree:**
  Possiamo rappresentare i _Dominators_ con un albero:
  - $a\rightarrow b$ nel dominator tree se $a$ domina **immediatamente** $b$
  - Il nodo ENTRY è `root` e ogni nodo $d$ domina solamente i suoi discendenti nell'albero.
  - **Immediate Dominator:** L'ultimo **dominator** di $n$ su _qualsiasi_ percorso da ENTRY ad $n$

![CFG before DomTree]()
![Dom Tree]()

# Loop Naturali:

Costrutto di Loop che si verifica naturalemente nel codice sorgente e che può essere **identificato** attraverso le proprietà strutturali del CFG.
Un Loop naturale ha le seguenti caratteristiche:

- Singolo Entry Point (Header):
- Dominanza dell'Header:
- BackEdge:
- Dominanza del BackEdge:
