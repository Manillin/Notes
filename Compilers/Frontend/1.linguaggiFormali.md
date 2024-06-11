# Linguaggi Formali 

Chiamiamo **linguaggio** un sistema di parole e segni che le persone usano per comunicare pensieri e sentimenti.
Questo tipo di linguaggio risulta inadeguato per lo sviluppo di teorie matematiche e di algoritmi. 

Per questo tipo di formulazioni nascono i **Linguaggi Formali**, e diremo che dato un linguaggio di questo tipo, è (quasi sempre) possibile stabili se:
- Una frase è corretta $\rightarrow$ **Sintassi**.  
- Associare un significato preciso (no ambiguità) alle frasi $\rightarrow$ **Semantica**.  

## Nozioni Importanti:

### Alfabeto ($\space \sum\space$):
Insieme infinito di simboli (caratteri)
Es:
- $A$ = $\{a,b,c\}$
- $B = \{0,1\}\rightarrow$ alfabeto binario 
- $...$

### Linguaggio: 
Un linguaggio su un dato alfabeto $\sum$ è un insieme di stringhe composte dai caratteri di $\sum$.  
Un linguaggio può essere finito (possibile elencazione delle sue stringhe) o infinito (quelli di nostro interesse).  


Poichè i lunguaggi sono insiemi, su di essi sono definite tuttel eoperazioni insiemistiche!

es: Due linguaggi $M$ e $N$ su uno stesso alfabeto $\sum$ si posono concatenare:

$$
L = MN = \{z\in {\sum}^*: \exists x \in M, \exists y \in N \rightarrow z = xy \}
$$

$L$ è quindi costituito da tutte le stringhe che possono essere scritte come concatenazione di una stringa di $M$ con una stringa di $N$.  
L'elemento neutro per la concatenazione di linguaggi è il linguaggio costituito dalla sola stringa vuota: $\{\epsilon\}$

**Operazioni con i Linguaggi:**
- **Chiusura riflessiva** di $L$ è il linguaggio:  
    $${L}^* = \bigcup_{n=0}^{\infty} L^n = L^0 \cup L^1 \cup L^2 \cup ...$$  
    In generale possiamo dire che ${\sum}^*$ è l'insieme di tutte le stringhe su un alfabeto $\sum$, compresa la stringa vuota.  

- **Chiusura non riflessiva** di $L\rightarrow$ ${L}^+ = LL^*$, ovvero:  
    $$L^+ = \bigcup_{n=1}^{\infty}L^n = L^1 \cup L^2 \cup ... $$  
    Praticamente tutte le stringhe senza contare la stringa vuota.  

### Specifica di linguaggi:
Se il linguaggio è infinito può comunque essere descritto con quantità finita di informazioni!  
es: Il linguaggio $L_2$ costituito da tutte le strighe su $B$ che terminano con il carattere $0$: 
 $$L_0 = \{x\in B^* | x=y0, y\in B^*\}$$

---

### Caratterizzazioni e Specifiche:

Nel mondo informatico risultano molto più interessanti le  seguenti Caratterizzazioni e Specifiche: 
1. **Caratterizzazione Algoritmica (o riconoscitiva):**
    Si ha un algoritmo di decisione, tale che data in input una stringa su un dato alfabeto, risponda `yes` o `no`. Si può usare tale algoritmo per definire un linguaggio:
    $$L_A = \{x \in {\sum}^* | A(x)=True\}$$  
    Es: Il linguaggio C++ è l'insieme delle stringhe sull'alfabeto ASCII per cui il compilatore non produce errore!

2.  **Specifica Generativa:**  
    Tecnica tramite la quale si danno regole mediante le quali è possibile generare tutte e sole le stringhe del lingagguo che si vogliono specificare.  
    I due formalismi più importnati sono: **_Espressioni Regolari_** e **_Grammatiche Context-Free_**  
    Vedremo che questi due formalismi sono fondamentali per definire il comportamento del lexer e del parser.  



