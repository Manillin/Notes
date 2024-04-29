# DataFlow Analysis:

# Table of Contents:

1. [Struttura della DataFlow Analysis](#dataflow-analysis)
2. [Reaching Definitions]()
3. [Liveness Analysis]()
4. [Generalizzazione ]()

# Tipi di Analisi:

Esistono diversi tipi di analisi:

### Analisi Locale (es: Local Value Numbering):

Analizza l'effetto di ogni **istruzione** e compone l'effetto delle istruzioni per derivare informazione dall'inizio del _BasicBlock_ ad ogni istruzione.

### Analisi Globale: $\rightarrow$ Dat Flow Analysis

1. Analizza l'effetto di ogni _BasicBlock_
2. Compone l'effetto dei vari _BasicBlock_ per derivare informazione ai confini (IN e OUT) dei _BasicBLock_ stessi.

**Note:** Dai confini dei _BasicBlocks_ si possono applicare tecniche (descritte sotto) per generare informazioni sulle istruzioni.

# Data Flow Analysis (DFA):
La DFA permette per ogni variabile `x` di un programm di derivare informazioni come: valore di `x` ? ; quale definizione definisce `x`; quale definizione di `x` è ancora valida? ...

## Effetti di Istruzioni e Basic Block:

### Effetti Istruzioni:

Un istruzione può avere i seguenti effetti:
es: `a = b + c `

- **Usa** delle variabili $\rightarrow$ `b,c`
- **Uccide** una precedente definizione $\rightarrow$ `a`
- **Definisce** una variabile $\rightarrow$ `a`

### Effetti Basic Blocks:

Facendo una composizione degli effetti delle istruzioni che compongono un BB si derivano gli effetti del Basic Block stesso.

-
-
