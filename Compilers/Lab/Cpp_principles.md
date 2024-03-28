# Most impactful C++ principles for llvm

I seguenti concetti di C++ sono essenziali per capire e poter usare la toolchain di llvm

## Namespace:

## Template:

Consentono di scrivere codice generico, parametrizzabile per lavorare con diversi tipi di dati senza dover essere riscritto per ogni tipo.  
Possono essere utilizzati per:

1. Classi $\rightarrow$ template di classe
2. Funzioni $\rightarrow$ template di funzione
3. Alias $\rightarrow$ Not important for llvm

**Attenzione:** Quando si usano i template è essenziale che il _parametro_ pasato sia compatibile con le funzioni operate dal template stesso; Se tipi di dato o classi non sono compatibili come parametri si verificheranno errori di compilazione.
Es:

```c++
// Istanziazione di una classe che eredita da una classe template:

class HelloWorldPass: public PassInfoMixin<HelloWorldPass>{
    // code
}
```

In questo esempio avremo che PassInfoMixin è una classe template, e nello specifico HelloWorldPass dovrà rispettare i prerequisiti di interfaccia richiesti, altrimenti questa operazione genererà un errore.
