# Most impactful C++ principles for llvm

I seguenti concetti di C++ sono essenziali per capire e poter usare la toolchain di llvm

## Namespace:

I name space in C++ sono usati per organizzare il codice gerarchicamente e per prevenire conflitti di nomi tra le varie diverse parti di un programma.  
Possono essere **usati per raggruppare** un insieme di nomi: funzioni, classi e variabili (in uno specifico ambito).

- Namespace _Globale_ al quale automaticamente appartengono tutti i nomi non esplicitmente dichiarati in uno specifico namespace.
- Namespace _definiti dall'utente_ :
  si definiscono con la kw `namespace` e vengono racchiusi tra parentesi graffe, essi non devono per forza essere definiti nello stesso file, si possono appendere nuovi nomi tramite l'impiego di diversi file.

Per questo progetto ci interessiamo dei namespace **locali**.  
Per accedere a tutti i tipi di nomi possibili in un namespace usiamo la notazione `::` seguita dal nome specifico.  
Es:

```c++
#include <iostream>

namespace my_namespace{
    int variabile = 5;

    int saluta(){
        std::cout<<"Ciao dal mio namespace!\n";
    }
}

int main(){
    my_namespace::saluta(); // stamperà il greeting message
    std::cout<< my_namespace::variabile <<endl; //stamperà 5
    return 0;
}
```

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

## File di Header .h

Un file di header CPP è un file che contiene dichiarazioni di funzioni e classi e consente l'utilizzo di tali ad altri file senza fornire i dettagli implementativi.  
Forniscono quindi una **Separazione** dall'_interfaccia_ rispetto a quella dell'_implementazione_

**USO:**

- I file di header vengono inclusi in altri file che vogliono usare le funzioni definite in esso tramite la direttiva `#include`.
- Il preprocessore C++ sostituisce la direttiva `#include` con il contenuto del file .h consentendo al compilatore di avere accesso alle dichiarazioni dei nomi.
- È buona norma usare `#ifndef`, `#define` , `#endif` per evitare inclusioni multiple e assicurarsi che un file .h venga incluso una sola volta.
