### Lambda Functions in C++:  

Piccola funzione anonima che non richiede di essere dichiarata esternamente  

**Sintassi:** `[capture] (parameters) { code }`   

- `[capture]`: cattura variabili , avviene al momento della definizione dell'oggetto lambda! Possiamo passare variabili separate da virgola (per valore o per riferimento).  
Se vogliamo tutto lo scope per valore usiamo `[=]` , se lo vogliamo per riferimento usiamo `[&]`.  
- `(parameters)`: i parametri in ingresso della lambda  
- `{ code }`: contiene il codice vero e proprio che fa la computazione

Es:

```c++
#include <iostream>

int main()
{
    int x = 10;
    auto lambda = [&x]()
    {
        x = 1;
    };

    auto lambda2 = [x]()
    {
        std::cout << "Valore di x: " << x << std::endl;
    };
    lambda();
    lambda2();
    std::cout << x << std::endl;
}
```

_note_:  
1. il valore di x viene catturato da lambda2 al momento della sua definizione, in questo istante vale 10 e il risultaot sarà quello, nonostante x valga 1 al momento della invocazione della lambda.  

2. modifichiamo x con lambda1 in quanto viene passato per riferimento

3. se avessimo passato x per valore a lambda1 avremmo avuto un errore in quanto i parametri catturati per valore sono _read only_.  


### Transform in C++ 