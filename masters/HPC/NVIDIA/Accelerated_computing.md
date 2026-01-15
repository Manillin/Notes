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

Implementa il concetto di map, prende una collezione di dati, applica una funzione a _ogni_ elemento e salva il risultato da qualche parte.  

**Sintassi**: `std::transform(inizio_input, fine_input, inizio_output, operazione)`    

- `inzio_input` e `fine_input` definiscono l'intervallo di dati su cui lavorare 
- `inizio_outpt` indica dove iniziare a scrivere il risultato, non serve specificare la fine dell'output in quanto viene dato per scontato che la dimensione sia pari a quella dell'input  
- `operazione` + la funzione che viene applicata.    

Applica a ogni elemento della collezione l'operazione definita, passando l'elemento attuale come parametro all'operazione!  

```c++ 
    float k = 0.5;
    float ambient_temp = 20;
    std::vector<float> temp = {42, 24, 50};

    auto op = [=](float t)
    {
        float diff = ambient_temp - t;
        return t + k * diff;
    };
    for (int step = 0; step < 3; step++)
    {
        std::transform(temp.begin(), temp.end(),
                       temp.begin(), op);
    }
```

_nota:_ in questo caso è fondamentale specificare il parametro `float t` nella firma della lambda, altrimenti op non sa con quale elemento operare.  


### Heterogeneous programming 

Possiamo compilare il codice C++ per farlo eseguire su CPU o GPU, il compilatore classico compila per CPU mentre `nvcc` compila per GPU.  

Di default tutte le funzioni sono compilate per CPU ma possiamo fare offloading di certe operazioni e funzioni su GPU, per farlo dovremo specificarlo esplicitamente nel codice.  

### Thrust 

![thrust](../../images/sw_stack_nvidia.png)

Thrust ci permette di trasformare algoritmi che girano su CPU in algoritmi che girano su GPU.  
È un tool semplice che richiede minime modifiche al software per raggiungere il suo obiettivo. 
- dovremo usare thrust al posto di std 
- specificare il device sul quale vogliamo che il codice esegua, permette flessibilità e portabilità!  

![porting algo in GPU](../../images/porting_GPU.png) 

1. Execution Policy: Dice dove thrust girerà l'algoritmo, funziona a runtime (non compila in automatico per quella location)
    - `trust::host`: esegue su CPU
    - `thrust::device`: esegue su GPU

2. Specifier: Indica dove il codice **può eseguire**, lo compila per la location specificata, non esegue il codie automaticamente lì (per questo serve la policy).  
    - `__host__`: cpu, dice a nvcc di usare il compilatore normale
    - `__device__`: gpu, dice a nvcc che la funzione deve essere compilata per GPU
    - Si possono usare entrambi


Es: Modificare il codice per eseguirlo su GPU (Algoritmo standard della `std`)

```c++
float median(thrust::universal_vector<float>& vec)
{
    std::sort(vec.begin(), vec.end());
    return vec[vec.size() / 2];
}
int main()
{
    for(int step=0; step < 3; step++ )
    {
        float median_temp = median(temp);
    }
}
```

**Codice trasformato:**      
 
```c++
float median(...)
{
    thrust::sort(thrust::device, vec.begin(), vec.end());
    return vec[vec.size() / 2];
}
int main()
{
    ...
}
```

_note:_   
- non serve mettere `__device__` davanti a `thrust::sort`, in quanto thrust::sort è una funzione che viene chiamata da CPU, è lei che internamente si occupa di lanciare i thread su GPU
- lo specifier `__device__` serve quando passiamo a thrust un algoritmo personalizzato (come una lambda), in quanto specifica che quell'algo deve essere eseguito su GPU. Nel caso di thrust::sort quella logica è già dentro thrust.  
- Tutto il codice esegue su CPU **TRANNE** la sort.  
    - la CPU invia il comando alla GPU che ordina i dati in parallelo, la CPU aspetta poi che la GPU finisca.  
    - la divisione (vec.size()/2) e l'accesso (vec[]) vengono fatte dalla CPU in quanto più efficiente.  
- **Come fa la CPU a leggere il valore se è sulla GPU?**
    - usiamo la **Unified Memory di NVIDIA**, che viene applicata con `thrust::universal_vector` 
    - Quando la GPU fa sort i dati vengono spostati automaticamente dal driver nella memoria video
    - Quando la CPU prova ad accedere al dato `vec[index]`, il driver CUDA si accorge che la CPU ha bisogno dei dati e riporta indietro quel valore dalla GPU alla RAM della CPU.  

### Unified Memory 

La unified memory usata nell'esepio precedente gestisce la memoria in blocchi chiamati **pages** che solitamente sono grandi 4KB.  
Quando scriviamo `return vec[vec.size() / 2]`:
- La CPU prova a leggere l'indirizzo che corrisponde a quell'indice, ma in quel momento i dati si trovano fisicamente nella GPU (che ha appena finito il sort) 
- Si verifica un **Page Fault**: l'hardware della CPU si accorge che quel dato non è in RAM
- Interviene il driver CUDA che guarda dove si trova la pagina di memoria che contiene quell'indice e la sposta dalla GPU alla CPU (tramite PCI-express o NVLink). 
- Una volta che la pagina arriva nella RAM della CPU, essa può finalmente leggere il valore e completare l'operazione.  

Non viene passato l'intero vettore! se il vettore fosse grande 4GB spostarlo tutto in RAM per leggere un float sarebbe uno spreco assurdo, per questo si usano le pages, spostando solo 4KB il sistema è molto più efficiente.  


### Extending Standard Algorithms

Non esiste un algoritmo `thrust::` per tutti i casi, a volte abbiamo algoritmi specifici non della libreria standard che vorremmo ottimizzare.  

Es: Calcolare la differenza massima tra due vettori:   

**Approccio Naive**:  

```c++
thrust::universal_vector<float>
unnecessarily_materialized_diff(a.size());
// compute abs differences
thrust::transform(
    thrust::device,
    a.begin(), a.end(), // first input sequence
    b.begin(),          // second input sequence

    unnecessarily_materialized_diff.begin(),

    []__host__ __device__(float x, float y) {
    return abs(x - y);
});

// compute max difference
return thrust::reduce(
    thrust::device,
    unnecessarily_materialized_diff.begin(),
    unnecessarily_materialized_diff.end(),
    0.0f, thrust::maximum<float>{});
```

Contiamo gli accessi in memoria di questo algoritmo:  
- legge 2$N$ elementi e fa $N$ differenze
- la reduce fa $N$ letture per restituire il risultato finale
- $\rightarrow$ 3$N$ letture e 1$N$ scritture 
- Soluzione subottimale!  

Se l'avessimo scritto in plain C++ avremmo fatto:  

```c++
float max = 0.0f;
for(int i=0; i<N; i++)
{
    float ar = a[i]; //r
    float br = b[i]; //r 
    float diff = abs(ar-br);
    max = diff > max ? 
        diff: max;
}
```

- Leggiamo 2$N$ elementi ed evitiamo di usare un array di appoggio (risparmiamo 1N scritture rispetto a prima). 
- $\rightarrow$ 2$N$ letture e 1 scrittura!  

Come possiamo ottenere questi vantaggi con gli algoritmi di thrust ? usando i **Puntatori**.    

Ricordiamo che i puntatori sono elementi che contengono indirizzi della memoria fisica, e dereferenziandoli possiamo accedere a tali elementi.  

```c++
std::array<int, 3> a{0,1,2};
int *ptr = a.data();
std::printf("pointer[0]: %d\n", ptr[0]); //prints 0
std::printf("pointer[1]: %d\n", ptr[1]); //prints 1
```

Accediamo a elementi con l'operatore di `subscript` ossia `[]`.  
In C++ possiamo fare overloading di questo operatore per fare in modo che si comporti come desideriamo noi.  

Es: simple counting iterator 

```c++
struct counting_iterator
{
    int operator[](int i)
    {
        return i;
    }
};

counting_iterator it;
std::printf("it[0]: %d\n", it[0]); // prints 0
std::printf("it[1]: %d\n", it[1]); // prints 1
```

**Vantaggi degli iterators**: 
- forniscono una interfaccia come quella dei puntatori
- generalizzano i puntatori con l'overloading dell'operatore
- non sono vincolati dall'accesso raw in memoria (che è lento!) 
- riducono il memory footprint
- riducono il traffico di memoria che migliora le performance. 

Questo tipo di iteratori sono **fondamentali** per la programmazione su GPU, in quanto non esiste il costrutto di for classico.  
Il classico `for(int i=0,...)` è codice **seriale**, sulla GPU è tutto parallelo e usiamo algoritmi specifici come thrust::for_each o thrust::transform.  
Questi algoritmi pretendono degli iteratori per sapere su quale intervallo di dati lavorare.  

In breve:
- il ciclo for è un ordine impartito a un solo core della CPU.
- l counting_iterator accoppiato a un algoritmo Thrust è un ordine impartito a migliaia di core della GPU.


**Nested Iterators**    


```c++
struct zip_iterator
{
    int *a;
    int *b;

    std::tuple<int, int> operator[](int i)
    {
        return {a[i], b[i]};
    }
};

struct transform_iterator
{
    zip_iterator zip;
    
    int operator[](int i)
    {
        auto [a,b] = zip[i];
        return abs(a-b);
    }
};

std::array<int, 3> a{ 0, 1, 2 };
std::array<int, 3> b{ 5, 4, 2 };
zip_iterator zip{a.data(), b.data()};
transform_iterator it{zip};
std::printf("it[0]: %d\n", it[0]);
```

### Thrust Fancy Iterators

Thrust mette a disposizione funzioni di fabbrica per creare questi iteratori.  

- `thrust::make_counting_iterator(n)`
- `make_transform_iterator(iteratore, func)`
- `make_zip_iterator(iteratore1, iteratore2)`. 

Quando si invocano bisogna passare come parametro altri iteratori (che possono essere puntatori come vec.begin() o altri fancy iterators). 

Es di transform iterator:

```c++
thrust::universal_vector<int> vec = {1, 2, 3, ...};
auto begin = 
    thrust::make_transform_itertor(
        vec.begin() // iteratore iniziale -> puntatore

        [] __host__ __device__ (int val){
            return value * 2;
        }
    );
auto end = begin + 42;
thrust::for_each(thrust::device, begin, end, print);

// prints: 2 4 6 8 ... 84
```

_nota:_  
- quando facciamo `auto end = begin + 42;` usiamo l'aritmetica degli iteratori, in C++ un range è definito come [inizio, fine) 
    - `begin` punta al primo elemento della catena (corrisponde a vec[0])
    - `end` punta a 42 posizioni dopo 

### Soluzione al problema iniziale: 

```c++
auto zip_it = thrust::make_zip_iterator(a.begin(), b.begin());
auto transform_it = 
    thrust::make_transform_iterator(zip_it, 
        [] __host__ __device__ (thrust::tuple<float,float> t)
        {
            return abs(thrust::get<0>(t) - thrust::get<1>(t));
        });
```

![zip transform iterator](../../images/zip_transoform_iterator.png)   

Adesso la computazione avviene quando accediamo all'iteratore:
- zip_it[i] restituisce la tupla 
- transform_it[i] restituisce la differenza

