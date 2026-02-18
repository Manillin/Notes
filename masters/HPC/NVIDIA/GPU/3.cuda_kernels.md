# Implementazione di algoritmi con kernel CUDA     


Le librerie come thrust e cub non sono magiche, sono solo codice ottimizzato scritto da altri programmatori, se l'algoritmo specifico che vogliamo usare non esiste in queste librerie dovremmo scriverlo noi da zero lanciando il lavoro su GPU.  

### Qualificatori

Per scrivere codice GPU usiamo delle keyword speciali che dicono al compilatore nvcc dove deve essere eseguito quel codice: 
- **__host__**: Codice standard C++, viene compilato per CPU, è il modo di esecuzione di defualt (nel caso non venga specificato altro).  
- **__device__**: Codice di funzioni che vivono **solo** su GPU, vengono chiamate dalla GPU ed eseguite sulla GPU. 
- **__global__**: Funziona come ponte tra CPU e GPU, infatti viene invocata dalla CPU (host) ma eseguita sulla GPU (device)   

Il nostro codice iniziale è:    

```cpp
using temperature_grid_f = 
    cuda::std::mdspan<float, cuda::std::dextents<int, 2>>;

void simulate(temperature_grid_f in,
            float* out, 
            cudaStream_t stream)
{
    auto ids = thrust::make_counting_iterator(0);
    cub::DeviceTransform::transform(
        ids, out, in.size(),
        [in] __host__ __device__ (int cell_id){
            return dli::compute(cell_id, in);
        },
        stream

    );
}
```


Scriviamo lo stesos codice usando un singolo Kernel:  

```cpp
__global__ 
void single_thread_kernel(dli::temperature_grid_f in, float *out)
{
    for(int id = 0; id<in.size;id++)
    {
        out[id] = dli::compute(id,in);
    }
}

void simulate(dli::temperature_grid_f in,
        float *out,
        cudaStream_t stream)
{
    single_thread_kernel<<<1,1,0>>>(in,out);
}
```


- la sintassi per usare i kernel è:   
    `kernel_function<<<X,Y,Z>>>(in, out)`    
    - X è il blocco 
    - Y è il numero di thread per blocco 
    - Z è il riferiemento allo stream di appartenenza. 
- questo codice è estremamente lento, ci mette circa 4.15s per fare la computazione vs i 0.000043s di CUB
    - questo perchè stiamo usando la GPU come se fosse una CPU, abbiamo usato un solo thread per fare tutto il lavoro.  


**Proviamo a fare la stessa cosa ma con due thread**:   

```cpp
const int number_of_threads = 2;

__global__ block_kernel(dli::temperature_grid_f in , float *out)
{
    int thread_index = threadIdx.x;
    for(int id = thread_index; id< in.size; id+=number_of_threads)
    {
        out[id] = dli::compute(id,in);
    }
}

void simulate(dli::temperature_grid_f in, 
        float *out, 
        cudaStream_t stream)
{
    block_kernel<<<1, number_of_threads, 0, stream>>>(in, out);
}
```

- threadIdx.x mi permette di ottenere il suo identificativo numerico 
- il ciclo for parte dal thread index e salta di 2 (numero di thread), stiamo intervallando le iterazioni da eseguire tra i due thread!  
    - th0 esegue celle: 0, 2, 4, ... 
    - th1 esegue celle: 1, 3, 5, ... 
    - lavorano insieme. questo pattern si chiama grid-stride loop.  

- Risultato: il tempo di esecuzione diventa 2.6s, è stato dimezzato!
- se proviamo con 256 thread (`<<<1,254,stream>>>`) il tempo crolla a 0.037s!  

Ogni architettura GPu ha un limite fisico di thread che possono stare dentro un singolo blocco, attualmente il limite è **2024 thread per blocco!**     
Se proviamo a chiederne di più il kernel non parte nemmeno e riceviamo errore.   


### Gerarchia GPU: 

La GPU organizza i thread in modo gerarchico per gestire milioni di operazioni:

- **thread**: unita singola (operaio)
- **block**: un gruppo di thread, tutti i thread in un blocco possono comunicare facilmente e risiedono sullo stesso multiprocessore (limite di 1024 thread per blocco)
- **grid**:  L'insieme di tutti i blocchi lanciati per un kernel


Variabili di sistema importantissime:
- `blockIdx.x`: restituisce l'indice del blocco attuale (es: sono blocco 2)
- `blockDim.x`: restituisce il numero di thread presenti in ogni blocco (es: 256)
- `gridDim.x`: restituisce il numero di blocchi totali che compongono la griglia.  


Per sapere quanti thread abbiamo attualmente attivi possiamo usare la formula (1).  
Per avere l'**indice globale** di un thread all'interno delll'intero sistema usiamo la formula (2) (ci restituisce la posizione assoluta alla griglia, non quella locale al blocco).  

```cpp
// Formula (1)

int numero_totale_thread = 
    blockDim.x. * gridDim.x;

// Formula (2)

int thread_index = 
    blockDim.x * blockIdx.x + threadIdx.x;
```


![blocks and grid](../../images/cuda_blocks_grid.png)      


### Scegliere le dimensioni:

- **Dimensioni del blocco**:   
    La dimensione del blocco _NON_ dipende dal dalla dimensione del problema _MA_ dall'architettura della GPU.    
    - Regola: usare sempre multipli di 32 (in quanto i thread vengono eseguite in gruppi da 32 chiamati warps)     
    - Tip: Si consiglia di usarne 256 di default. 

<br>

- **Dimensioni della griglia**:  
    La griglia dipende dalla dimensione del problema!  
    Vogliamo che esistano abbastanza blocchi da coprire tutti i dati, useremo la seguente formula:
    - grid_size = `ceil_div(totale_elem / block_size);`  
    - Se abbiamo ad esempio 1000 elementi e blocchi da 256, serviranno 4 blocchi totali! (se ne usassimo 3 avremmo 768 elementi coperti).  



Risultato finale:

```cpp
__global__
void grid_kernel(dli::temperature_grid_f in, float *out)
{
    int thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
    int n_threads = blockDim.x * gridDim.x;

    for(int id = thread_index; id < in.size(); id+= n_threads)
    {
        out[id] = dli::compute(id,in);
    }
}

void simulate(dli::temperature_grid_f in,
        float *out, 
        cudaStream_t stream)
{
    int block_size = 256;
    int grid_size = cuda::ceil_div(in.size(), block_size);
    
    grid_kernel<<<grid_size, block_size,stream>>>(in, out);
}
```


### Spiegazione sull'esecuzione del codice:

- Osserviamo `__global__` sulla firma della funzione kernel, questo fa si che quando la CPU arriva a leggere `grid_kernel<<<grid_size, block_size,stream>>>(in, out);` non esegue lui direttamente la funzione ma la manda alla GPU.  
- Invia un pacchetto di istruzioni al controller della GPU, in cui dice quante grid e quanti thread istanziare. 
- La CPU subito dopo aver mandato questo ordine, prosegue alla riga successiva del programma! il lancio del kernel e la sua computazione è un operazione ASINCRONA non bloccante per la CPU.         

<br>

- L'hardware della GPU (scheduler) accede migliaia di thread contemporaneamente e fornisce a tutti gli stessi dati (grid)
- Ogni thread svolgerà un operazione diversa nella grid grazie al loro identificatore thread_index e grazie al fatto che la logica preveda che il ciclo for avvenga con accesso a stride.  
- supponiamo di avere 1000 elementi e 256 thread totali (e 1 solo blocco)
    - th0: entra nel loop con `id=0`, fa il calcolo e poi fa `id+256`. 
        al prossimo giro lavorerà su `256`, poi su `512` e poi `768`.
    - th1: entra con `id=1`, poi farà `257`, `513`, `769`.


<br>

## Gerarchia dei thread

I thread sono divisi in blocchi per una buona ragione, la risposta è che i thread nello stesso blocco possono scambiarsi dati velocemente tramite la **shared memory**.  


**Gestire i dati nei kernel in modo sicuro**:   

NVIDIA consiglia l'uso di uno strumento del C++ per gestire i dati in modo sicuro all'interno dei kernel: `span`. 

```cpp
cuda::std::span<int> span(a.data(), 3);
span.size(); // restituisce 3 
```

È la variante monodimensionale di mdspan che abbiamo visto in precedenza.  
- vantaggi: È una view ma include informazioni sulla dimensione, è molto più _sicuro_ dei puntatori nudi perchè riduce gli errori ed è leggero.  

Per inizializzare un oggetto span gli si passa un puntatore ai dati e la dimensione.  

Ci possiamo interfacciare ai dati come ad un array standard con l'operatore `[]` che ci permette l'accesso (span[0] legge il primo elemento).   


### Esempio dell'Istrogramma 

Torniamo alla nostra simulzione del calore, ora vogliamo creare un istrogramma dalle temperature ottenute.   
Un istrogramma raggruppa dati in bin contando quanti valori cadono nell'intervallo definito nel bin.  

Nel nostro caso questo mostra la distribuzione della temperatura 
(l'ortogramma è usato per valori discreti, non utile in questo caso).  

![histogram](../../images/histogram_cu.png)


Ogni bin copre un range di temperature, e l'altezza della barra mostra quante celle cadono in tale range.  


Lo eseguiamo con un kernel:  

```cpp
__global__ void histogram_kernel(
    cuda::std::span<float> temperatures, 
    cuda::std::span<int> histogram
    )
{
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    int bin = static_cast<int>(temperatures[cell] / bin_width);

    int old_count = histogram[bin];
    int new_count = old_count +1;
    histogram[bin];
}
```

**!!! Attenzione: il codice presenta una data race !!!**     


Il nostro kernel ha una **datarace**.  
Nel nostro esempio abbiamo circa 4 milioni di celle; milioni di thread leggono e scrivono dalla e nella stessa locazione di memoria!  

![histogram datarace](../../images/hist_datarace.png)


### Operazioni di memoria Atomiche    

Le operazioni di memoria atomiche sono la soluzione alle data race.  
Ci permettono di render un operazione di `Read-Modify-Write` indivisibile.  

```cpp
cuda::std::atomic_ref<int> ref(count[0]);
ref.fetch_add(n);
ref.fetch_sub(n);
ref.fetch_and(n);
```

- `cuda::std::atomic_ref<T>` è un wrapper che applicato a una variabile permette di applicare operazioni atomiche su di essa.    
- le funzioni comuni iniziano con .fetch_op();
- **vantaggio:** È estremamente leggero e permette di usare le operazioni atomiche solo dove servono senza dichiarare tutta la memoria come atomica.  

![atomic timeline](../../images/atomic_timeline.png)     


L'istogramma in questo modo segna il valore corretto, non avremo ghost updates! L'hardware ha garantito che i thread non si calpestassero i piedi.  

- esempio di blocco hardware:  
    - th1 lancia fetch_add(1) sul bin0, quindi legge il valore 0 e calcola 1 come nuovo valore 
    - mentre sta calcolando (o dopo aver calcolato) il th2 cerca di fare anche lui la sua fetch_add(1) sullo stesso bin0.  
        il th2 viene messo in attesa! non può leggere ne scrivere fino a quando l'intero ciclo read-modify-write del th1 non ha finito!  
    - il th1 finisce la sua operazione atomica
    - a questo punto il th2 può può finalmente iniziare e fare la sua operazione.  

Prezzo da pagare: le operazioni atomiche sono più lente di una scrittura normale perchè costringono i thread che colpiscono lo stesso bin a mettersi in coda (**serializzazione**), ma è l'unico modo per avere risultati corretti ed eliminare le data race in algoritmi come quello sopra dell'istrogramma.   


<br>

### Privatizzazione 

Abbiamo visto che le operazioni atomiche ci permettono di avere codice corretto ma c'è un grosso problema per le performance:  

**Le operazioni atomiche sono lente se c'è troppa contention (competizione)**    

Se abbiamo 4 milioni di celle e tutte cadono nello stesso bin, i 4 milioni di thread devono mettersi in fila indiana.  
- Impatto su GPU: la GPU che è nata per fare migliaia di cose insieme, finisce per fare una sola operazione all volta $\rightarrow$ serializzazione!!!  
- I numeri: con 16.384 blocchi da 253 thread, se tutti puntano allo stesso bin creiamo una coda di 4.194.304 operazioni atomiche! abbiamo trasformato la nostra GPU in una CPU... dobbiamo evitare al massimo la serializzazione di tutte le operazioni su GPU, quando possibile.  


![istogramma serializzato](../../images/hist_ser.png)


La soluzione per diminuire il numero di operazioni in coda serializzate è quella di usare **Istogrammi privatizzati**.  

- creiamo un piccolo istrogramma privato per ogni blocco di thread, in questo modo i 256 thread si contendono i dati (meglio di 4milioni), molto meno traffico.  
- una volta che il blocco ha finito di contare i suoi dati, il risultato parziale del blocco biene aggiunto in modo atomico all'istrogramma globale 
- **Risultato:** la contention sull'istrogramma globale si riduce drasticamente, enllo specifo, di un fattore pari alla dimensione del blocco, quindi 256 volte meno traffico!  


![istogramma privatizzato](../../images/hist_priv.png)


**Implementazione tecnica**:   

Passiamo al Kernel un buffer chiamato `block_histograms`.  
Useremo la funzione `subspan` per ritagliare una fetta di questo buffer e assegnarla come esclusiva a un blocco di thread.  
- per calcolare l'offset: `blockIdx.x * histogram.size()` - ogni blocco riceve la sua porzione di memoria basandosi sul suo ID. 



```cpp
__global__ void histogram_kernel(
    cuda::std::span<float> tempertures, 
    cuda::std:span<int> block_histograms,
    suda::std::span<int> histogram)
{
    cuda::std::span<int> block_histogram = 
        block_histograms.subspan(
            blockIdx.x * histogram.size(),
            histogram.size());
        int cell = blockIdx.x + blockDim.x + threadIdx.x;
        int bin = static_cast<int>(temperatures[cell] / bin_width);

        cuda::std::atomic_ref<int> block_ref(block_histogram[bin]);
        block_ref.fetch_add(1);

        if(threadIdx.x < histogram.size())
        {
            cuda::std::atomic_ref<int> ref(histogram[threadIdx.x]);
            ref.fetch_add(block_histogram[threadIdx.x];)
        }
}
```


### Spiegazione codice:  
- `temperatures` è la nostra solita griglia con 4M di cell (in questo caso viene trattata in 1D)   
- `block_histograms` è un'array gigante che contiene la replica totale dell'istrogramma per ogni singolo blocco!  
    Se l'istogramma globale contiene 10 bin e abbiamo 1000 blocchi allora block_histograms conterrà (10 x 1000) = 10.000 elementi.  
- `histogram` è l'istrogramma globale, ha 10 elementi (uno per ogni bin).

<br>

- `.subspan(offset, count)` metodo di uno span che restituisce un nuovo oggetto span che punta a una finestra specifica; lo usiamo per fare accedere a tutti i thread di uno specifico blocco alla loro porzione di block_histograms corretta! 
- `cell` tramimte la formula permette di fare lavorare a ogni thread su un dato diveso (attenzione: qui viene omesso il controllo di boundaries per brevità, ma serve nel caso in cui avessimo più thread che celle).   

<br>

- `__synchtreads()`: permette la sincronizzaizone tra i thread di un blocco! viene invocata cooperativamente da tutti i thread di un blocco e fa si che i thread che arrivano a questo punto si fermino ad aspettare che tutti gli altri thread del blocco arrivino anch'essi a destinazione.  
    Simile a una barriera / fence.  
    

<br>

- `threadIdx.x < histogram.size()`: attiva l'aggiornamento finale! dopo la `__synchthreads()` ogni blocco ha il suo istogramma locale pieno, ora dobbiamo sommarlo a quello globale!  
    - ci sono solo 10 bin ma abbiamo 256 thread nel blocco 
    - allora diciamo solo ai primi 10 thread del blocco di mandare gli aggiornamenti 
    - la condizione sopra fa si che solo i thread con id da 0 a 9 lavorino, gli altri restano fermi.  
- `ref.fetch_add(block_histogram[threadIdx.x])` prende il bin di id threadIdx.x (che grazie alla condizione sopra sappiamo essere tra 0 e 9) e lo aggiunge all'istogramma globale!  


### Thread Scopes 

Distinguiamo tra due possibili modi di fare operazioni atomiche:
- `cuda::std::atomic_ref`: segue lo standard C++ classico, è sicura e non puoi specificare il raggio d'azione.  

<br>

- `cuda::atomic_ref<T, scope>`: È l'estensione specifica di NVIDIA che aggiunge il paramentro `thread_scope`.  
    - lo scope è il parametro entro il quale l'operazione atomica è garantita! Più è piccolo lo scope e più diventa veloce l'operazione.  


**Tipi di scope:**    

1. `cuda::thread_scope_system`: l'operazione atomica è sincronizzata tra tutto il sistema (coinvolge tutti i thread di tutte le GPU e anche la CPU), da usare solo se si hanno dati che vengono modificati contemporaneamente da una CPU e una GPU.  
2. `cuda::thread_scope_device`: l'operazione è garantita solo tra tutti i thread di una GPU 
3. `cuda::thread_scope_block`: l'operazione è garantita tra tutti i thread dello stesso blocco! i thread del blocco0 non vedono e non sono rallentati dalle operazioni atomiche del blocco1.  
    La competizione del dato avviene solo denrto l'**SM** (streaming multiprocessor)



```cpp

__global__ void histogram_kernel(
    cuda::std::span<float> tempertures, 
    cuda::std:span<int> block_histograms,
    suda::std::span<int> histogram)
{
    cuda::std::span<int> block_histogram = 
        block_histograms.subspan(
            blockIdx.x * histogram.size(),
            histogram.size());
        int cell = blockIdx.x + blockDim.x + threadIdx.x;
        int bin = static_cast<int>(temperatures[cell] / bin_width);

        cuda::atomic_ref<int, cuda::thread_scope_block> \
            block_ref(block_histogram[bin]);

        block_ref.fetch_add(1);

        __synchthreads();

        if(threadIdx.x < histogram.size())
        {
            cuda::atomic_ref<int, cuda::thread_scope_device> \
                ref(histogram[threadIdx.x]);
            ref.fetch_add(block_histogram[threadIdx.x];)
        }
}
```




<br> 


### Architettura di una GPU

Una GPU è composta da un insieme di SM - Streaming Multiprocessors.  
Un blocco è il gruppo di thread che lanciamo.  
- quando lanci un kernel, CUDA assegna ogni blocco a un SM; un SM può gestire più blocchi ma un blocco non può mai essere diviso tra due SM. 
- La cache L1 / shared memory sono fisicamente dentro l'SM, possiamo sfruttare questo fatto per valocizzare e mugliorare le performance dei nostri algoritmi.  
- Global Memory (VRAM) è fisicamente più lontana rispetto ai SM, è enorme, accessibile da tutti ma lenta.  
- L2 cache: una via di mezzo, condivisa tra tutti gli SM 
- L1 cache: vive dentro ogni SM, velocissima e gestita dall'hardware

![gpu arch](../../images/GPU_arch.png)   


Analizziamo il codice precedentemente visto, così com'è presenta una sotto-ottimizzazione:
- le operazioni atomiche fatte sulla memoria globale finiscono nella L2 cache
- stiamo sprecando spazion in VRAM per salvare istogrammi parziali che servono solo temporaneamente
- vorremmo poter allocare questi istogrammi parziali nella L1 cache che è molto più vicina ai core di calcolo.  



### Shared Memory

È una cache gestita dal programmatore, fisicamente si trova nello stesso posto della L1 cache (massima velocità) 
- È software defined: decide il programmatore stesso cosa metterci dentro (a differenza della L1 dove è l'hardware a decidere).  

Per usare questa memoria usiamo la seguente sintassi:

```cpp
__shared__ int shared_array[4];
```

Questa riga, letta dentro un kernel dice alla GPU di creare un array visibilki a tutti i thread di _questo_ blocco, ma invisibile a tutti gli altri blocchi.  


**Sincronizzazione**:  
Si usa come un array normale ma bisogna stare molto attenti alle **data race** .  
È obbligatorio usare `__syncthreads()` per coordinare i lavori


Codice ottimizzato:  

```cpp 
__global__ void histogram_kernel(
    cuda::std::span<float> temperatures, 
    cuda::std::span<int> histogram)
    {
        // istogramma parziale vive nella shared memory!
        __shared__ int block_histogram[num_bins];

        if (threadIdx.x < num_bins)
        {
            // init a zero dai primi 10 thread  
            block_histogram[threadIdx.x] = 0;
        }
        // synch per assicurarci l'init sia conclusa
        __syncthreads();

        int cell = blockIdx.x * blockDim.x + threadIdx.x;
        int bin = static_cast<int>(temperatures[cell] / bin_width);

        cuda::atomic_ref<int, cuda::thread_scope_block> \
            block.ref(block_histogram[bin]);
        
        block_ref.fetch_add(1);
        
        __syncthreads();
        // ... resto del codice ... //
    }
```

- inizializzare con 0 l'istrogramma parziale in shared memory è essenziale, in quanto quando definiamo un array/struttua in shared memory questa contiene inizialmente solo spazzatura.   
    Usiamo i thread nel blocco per eliminare questa spazzatura.  

- Questa ottimizzazione ci porta da 100GB/s a 400GB/s 


<br>

### Algoritmi Cooperativi

Scrivere i kernel da zero è potente ma richiede tempo, esistono librerie cooperative che girano dentro i kernel! conviene sfruttarle per velocizzare la scrittura di codice e avere maggiori performance.  

**Gerarchia di lavoro**:
1. Seriale: un thread fa tutto il lavoro (lento)
2. Cooperativo: molti thread di un blocco lavorano insieme su un unico compito 
3. Parallelo: più algoritmi cooperativi girano in parallelo su blocchi divesi 


![algoritmi cooperativi](../../images/cooperative_algos.png)    

Esistono librerie cooperative messe a disposizione da NVIDIA:
- CUD: algoritmi generici 
- cuBLASDx / cuFFTDx: algoritmi di algebra lineare e trasformazioni di fourirer ottimizzate per girare in kernel
- cooperativo significa: i thread del blocco lavorano insieme per risolvere un **unico** problema!  


**Modello mentale**:
- gli algoritmi cooperativi usando sempre la shared memory per passarsi dati tra thread
- tutti i thread del blocco **devono** partecipare. se metti un algoritmo CUB dentro un `if` che solo metà thread eseguono il kernel si blocca (deadlock)
- Alla fine del calcolo, spesso, il risultato si trova nel thread 0 (leader del blocco)    


### Interfaccia di CUB 

Per sfruttare CUB a basso livello, sfruttiamo e usiamo le **strutture template**.  
CUB espone i suoi algoritmi cooperativi come **struct template**     


```cpp
template <typename T, int blockDimX>
struct cub::BlockReduce
{
    struct TempStorage { ... };
    __device__ BlockReduce(TempStorage& temp_storage){ ... }
    __device__ T Sum(T input) { return ...;}
}
```

I parametri template servono per specificare algoritmi:
- specificati in Template<typename ....>
- data type 
- numero di thread 
- grain size (numero di item per thread)

Il TempStorage nested serve a dare un tipo allo storage temporaneo neccessario agli algormit cooperativi per la comunicazione
- deve essere il programmatore ad allocare questo spazio con `__shared__` e passarlo all'algoritmo.  
- consiste nel 'tavolo da lavoro' nella shared memory


### Usare CUB dentro un kernel 

Si devono fare i seguenti 3 passaggi:

1. **Allocazione**: `__shared__ cub::BlockReduce<int,4>::TempStorage storage;`
2. **Costruzione**: `cub::BlockReduce<int,4> reducer(storage);`    
    Creiamo l'oggetto passandogli lo storage  
3. **Invocazione**: `int blocKk_sum = reducer.Sum(threadIdx.x);`   
    Il thread 0 riceverà la somma di tutti i valori passati dai thread del blocco.  


**Esempio concreto:**     


```cpp
template <
    typename T,
    int BlockDimX,
    int ItemsPerThread,
    int Bins,
    cub::BlockHistogramAlgorithm \
        Algorithm = BLOCK_HISTO_SORT >

class cub::BlockHistogram
{
    struct TempStorage
    { ... };
    __device__ BlockHisogram(TempStorage& temp_storage)
    { ... };
    template <typename CounterT> \
        __device__ void Histogram(T (&items)[ITEMS_PER_THREAD],
                        CounterT histogram[BINS])
    { ... }
};  
```

- `T` è il tipo di dati (es.int)
- `BlockDimX` dice quanti thread ci sono nel blocco
- `ItemsPerThread` dice quanti dati processa ogni singolo thread (CUB è ottimizzato per far processare più di un dato a ogni thread per nascondere la latenza)
- `Bins` il numero di bin dell'istogramma
- `Algorithm` permette di scegliere come CUB calcolerà l'istogramma, di default sceglie il modo migliore per la GPU che trova.  

- Il metodo `Histogram`: (istruzione chiave)   
    `Histogram(items, histogram_buffer)`
    - prende i voti del thread `items`
    - riempie il buffer in shared memory `histogram_buffer`
    - cosa fa: inizializza la memoria, gestisce i conflitti tra thread e sincronizza tutto internamente 


Il nostro codice precedente, usando CUB diventa:  

```cpp
//buffer finale del blocco
__shared__ int block_histogram[10]; 

int cell = blockIdx.x * blockDim.x + threadIdx.x;
//prepariamo l'input per CUB
int bins[1] = {static_cast<int>(temperatures[cell]/10)};

//definiamo tipo dell'algo con aliasing 
using histogram_t =
    cub::BlockHistogram<int, block_size, 1, 10>;

//allochiamo 'tavolo da lavoro'
__shared__ typename histogram_t::tempStorage temp_storage;

//eseguiamo l'algoritmo
histogram_t(temp_storage).Histogram(bins, block_histogram);

__syncthreads();
```

