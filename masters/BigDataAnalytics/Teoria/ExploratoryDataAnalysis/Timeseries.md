# Time series

Una Time Series è una serie di data points indicizzate nel tempo e in ordine 

$$
\mathfrak{D} = \{y_{t_1},y_{t_2},...,y_{t_n}\}
$$

Una time series può essere indentificata da 4 componenti:
1. Time Range (Intervallo di tempo coperto)
2. Frequenza Campionaria (Ogni quanto vengono raccolti i dati: annuale,mensile,giornaliero,...)
3. Comportamento Deterministico
4. Comportamento Stocastico 


Una Time Series è la realizzazione di un processo stocastico $\rightarrow$ L'obiettivo sarà identificare il processo che ha generato i dati.  
- In molti casi questo si traduce nel distinguere la componente deterministica da quella stocastica.  


La parte **Deterministica (o Systematic)** è divisa in 3 componenti: 
1. **Level** Valore medio della serie 
2. **Trend** Cambiamento nella serie tra un periodo e un altro
3. **Seasonality** Un comortamento breve ciclico che può essere osservato più volte nella serie

La parte non deterministica/systematic è definita come **noise**
- _noise:_ $\rightarrow$ modellata come una variabile stocastica con media 0 e varianza costante.  



## Struttura 
Le serie si distinguono in due modelli principali:

1. **Additiva:** 
    $$
    y_t = \text{Level}+\text{Trend}+\text{Seasonality}+\text{Noise}
    $$
    In una serie additiva l'ampiezza delle osccillazioni stagionali rimane relativamente costsante nel tempo



2. **Moltiplicativa:**  
    $$
    y_t = \text{Level}\times\text{Trend}\times\text{Seasonality}\times\text{Noise}
    $$  
    In log:
    $$
    \log(y_t) = \log(\text{Level}) + \log(\text{Trend}) + \log(\text{Seasonality}) + \log(\text{Noise})
    $$
    In una serie moltiplicatival'ampiezza delle oscillazioni varia proporzionalmente al livello della serie.  

_Nota:_ Il termine Level viene spesso inglobato nel Trend $\rightarrow y_t = T_t+S_t+R_t$ quando si riporta la formula.  

L'obiettivo è modella i pattern regolari (sistematici) per poter fare previsione e contemporaneamente capire quanto è grande la componente casuale (rumore) per sapere quanto possiamo fidarci delle nostre previsioni.  
Per fare questo dobbiamo quindi **isolare** la componente sistematica e quantificare il rumore.  


### Missing Values:  
Come in un normale dataset le time series possono soffrire di mancanza di data points e outliers. 
La gestione di dati mancanti nelle serie temporali richiede più attenzione rispetto ai dataset non temporali a causa della dipendenza temporale e della presenza di componenti come trend e stagionalità.  
Nelle serie temporali, l'ordine delle osservazioni è cruciale perché c'è una dipendenza temporale: il valore di oggi può dipendere da quello di ieri, dalla stessa stagione dell'anno scorso, ecc. Questa evoluzione temporale deve essere considerata quando si trattano i dati mancanti. 

- Metodi semplici come ffill, bfill o imputazione con media/mediana generale sono spesso inadeguati se c'è trend o stagionalità
- **L'interpolazione** è un alternativa migliore perchè considera i valori vicini nel tempo, è importante scegliere un metodo di interpolazione che sia adatto alla forma del trend (es. un polinomio di grado 1 per un trend lineare, grado 2 per un trend quadratico).  

<br>

### Data Smoothing: Rolling Window

Lo scopo del data smoothing è quello di ridurre il rumore per permettere di evidenziare pattern e trend.  
Si definisce quindi una finestra di una certa ampiezza $w$ (numero di periodi)
- Questa finestra si sposta lungo la serie temporale, un punto alla volta 
- Per ogni posizione della finestra si calcola la media dei valori della serie originale che cadono all'interno di quella finestra 
- il risultato è una nuova serie temporale 

Esistono due tipi di rolling window: **Centered** e **Trailing**:  

La differenza st ain come la finestra è posizionata rispetto al punto temporale $t$ per il quale si sta calcolando la media mobile $MA_t$  

1. **Centered:**  
    - Posizionamento della finestra: La finestra è _centrata_ al punto $t$, include un numero uguale di osservazioni prima di $t$ e dopo di $t$.  
    $$
    MA_t = \frac{1}{w} \sum_{i=-k}^k y_{t+1}
    $$  
    - $k=\frac{w-1}{2}$ è il numero di punti da prendere prima e dopo $t$  
    - $y_{t+1}$ è il valore della serie originale al tempo $t+1$
    - la somma va da $t-k$ a $t+k$ includendo $w$ punti totali   
    Fornisce una stima migliore del trend sottostante al tempo $t$ perchè utilizza le informazioni attorno ad esso.  
    NON introduce un ritardo (_lag_) nella serie lisciata rispetto alla originale, il picco della lisciata coincide con il picco della serie originale.  

2. **Trailing:**  
    - Posizionamento della finestra: La finestra include il punto $t$ e i $w-1$ punti precedenti, utilizza solo informazioni passate e persenti. 
    $$
    MA_t = \frac{1}{w}  \sum_{i=0}^{w-1} y_{t+1}
    $$  
    Può essere calcolata fino all'ultimo punto della serie ed è una buona base per la previsione -> la previsione per l'ultimo periodo è semplicemente l'ultimo valore della media mobile finale.  
    Introduce però ritardo (_lag_) nella serie lisciata, si avrà che il picco della media mobile finale apparirà dopo il picco della serie originale  

**Scelta di $w$**  
- finestre più ampie espongono meglio i trend globali ($w$ grande)
- finestre più strette rilevano meglio i trend locali o cambiamenti rapidi, smussano meno ($w$ piccolo)  

Bisogna scegliere un valore adeguato di $w$ per evitare il fenomeno di under-smoothing o over-smoothing.  

### Ritardo nelle Medie Mobili: Centrata vs. Trailing

Una distinzione cruciale tra le medie mobili centrate e quelle finali (trailing) riguarda il **ritardo (lag)** che possono introdurre rispetto alla serie originale, specialmente nell'identificazione di punti di svolta come picchi e valli.

*   **Media Mobile Centrata:**
    Calcolata utilizzando un numero (quasi) uguale di osservazioni precedenti e successive al punto temporale `t`, la media mobile centrata tende a riflettere i cambiamenti e i punti di svolta della serie originale **senza un ritardo significativo**. Il picco (o la valle) della serie lisciata tende a coincidere temporalmente con quello della serie originale. Questo la rende ideale per l'analisi storica del trend.

*   **Media Mobile Finale (Trailing):**
    Calcolata utilizzando solo il punto temporale `t` e le `w-1` osservazioni precedenti, questa media mobile **introduce intrinsecamente un ritardo**. Poiché si basa unicamente su dati passati, la serie lisciata reagirà ai cambiamenti (es. l'inizio di un picco) solo dopo che questi si sono consolidati. Di conseguenza, i picchi e le valli nella serie delle medie mobili finali appariranno **temporalmente spostati in avanti (in ritardo)** rispetto a quelli della serie originale. L'entità di questo ritardo è approssimativamente `(w-1)/2` periodi, dove `w` è l'ampiezza della finestra.

<br>

---

<br>

### Modellamento delle Time Series  

Dobbiamo trovare modi per identificare e modellare la parte sistematica (prevedivile) e di rumore (non prevedibile) di una serie.  

Esistono due metodi principali:
1. **Modelli di Regressione Multipla:** si modella la relazione tra la variabile di interesse $y_t$, il tempo $t$ stesso e/o altre variabili esogene (variabili esterne es: festività, interventi di policy,...).  
2. **Modelli Autoregressivi:** Si modella la relazione rta $y_t$ e i suoi valori passati ($y_{t-1},y_{t-2},...$). L'idea è che il valore attuale della serie dipenda dai suoi valori precedenti.  


### 1. Modelli di Regressione Multipla:  
L'idea è di inferire un equazione nascosta dei dati che descriva i pattern globali.  

<br>

Se la serie ha un solo trend (assenza di stagionalità) e rumore si possono usare uno dei seguenti modelli di regressione.  

- Trend Lineare:   
    Questo modello assume che la serie cresca o descresca a un tasso costante nel tempo.  
    $$
    y_t = \beta_0 + \beta_1t + \epsilon
    $$

- Trend Esponenziale:    
    Questo modello assume che la serie cresca o decresca a un tasso percentuale costante
    $$
    y_t = ce^{\beta_1t}+\epsilon
    $$
    nota: applicando il logaritmo trasformiamo il trend esponenziale in lineare su scala logaritmica.  

- Trend Polinomiale:   
    Permette di modellare trend che non sono lineari ma che hanno una curvatura, si possono aggiungere termini di grado superiore ($t^3,t^4,...$) per curve più complesse.  
    $$
    y_t = \beta_0 + \beta_1t + \beta_2t^2 + \epsilon 
    $$

<br>

Se la serie presenta _stagionalità_, bisogna adattare i modelli e introdurre l'uso di variabili **dummy**  

La stagionalità implica che ci sia un comportamento medio diverso per ogni 'stagione' (ogni mese,anno,giorno,...)  

Si crea allora una variabile categorica stagionale (S): 
- si definisce una variabile $S$ che indica la stagioen per ogni osservazione $y_t$  
- se i dati sono mensili $S$ assume valori = {Gennaio,Febbraio,Marzo,...,Dicembre}.  

I modelli di regressione richiedono input numerici per questo trasformiamo $S$ in variabili dummy
- Se $|S| = M$ avremo $M-1$ variabili dummy, saranno una in meno rispetto alla cardinalità per evitare la multicolonearità perfetta (nel modello una dummy diventerebbe combinazione lineare perfetta delle altre). La categoria esclusa diventa la categoria di **riferimento** 

TODO: capire la multicollinearità perfetta 

- **Stagionalità Additiva:**  

$$
y_t = L_0 + \sum_{i=1}^{M-1} c_i s_i
$$


- **Stagionalità Moltiplicativa:**

$$
\log(y_t) = L_0 + \sum_{i=1}^{M-1} c_is_i
$$