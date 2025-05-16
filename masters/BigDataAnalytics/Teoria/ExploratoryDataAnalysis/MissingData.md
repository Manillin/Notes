# Missing data: 

I dati mancanti sono un problema comune nell'analisi dei dati, soprattutto quando si lavora con dataset reali.  
I dati mancanti posso derivare da diverse situazioni: 
- **Dati corrotti o manutenzione inefficace:** Con il tempo il dataset potrebbe degradarsi e si potrebbero perdere dati in questo modo.  
- **Errori nella raccolta:** Potrebbero non essere stati raccolti correttamente, es: un sensore che malfunziona e non raccoglie i dati.  
- **Mancate risposte ai sondaggi:** Si distinguono due casi: 
    - Mancanza _intenzionale_: le persone evitano di rispondere a domande sensibili o per mancanza di conoscenza
    - Mancanza _accidentale_: Errore umano, come saltare una domanda del sondaggio per distrazione.  

L'assenza di dati non è solo un buco da riempire, essa provoca problemi e conseguenze concrete:  
- **Riduzione del campione utile:** Meno dati abbiamo e più debole sarà la potenza statistica, se il 40% dei dati è mancante l'analisi risulta inaffidabile!
- **Rischi di interpretazione errata:** Se i dati mancano in modo _sistematico_ le conclusioni potrebbero essere distorte e affetti da Bias (es: In uno studio medico, se i pazienti più anziani evitano di riportare effetti collaterali, il farmaco sembrerà più sicuro di quanto non sia.) 
- **Incompatibilità con algoritmi:** La maggior parte dei modelli di Machine Learning richiede input completi, dati mancanti bloccherebbero completamente il modello.  

Il problema si affronta con tre tecniche: **Eliminazione, Imputazione, Analisi della causa**.  


<br>

--- 




### Tipologie di dati mancanti 

Quando lavoriamo con dati incompleti, è fondamentale capire **perché quei dati mancano**. Le tre categorie principali _(MCAR, MAR, MNAR)_ descrivono il meccanismo alla base della mancanza, e questa distinzione determina come possiamo (o non possiamo) gestirli senza introdurre errori.  

1. **MCAR(Missing Completely At Random):** Sono dati che mancano in modo completamente casuale, senza alcun legame con altre variabili osservate o non osservate.  
    L'analisi statistica non risulta distorta (unbiased) perchè i dati mancanti non introducono pattern nascosti  

2. **MAR(Missing At Random):** La probabilità che il dato manchi dipende da altre variabili _osservate_ ma non dalla variabile mancante stessa o da variabili non osservate.   
    Possiamo correggere la variabile mancante usando le variabili correlate alla mancanza.  
    (Es:In uno studio clinico, i pazienti anziani (età >70) hanno maggiori probabilità di saltare la misurazione della pressione sanguigna a causa della fatica.  
    La mancanza è legata all’età (variabile osservata), ma non al valore specifico della pressione (variabile mancante).  
    Se conosciamo l’età dei pazienti, possiamo usarla per imputare i valori mancanti in modo accurato!!)  
    $\rightarrow$ se i dati mancano più spesso in certi gruppi ma non dipendono dal valore mancante stesso allora si tratta di MAR
3. **MNAR(Missing Not At Random):** La mancanza del valore è legata al valore mancante stesso o a variabili non osservate!  
    Implica un analisi distorta (_biased_) perchè il motivo della mancanza è il dato stesso e non è correggibile con i dati disponibili.  
    (Es: In un sondaggio sul reddito, le persone con stipendi molto alti o molto bassi potrebbero rifiutarsi di rispondere. La mancanza dipende proprio dal reddito (variabile mancante), che non osserviamo per chi non risponde.  
    Anche se abbiamo altre variabili (es. istruzione), non possiamo stimare il reddito mancante senza fare assunzioni rischiose.)  

nota: La mancanza dipende da qualcosa che vedo (MAR), da qualcosa che non vedo (MNAR), o è davvero casuale (MCAR)?

<br>

---


### Gestione di dati mancanti:  

Quando si hanno dati mancanti in un dataset, una delle strategie più immediate è quella di eliminare le osservazioni o le variabili problematiche.  

1. **Listwise Deletion:** Elimina intere righe (record) che contengono valori mancanti, utile quando il numero di righe con dati mancanti è ridotto.  
    Riduce però la dimensione del dataset, soprattutto se i dati mancanti sono diffusi.

    <br>

2. **Columnwise Deletion:** Elimina intere colonne che contengono troppi valori mancanti, utile quando un'intera colonna è molto sparsa o poco rilevante per l'analisi.  

    <br>

3. **Pairwise Deletion:** Mantiene tutte le osservazioni ma le esclude solo per le analisi 
specifiche in cui i dati mancano, evitando di ridurre cosi le dimensioni del dataset.  
    Si usa quando si vuole *massimizzare l'uso dei dati* disponibili senza eliminare intere righe/colonne.  
    Svantaggi: Potrebbe creare incoerenze in quanto il numero di osservazioni varia a seconda dell'analisi che si sta facendo (Es: si usano 100 record per la prima correlazione, e se ne usano solo 75 per la seconda (25 record hanno un dato mancante per questa analisi)).  


Queste tecninche sono molto efficienti per i dati MCAR, non introducono nessun bias, possono essere adatte anche per i dati MAR ma potrebbero anche introdurre stime biased.   


<br>


## Tecniche di Imputazione:  


### Manual Fill-in:  
Si inseriscono i dati mancanti manualmente, adatto per dataset di dimensioni piccolo-medie, tecnica costosa.  

### Fixed Value Replacement:  
Si sostituiscono tutti i missing value con una costante definita lato user(user-defined), è adatto solo per dati mancanti di tipo _MCAR_.  

### Min-Max Replacement:  
Utile quando si hanno casi in cui si conosce la misura dopo la quale i dati non vengono registrati, si può usare il range min - max pe rsostituire i dati mancanti in questo caso (molto specifico, non adatto a tutte le situazioni).  

### Mean-Median-Mode Imputation:
Si sostituiscono i dati mancanti con uno dei tre parametri tra media (se sono presenti outlier questa tecnica non sarà efficace), mediana e moda (utile nel caso di dati di tipo categorico).  

### Forward/Backward imputation:
Si sostituiscono le celle vuote con i dati delle celle adiacenti verso il basso o verso l'altro, utile solo in caso di dati panel (dati ripetuti sugli stessi oggetti nel tempo).  


### Metodi di Interpolazione:  
Si rimpiazza ogni missing value con un valore ottenuto da un metodo di interpolazione (lineare, quadratico o polinomiale), particolarmente efficiente per dati numerici continui di tipo panel/temporanei.  

### Metodi Multivariati di Inputazione:  
Si inseriscono i valori mancanti con il valore più probabile, tale valore può essere derivato applicando tool di ML.  

**1. Iterative Imputation:**  
Tecnica per gestire dati mancanti quando esiste una correlazione tra loro, questa sfrutta le relazioni tra le colonne per stimare i valori mancanti in modo più accurato.  




**2. kNN Imputation:**  
il k-NearestNeighbors imputation è una tecnica di imputazione multivariata che utilizza la similarità tra i dati per stimare i dati mancanti, si basa sull'ipotesi che alcuni punti hanno dati simili.  

- È un algoritmo che trova i $k$ data points più simili a quello con il valore mancante e usa i loro valori per imputare il dato mancante.  
- La similarità è misurata dal concetto di _distanza_ (calcolata sulle feature disponibili)  

![kNN imputation](../../images/kNN_imputation.png)

Nell'esempio riportato si vuole calcolare il valore $V2$ per la entry $Y1$  
- si prendono le altre entry della tabella che hanno $V2$ noto ($Y6$ è escluso)  
- si calcola la distanza tra $Y1$ e gli altri punti usando **solo** le feature _NON_ mancanti ($V1$, $V3$, $V4$, $V5$) usando la formula ($N$ sono il numero totale di feature ed $n$ sono le feature non mancanti per $x$ e $y$)

$$
d(x,y) = \sqrt{w \sum_{i=1}^n (x_i - y_i)^2},\space\space\space\space w = \frac{N-1}{n}
$$

$$
\text{Applicata a Y1 e Y2 diventa:} \space\space\space\space
$$

$$
d(Y1,Y2) = \sqrt{\frac{4-1}{4} [(0.11-0.12)^2 + (0.42-0.25)^2 + (0.25-0.12)^2 + (0.12-0.53)^2]} = 0.463
$$  


- Si calcola la distanza tra il target e tutti gli altri punti e si selezionano (in questo  caso con $k=2$ punti) i $k$ punti più vicini
- Da questi 2 punti si imputa il nuovo valore facendo al media e tale sarà il nuovo valore del precedente missing value.  


---

<br><br><br>
 

## Osservazioni sui metodi di imputazione:  

- Gli approcci univariati e multivariati visti fin'ora trattano i dati imputati come veri, ignorando il fatto che nessun metodo di imputazione possa fornire il valore esatto.  
Non riflettono quindi la 'incertezza' dei missing values.  
Gli approcci che risolvono questo problema sono i **multiple imputation methods**, che creano varie imputazioni per ogni dato mancante (più accurati ma più computazionalmente pesanti).    

- L'imputazione univariate sono applicabili solo per piccole quantità di dati **MCAR**, in quanto ignorano le relazioni tra variabili, se usati con MAR o MNAR introducono bias.  
- L'imputazione multivariate sono applicabili per dati **MCAR e MAR**, hanno il vantaggio di preservare le relazioni tra variabili.  
- L'imputazione multipla va usata quando i dati mancanti sono numerosi, genera più dataset (che dovremo poi aggregare) per quantificare incertezza 


### Aspetto Fondamentale: 
Prima di applicare qualsiasi metodo di imputazione multivariato è _**ESSENZIALE**_ riscalare le feature per evitare che le differenze nelle scale distorcano i risultati.  
- se le variabili hanno scale diverse (età (1-100) e reddito (0-1000000)), algoritmi come la kNN o regressione daranno più peso alle feature con valori più grandi.  
    - la distanza euclidea tra due punti usando l'esempio sopra sarà dominata dal reddito, rendendo l'età irrilevante
- **Soluzione:** La soluzione è standardizzare o normalizzare le feature (z-score o min-max).  



