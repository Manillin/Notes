# Missing data: 

I dati mancanti sono un problema comune nell'analisi dei dati, soprattutto quando si lavora con dataset reali.  
I dati mancanti posso derivare da diverse situazioni: 
- **Dati corrotti o manutenzione inefficace:** Con il tempo il dataset potrebbe degradarsi e si potrebbero perdere dati in questo modo.  
- **Errori nella raccolta:** Potrebbero non essere stati raccolti correttamente, es: un sensore che malfunziona e non raccoglie i dati.  
- **Mancate risposte ai sondaggi:** Si distinguono due casi: 
    - Mancanza _intenzionale_: le persone evitano di rispondere a domande sensibili o per mancanza di conoscenza
    - Mancanza _accidentale_: Errore umano, come saltare una domanda del sondaggio per distrazione.  

L'assenza di dati non è solo un buco da riempire, essa provoca problemi e conseguenze concrete:  
- **Riduzione del campione utile:** Meno dati abbiamo e più debole sarà la potenza statistica, se il 40% dei dati è mancante l'analisi risulterebbe inaffidabile!
- **Rischi di interpretazione errata:** Se i dati mancano in modo _sistematico_ le conclusioni potrebbero essere distorte e affette da Bias (es: In uno studio medico, se i pazienti più anziani evitano di riportare effetti collaterali, il farmaco sembrerà più sicuro di quanto non sia.) 
- **Incompatibilità con algoritmi:** La maggior parte dei modelli di Machine Learning richiede input completi, dati mancanti bloccherebbero completamente il modello.  

Il problema si affronta con tre tecniche: **ELiminazione, Imputazione, Analisi della causa**.  


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

1. **Listwise Deletion:** Elimina intere righe (record) che contengono valori mancanti, utile quando il numro di righe con dati mancanti è ridotto.  
    Riduce però la dimensione del dataset, soprattutto se i dati mancanti sono diffusi.

    <br>

2. **Columnwise Deletion:** Elimina intere colonne che contengono troppi valori mancanti, utile quando un'intera colonna è molto sparsa o poco rilevante per l'analisi.  

    <br>

3. **Pairwise Deletion:** Mantiene tutte le osservazioni ma le esclude solo per le analisi 
specifiche in cui i dati mancano, evitando di ridurre cosi le dimensioni del dataset.  
    Si usa iando si vuole *massimizzare l'uso dei dati* disponibili senza eliminare intere righe/colonne.  
    Svantaggi: Potrebbe creare incoerenze in quanto il numero di osservazioni varia a seconda dell'analisi che si sta facendo (Es: si usano 100 record per la prima correlazione, e se ne usano solo 75 per la seconda (25 record hanno un dato mancante per questa analisi)).  


Queste tecninche sono molto efficienti per i dati MCAR, non introducono nessun bias, possono essere adatte anche per i dati MAR ma potrebbero anche introdurre stime biased.   


<br>


## Tecniche di Imputazione:  


### Manual Fill-in:  
Si inseriscono i dati mancanti manualmente, adatto per dataset di dimensioni piccolo-medie, tecnica costosa.  

### Fixed Value Replacement:  
Si sostituiscono tutti i missing value con una costante definita da un utente (user-defined), è adatto solo per dati mancanti di tipo MCAR