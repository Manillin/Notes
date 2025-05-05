# Outlier detection and Feature Engineering



### Table of contents 
![]()




<br><br>

## Outlier Detection 

Un outlier è un data object che devia significativamente dal resto degli oggetti, come se fosse generato da un altro meccanismo.  
Punti che non seguono la stessa distribuzione degli altri oggetti sono chiamati **outliers** e si distingue un outlier da dati rumorosi in quanto non sono legati a errori casuali o alta varianza.  
Per questo motivo è fondamentale _**giustificare**_ gli outlier e il meccanismo che li genera, questo si ottiene facendo varie assunzioni sul resto dei dati e mostrando che gli outlier violano queste assunzioni mentre gli altri dati li rispettano.  

Gli effetti negativi degli outlier sono:
1. **Misure statistiche distorte:** Gli outlier distorgono/alternano (skew) le medie, aumentano gli errori di varianza e riducono la potenza statistica
2. **Identificazione di pattern falsi:** Gli outlier creano trend apparenti o correlazioni che in realtà non esistono, portando conseguentemente a inferenze errate 
3. **Degradazione delle performance di un modello:** Modelli di ML potrebbero overfittare sui dati in reazione agli outlier, compromettono la loro proprietà di generalizzazione.  
4. **Errori di decision making:** Decisioni di Business o Ricerca influenzati dalla presenza di outlier portano a errori costosi e pericolosi. 



### Tipi di Outlier: 

**1. Global Outlier**  
Sono punti che deviano significativamente dal resto del dataset, possono essere viste come 'anomalie' e sono il tipo di outlier più semplice.  
Per identificare global outlier bisogna trovare una misura di deviazione appropriata rispetto all'applicazione in questione.  

**2. Local Outlier**  
Sono oggetti nel dataset la cui densità devia significativamente da quella dell'area locale in cui compaiono

![global and local outlier](../../images/global_local_outlier.png)  

**3. Contextual Outlier**  
Data points che deviano significativamente rispetto al contesto specifico degli oggetti, sono conosciuti anche come **conditional outliers** in quanto sono condizionali al contesto selezionato.  
Nel problema di individuazione di outlier di contesto bisogna **specificare il contesto** come parte della definizione del problema; richiede informazioni di background.  
Es: temperatura oggi è 30 gradi -> outlier ? -> dipende dal contesto: (orario e location); se è inverno in Aosta allora si, 30 gradi sarebbe un outlier.  

**4. Collective Outlier**  
Un sottoinsieme di data objects formano un collective outlier se gli oggetti deviano significativamente dall'intero dataset.  
A differenza del rilevamento di outlier locali o globali, quando si vogliono individuare outlier collettivi bisogna considerare non solo il comportamento dei singoli dati ma quello di un gruppo di dati.  
Per rilevarli avremo bisogno delle relazioni tra data objects come la distanza o la similarità tra oggetti.   
Es: A stock transaction between two parties is considered normal, but a large set of transactions of the same stock among a small party in a short period are collective outliers because that may be evidence of some people manipulating the market.



## Metodi per rilevare Outlier 

- **Supervised methods:** Sappiamo che il nostro dataset è labeled, la detection di outlier può essere modellata come un problema di classificazione, l'obiettivo è trainare un modello in grado di rilevare outlier.  
- **Unsupervised methods:** Non abbiamo label e non sappiamo quali siano gli oggetti normali e quali gli outlier, questi metodi fanno un assunzione implcita, ossia assumono che gli oggetti normali siano in un qualche modo raggruppati (clustered) e che seguano un pattern rispetto agli outlier.  
Gli oggetti normali possono formare molteplici gruppi, dove ogni gruppo ha feature distinte, ci si aspetta di individuare outlier lontani da questi gruppi formati da oggetti normali nello spazio delle feature.  
- **Statistical methods (model-based methods):** Sono metodi unsupervised che assumono la normalità dei dati. Si assume che i dati normali siano generati da un modello statistico stocastico e che i dati che non seguono tale modello siano outlier.  
- **Proximity-Based methods:** Metodi unsupervised che assumono un outlier quando tale oggetto è lontano dai suoi vicini (nearest neighbors) nello spazio delle feature; quindi quando un oggetto devia significativamente dalla prossimita al suo vicino rispetto alla maggior parte degli oggetti dai loro vicini. L'efficienza di questo metodo dipende fortemente dalla misura di prossimità utilizzata.  


<br>




## Statistical Methods: Parametric methods

I metodi parametrici assumono che i dati normali siano generati da una distribuzione parametrica con un numero finito di parametri.  

<br>

**Univariate outliers basati su distribuzione normale:** Consideriamo un sample di una variabile scalare stocastica $x$ compatibile con una distribuzione normale, possiamo impostare i punti che sono a più di $3\sigma$ dalla media come outlier (sappiamo infatti dalla teoria che la regione $\mu\pm 3\sigma$ contiene circa il 99.7% dei dati)









```python
from scipy.stats import zscores
z_values = zscore(df[col],axis=1,nan_policy='omit')
outliers = (np.abs(z_values) > 3) # boolean outlier mask
```


<br>

---

**Multivariate outliers basati sulla normality assumption in $R^d$**:  

L'analisi multivariata è fondamentale perchè le variabili potrebbero essere **correlate** tra loro - un valore potrebbe non sembrare un outlier se analizzato da solo, ma potrebbe esserlo se analizzato nel contesto delle altre variabili.  

Si generalizza il concetto precedente al caso multivariato in $R^d$, sappiamo che per una variabile $x \in R^d$ che segue una distribuzione multivariata gaussiana vale la seguente regola:

$$
\sum_{i=1}^d z_i^2 \sim \chi^2(d) \space\space\text{dove}\space\space z = \Sigma^{-\frac{1}{2}}(x-\mu) \space\space \text{e}\space\space x \sim N(\mu,\Sigma)
$$  


dove:
- $x\in R^d$ è un vettore di $d$ variabili (un punto nello spazio delle feature a $d$ dimensioni)
- $x \sim \mathcal{N}(\mu,\Sigma)$ dice che $x$ segue una distribuzione normale multivariata dove:
    - $\mu$ è il vettore delle medie (dimensione $d\times 1$)
    - $\Sigma$ è la matrice di covarianza (dimensione $d\times d$) - misura quanto ogni coppia di variaibili varia insieme.  
- $z = \Sigma^{-\frac{1}{2}}(x-\mu)$ effettua una normalizzazione multivariata, $z$ è quindi una trasformazione di $x$ che segue una normale standard multivariata $z \sim \mathcal{N}(0,I)$
- $\sum_{i=1}^d z_i^2 \sim \chi^2(d)\rightarrow$ Se $z \in R^d$ segue una normale multivariata standard **allora** la somma dei quadrati dei suoi componenti segue una distribuzione _Chi quadro_ con $d$ gradi di libertà.  

L'ultimo punto (il fatto che $\sum_{i=1}^d z_i^2$ segua una distribuzione Chi quadrato) è **fondamentale** per rilevare outlier nel caso multivariato


**Uso della distanza di Mahalanobis:**  
Notiamo che il calcolo del quadrato delle componenti standardizzate è proprio la distanza di **Mahalanobis** al quadrato, infatti:

$$
MD^2(x) = (x-\mu)^T\Sigma^{-1}(x-\mu) = ||\Sigma^{-\frac{1}{2}}(x-\mu)||^2 = \sum_{i=1}^d z_i^2
$$

Alla luce di ciò possiamo usare direttamente la distanza di Mahalanobis per non dover calcolare esplicitamente $z$, risparmiando passaggi 

<center>

**Passaggi da Seguire:**  

</center>


Il metodo consiste dei seguenti passaggi:  
1. Si calcola la stima della distribuzione, ottenendo la media $\hat{\mu}$ e la matrice di covarianza $\hat{\Sigma}$ dai dati 
2. Per ogni punto $x_i$ si calcola la distanza di Mahalanobis al quadrato
    $$
        MD^2(x_i) = (x_i - \hat{\mu})^T\hat{\Sigma^{-1}}(x_i-\hat{\mu})
    $$  

3. Si fissa un intervallo di significatività $\alpha$, tipicamente 0.005, ossia confidenza del 95%, e questo serve a determinare la soglia oltre la quale considerare un punto un outlier.  

4. Calcola la soglia critica (quantile della distribuzione \Chi^2 con $d$ gradi di libertà):
    $$
        \text{Soglia Critica} = \chi^2_{d,1-\alpha}
    $$  

5. Confronta $MD^2(x_i)$ con la soglia:
    - Se $MD^2(x_i) \gt \chi^2_{d,1-\alpha}$ allora $x_i$ è un outlier multivariato

Questo metodo generalizza la regola di $3\sigma$ al caso multivariato.   

---

domande:
differenze tra data points, data objects, e objects  



