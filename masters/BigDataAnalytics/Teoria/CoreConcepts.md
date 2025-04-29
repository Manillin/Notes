# Core concepts: Elementi base 




### Elementi base:


**Spazio delle feature (feature space)**: È lo spazio in cui si trovano i dati che stiamo studiando, corrisponde alla dimensionalità di questi.  
Se i dati hanno una sola caratteristica (feature) siamo in 1 dimensione, se hanno 2 caratteristiche allora siamo in 2 dimensioni e cosi via.  
Ogni punto nel dataset è un _vettore_ nello spazio delle feature la cui dimensione dipende dal numero di feature.  

<br>

---

**Univariate vs Multivariate**: 
- Univariate: si analizza una sola variabile/feature (es: altezza della persone), i dati si trovano su una retta 
- Multivariate: Si analizzano più variabili contemporaneamente (es: altezza, peso, età), i dati sono punti in uno spazio d-dimensionale ($R^d$)
Effetto del passaggio al multivariabile $\rightarrow$ Bisogna considerare la correlazione tra le variabili e risulta necessario usare la covarianza e misure adeguate di distanza (non basta la deviazione standard). Inoltre la normalità non sarà più su una curca ma su una 'nuvola' di punti a forma ellittica.  

<br>

---

**Variabile stocastica:**  
È una variabile che può assumere diversi valori casuali secondo una certa distribuzione di probabilità, possiamo dire che una variabile stocastica è compatibile con una distribuzione se:
- i suoi valori sono distribuiti in modo simile a quello della distribuzione 
- se possiamo modellarla con quella distribuzione, ossia assumere che i dati provengano da tale distribuzione
es: Dire che $x$ segue una distribuzione normale significa che la stiamo modellando con una normale ossia: $x \sim N(\mu,\sigma^2)$

<br>

---


**Varianza ($\sigma^2$) e deviazione standard ($\sigma$)**:
- La varianza ($\sigma^2$) è la media dei quadrati delle distanze della media, misura la dispersione dei dati
- La deviazione standard ($\sigma$) è la radice quadrata della varianza, misura anch'essa la dispersione dei dati ma è più intuitiva in quanto ha la stessa unità di misura dei dati.  

Se i dati sono stretti (vicini) alla media allora $\sigma$ sarà piccolo, contrariamente se i dati sono lontani dalla media allora $\sigma$ sarà grande:  

- $\mu \pm 1\sigma\rightarrow$ contiene circa il 68% dei dati
- $\mu \pm 2\sigma\rightarrow$ contiene circa il 95% dei dati 
- $\mu \pm 3\sigma\rightarrow$ contiene circa il 99.7% dei dati   




**Spazi mutidimensionali: Matrice di covarianza**: In più dimensioni si adeguano i concetti visti precedentemente, oltre alla media e alla deviazione standard serve anche la matrice di covarianza. **_[WIP]_**


<br>

---

**Quantili:**  
Dividono in parti uguali la distribuzione di probabilità  
es: [10,20,30,40,50,60,70,80] $\rightarrow$ $Q_1$ = 25  
Questo è il primo quartile (25-esimo percentile) ossia quel valore sotto al quale troviamo il 25% (1/4) dei dati $\rightarrow$ alla sua destra avremo il 75% dei dati (3/4)  
_nota_: Il valore del quartile potrebbe non appartenere ai dati (come nell'esempio sopra).  

Esistono diversi metodi per calcolare i quantili, ogni metodo differisce leggermente e si potrebbero ottenere valori leggermente diversi.  

$$
P(q) = x_k + \alpha (x_{k+1} + x_k)
$$

$$
\text{posizione} = q \times (n-1) + 1
$$

$$
k = \lfloor \text{posizione} \rfloor \space\space\space\space\space \alpha=\text{posizione}-k
$$




<br>

---

### Distribuzione parametrica:

Una distribuzione parametrica è una distribuzione di probabilità che può essere descritta completamente da un numero finito di parametri.  
Es $\rightarrow$ La distribuzione normale/gaussiana è parametrica e si descrive con due parametri:  
- $\mu$: la media, ossia il centro della distribuzione 
- $\sigma$: la deviazione standard, cioè quanto si allarga la distribuzione attorno alla media

