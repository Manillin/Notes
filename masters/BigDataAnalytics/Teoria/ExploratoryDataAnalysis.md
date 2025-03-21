# Exploratry Data Analysis


### Table of Contents:

1. [Data Types](#data-types)
2. [Central Tendency](#central-tendency)
3. [TBD]()



<br> <br>


## Data Types  

1. ### Qualitative Data: 
    - **Attributi Categorici (nomimali)**: Rappresentano nomi e categorie.  
    es: cane $\gt$ gatto $\rightarrow$ non ha senso
    - **Attributi Ordinali**: Si hanno con classi che hanno un ordine implcito.  
    es: basso, medio, alto $\rightarrow$ ordine chiaro ma non quantificabile il singolo elemento della classe.   

2. ### Quantitative Data:
    - **Attributi su scala a intervalli**: Sono misurati su una scala composta da unità di uguali dimensione.  
    Il **Ranking** e la **differenza** tra i valori sono significativi mentre la **moltiplicazione e rapporti** non lo sono, poichè manca un vero punto zero 
    - **Attributi su scala a rapporti**: Sono attributi numerici con un punto zero intrinseco.  
    Il **Ranking, differenza, moltiplicazioni e rapporti** sono tutti significativi.  
    Es: denaro, velocità, temperatura in Kelvin($K$).  


<br>

## Central Tendency:  

Consideriamo un attributo $X$ e un set $\{x_i\}_{i=1,...,N}$ come insieme di $N$ valori osservati.  

### Media Aritmetica: 

$$
\bar{x} = \frac{1}{N} \sum_{i=1}^{N}x_i  
$$  

Nel caso non tutti i valori siano ugualmente significativi, si può definire un set di pesi $\{w_i\}_{i=1,..,N}$ per ogni valore e calcolare: 

### Media aritmetica pesata:  

$$
\bar{x} = \frac{1}{N} \sum_{i=1}^{N}w_i x_i  
$$


### Mediana 

Se ordiniamo i valori osservati in ordine crescente $\{\bar{x_i}\}_{i=1,...,N} \space\space\space\space \bar{x_i} \lt \bar{x_{i+1}}$  

$$
x_{50} = \begin{cases}
1. & \bar{x_{(N+1)/2}} \space\space\space\space N\text{dispari}\\
2. & \frac{x_{N/2} + x_{(N+1)}}{2} \space\space\space\space N\text{pari}
\end{cases}  
$$

<br>

La mediana è robusta rispetto agli **outlier**, ma può essere costosa da calcolare per grandi valori di $N\rightarrow$ serve ordinamento.  

**Outlier:** Sono valori anomali o estremi che si discostano significativamente dagli altri dati in un insieme. Ad esempio, in un insieme di dati come [1, 2, 3, 1000], il numero 1000 è un outlier.  

Per grandi valori di $N$ (cioè quando il numero di dati è molto elevato), calcolare la mediana esatta può essere computazionalmente costoso. Per questo motivo, si utilizza un'approssimazione basata sulla funzione di distribuzione cumulativa (CDF).


$$  
x_{50} = L_{50} + \left( 0.5 - CD(L_{50}) \right) \cdot \frac{U_{50} - L_{50}}{CD(U_{50}) - CD(L_{50})}  \space\space\space\space\space\space CD(y) = \frac{1}{N} \sum_{i=1}^{N} \delta_{x_i < y}  
$$  


- $x_{50}$ è la mediana approssimata
- $L_{50}$ limite inferiore dell'intervallo contenente la mediana 
- $U_{50}$ limite superiore dell'intervallo 
- $CD(y)$ funzione di distribuzione comulativa



### Moda:  

La moda è il valore $x$ dove la PMF (Probability Mass Function) di $X$ prende il valore massimo - nel caso di variabili aleatorie discrete.  
Nel caso di variabili aleatorie continue la moda è quel valore che massimizza la PDF $f(x)$, in altre parole:
$$
\text{Moda} = \text{arg}\space\space {max}_x f(x)  
$$



### Relazioni:

Può essere dimostrato che per una distribuzione unimodale si ha questo comportamento: 

$$
|\bar{X} - X_{50}| \leq 0.77\sigma  
$$

$$
|\text{mode}(X) - X_{50}| \leq 1.73\sigma  
$$

![](../../images/moda_media_mediana_skewed.png)

Queste formule indicano che la media tende a essere più influenzata da valori estremi (outlier) rispetto alla mediana e alla moda.

<br>

**Recall: Deviazione Standard**  

$$
\sigma = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(x_i - \bar{X})^2}   
$$

La deviazione standard $\sigma$ misura la dispersione dei dati, se $\sigma$ è grande significa che i dati sono molti dispersi.   



---

### Covarianza:

$$  
\text{Cov}(A, B) = \mathbb{E} \left[ (A - \bar{A})(B - \bar{B}) \right] = \frac{1}{N} \sum_{i=1}^{N} (a_i - \bar{A})(b_i - \bar{B}) = \left( \frac{1}{N} \sum_{i=1}^{N} a_i b_i \right) - \bar{A} \bar{B}  
$$  


---

### Person's Correlation: 

$$  
r_{A,B} = \frac{1}{N \sigma_A \sigma_B} \sum_{i=1}^{N} (a_i - \bar{A})(b_i - \bar{B}) \space\space\space\space \text{values bounded to} -1 \leq r_{A,B} \leq 1  
$$  







---

### Correlazione $\chi^2$ di Pearson: 

$$
\chi^2 = \sum_{i=1}^{\alpha}\sum_{j=1}^{\beta} \frac{(M_{ij}-E_{ij})^2}{E_{ij}}  
$$



