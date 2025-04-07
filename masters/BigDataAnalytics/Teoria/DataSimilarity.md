# Data Similarity



### Dipendenza dal tipo di dati: 

La similarità tra dati non è universale, dipende dal tipo di **attributi** (nominali o numerici) e dal **contesto** (es: due colori possono essere considerati simili in base a specifici criteri).  

**Matrice dei Dati:**   
Formato tabellare dove le righe = oggetti e le colonne = attributi 


**Matrice di Dissimilarità:**   
Matrice Simmetrica oggetto x oggetto con valori $d(i,j)$ dove:  
- $d(i,j)=0$ se $i=j$ (diagonale principale con tutti zeri)
- $d(i,j)=d(i,j)$ (simmetria)  

La simmilarità si ottiene $\rightarrow \text{sim}(i,j) = 1 - d(i,j)$  

Es: [inserire matrice di similarità e dissimilarità]  

<br>

### Attributi Nominali: Misura di Dissimilarità:

Gli attributi nominali sono categorie **senza un ordine implicito** (es: colore blu, trattamento medico 'CS' o 'CFR',...) e la dissimilarità tra due record $i,j$ può essere espressa come:  


$$
d(i,j) = \frac{p-m}{p}
$$


$$
\text{dove } m=\sum_{l=1}^{p} \delta_{i_lj_l} \space\space\text{  e con: } \space\space\delta_{i_lj_l}=1 \space\text{se}\space i_l=j_l \space\space\text{o 0 altrimenti}
$$

- $m$ è il numero di attributi uguali tra $i$ e $j$ (i record confrontati)
- $p$ è il numero totale di attributi nominali 

<br><br>

Esempio Matrice Similarità e Dissimilarità:  


| Pz/index | Condition | Treatment |
|---|---|---|
| 1    | A    | CS    |
| 2    | B    | CFR    |
| 3    | C    | CFR    |
| 4    | A    | CS    |

<br>

Es celle notevoli:
- $d(1,4) = \frac{2-2}{2}=0$, dove $m=2$ in quanto Cond(1) e Treatment(1) sono uguali al paziente 4
- $d(2,3)=\frac{2-1}{2}=0.5$, dove $m=1$ in quanto Treatment(2) = Treatment(3) ma condizioni diverse.  
- ...

<br>


|    | 1   | 2   | 3   | 4   |
|---|---|---|---|---|
| 1  | 0   | 1   | 1   | 0   |
| 2  | 1   | 0   | 0.5 | 1   |
| 3  | 1   | 0.5 | 0   | 1   |
| 4  | 0   | 1   | 1   | 0   |



<br>

---

### Attributi Binari:  

Gli attributi binari possono assumere solo due valori e la loro analisi dipende dal contesto:  
- Simmetrici:  Entrambi i valori hanno uguale importanza (es: maschio/femmina). 
- Asimmetrici: Un valore (tipicamente 1) è più significativo dell'altro.  

**Matrice di Contingenza e Simmilarità _Simmetrica_:**    

Per due oggetti con $p$ attributi binari, si costruisce una matrice di contingenza, se si hanno più di 2 oggetti si dovranno creare matrici separate per ogni possibile coppia di oggetti.  

| $i/j$ | 1 | 0 |
|---|---|---|
| 1    |  $q$   | $r$    |
| 0    |  $s$   | $t$    |

- $q$ attributi con 1 in entrambi gli oggetti 
- $r$ attributi con 1 nel primo oggetto e 0 nel secondo
- $s$ attributi con 0 nel primo oggetto e 1 nel secondo 
- $t$ attributi con 0 in entrambi

$$
d(i,j) = \frac{r+s}{q+r+s+t}
$$  


**Attributi Asimmetrici e Coefficiente di Jaccard:**  
Negli attributi asimmetrici le corrispondenze 0-0 sono irrilevanti di natura, la dissimilarità in questi casi si calcola come:  

$$
d(i,j) = \frac{r+s}{q+r+s}
$$

**Coefficiente di Jaccard:**  

$$
J(i,j) = 1-d(i,j)= \frac{q}{q+r+s}
$$

