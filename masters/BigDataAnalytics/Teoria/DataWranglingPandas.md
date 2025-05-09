# Data Wrangling with Pandas 


### Ripasso concetti generali: 

**Tabella:** Ha una colonna speciale `ID` o `Key`, il resto delle colonne sono attributi o variabili; ogni riga sono osservazioni o instanze.  

**Operazioni su Tabelle:**
1. `Slicing`: Selezionare solo alcune delle colonne o delle righe
2. `Aggregate`: Permette di aggregare dati (sum, max, ...) su una colonna (farlo su una riga non avrebbe molto senso)  
3. `Map`: Applica una funzione ad ogni riga, possibilmente creando più righe o meno righe.  
4. `Group-By`: Permette di ragruppare tuple per colonna o dimensione; può essere combinato con una `aggregate` per fare aggregazioni sulle tuple che risultano dalla clausola di group-by.  
5. `Union-Intersection-Difference`: Operazioni che crano nuove tabelle solo se le due hanno stesso dominio (stesse righe e colonne)
6. `Merge - Join`: Combina righe / tuple attraverso diverse tabelle se hanno la stessa chiave

Per maggiori dettagli ed esempi grafici controllare il primo pacco di slide da pagina 1-23

<br>

# Introduction to Pandas: 

### Series:

È un array monodimensionale con indici, può avere qualsiasi tipo di contenuto nel field data.

```python
import pandas as pd 
s = pd.Series([5,6,12,-5,6.7], index= ['A','B','C','D','E'])
```

Si possono creare nei seguenti modi:  

```python
d = {'b':1,'a':0,'c':2}
s = pd.Series(d)
s2 = pd.Series(5.,index=['a','b','c','d','e'])

# --- Res: 

b 1 
a 0 
c 2 
```

Le serie sono manipolabili come array o dizionari $\rightarrow$ `s[0]` o `s['a']` per accedere agli elementi a partire dagli indici.  

Per maggiori dettagli consultare la [documentazione Series](https://pandas.pydata.org/docs/reference/api/pandas.Series.html)


### Data Frame:  

È una struttura con label a due dimensione, stile matrice dove ogni colonna può avere un tipo diverso di dato con indice diverso.  

```python
DataFrame(data,index=index,column=column)

ds = pd.DataFrame({'foo':[0.,4.,8.,-12.,16], 'bar':['x','y','z','w','a'],...}, index=['A','B','C','D','E'])
```
Il risultato ha questa struttura: 

![dataframe](../../images/dataframe.png)

Può anche questo essere creato da un dizionario di serie nel seguente modo:

```python
d = {'one' : pd.Series([1., 2., 3.], index=['a', 'b', 'c']), \
    'two' : pd.Series([1., 2., 3., 4.], index=['a', 'b', 'c', 'd'])}

#--- Res
    'one'   'two'
a    1.0     1.0
b    2.0     2.0
c    3.0     3.0
d    NaN     4.0

```

Quando mancano dei dati il sistema fa automaticamente padding e inserisce dei `NaN` (not a number) in corrispondenza dei valori non definiti.  

Possiamo crearlo anche da un dizionario di dizionari


```python 
ds=pd.DataFrame({'foo':{'a':1,'b':1},'fee':{'a':1,'b':2}})

#--- Res
    'foo'   'fee'
a    1        1
b    1        2
```

Consultare la documentazione per gli altri metodi di creazione.  


### Lettura e Manipolazione di dati con pandas

Pandas è in grado di leggere dati in diverso formato, rifarsi alla [documentazione](https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html) per maggiori dettagli.  


Possiamo esplorare il DataFrame importato :

```python
df = pd.read_csv("....")
# list first 5 records
df.head()
```

```python
# restituisce il tipo di dato a cui accediamo con la chiave specificata 
df['salary'].dtype

# restituisce il tipo di dato di tutte le colonne 
df.types 
```

Fare riferimento al ipynb per visualizzare i comandi interattivamente  

### Data Types:
- `Object`: Corrispode a `string` in python (assegnato di default se ha numeri e caratteri)
- `int64`: Corrisponde a `int`, allora 64b di memoria per questo intero
- `float64`: Analogo a sopra ma per i `float`
- `datetime64,timedelta[ns]`: Non esiste una corrispondenza in pyrhon ma simile al modulo `datetime`.   



## Uso di Filtri sul DataFrame:

Possiamo filtrare la tabella per ottenere solo le informazioni che voglioamo, per fare ciò si usano i **Filtri Booleani**.  
Questi ritorneranno un T o F e selezioneremo solo gli elementi della tabella conformi al tipo di ritorno che vogliamo (T).  

```python
filtro = df['salary'] > 120000 # seleziono gli elementi da salary con valori > 120k

df[filtro] # per visualizzarli 
df[ df['salary'] > 120000 ] # stessa cosa ma inline

```

Si possono anche concatenare più filtri per ottenere un risultato più specifico con l'operatore `&`, racchiuedere le condizioni multiple tra parentesi tonde `()`:

```python
'female professors with salary over 120k'
df_fsalary_over120 = df[(df['salary'] > 120000) & (df['sex'] == 'Female')]
print(df_fsalary_over120.head()) #stampo prime 5 entry 
```

**Nota:** Quando si applicano filtri si vede l'intero DataFrame, se si vuole visualizzare solo una tabella la si può proiettare con l'operatore `[]`, se se ne vogliono visualizzare di più ricordati che devi annidare le `[]`

```python

# --- print() omessi

#visualizzare solo 'salary'
df[df['sex'] == 'Female']['salary'] 

# visualizzare colonne 'salary' e 'sex'
df[df['sex'] == 'Female'][['salary','sex']]

```

<br>

### loc vs iloc:


- `loc[i:j]`: Seleziona i dati in base alle etichette degli indici, gli estremi sono **_inclusi_** 

- `iloc[i:j]`: Seleziona i dati in base alla posizione (come se fosse un array), l'estremo superiore è **_escluso_** mentre quello inferiore è incluso.  


Esempio pratico 

```python
data = {'valore':[10,20,30,40,50,60]} 
df = pd.DataFrame(data, index=[1,2,5,9,12,19])  

# DataFrame Risultante: 

'''
    valore
1      10
2      20
5      30
9      40
12     50
19     60

'''

df.loc[0:5]

#Risultato: cerca di selezionare le righe con indice compreso tra 0 e 5
'''
   valore
1      10
2      20
5      30

'''

df.iloc[0:5]

#Risultato: Prende le righe dalla posizione 0 alla 4 indipendentemente dai loro indici (si guarda solo la posizione delle righe).  
'''
    valore
1      10  # posizione 0
2      20  # posizione 1
5      30  # posizione 2
9      40  # posizione 3
12     50  # posizione 4

'''
```

**In breve:**


| **Metodo** | **Basato su** | **Inclusione estremi** | **Esempio** |
|------------|-------------|-----------------|-----------|
| `.loc[]`  | **Indice** | Inclusi | `df.loc[1:5]` prende le righe con indice 1, 2 e 5 |
| `.iloc[]` | **Posizione** | Escluso superiore | `df.iloc[0:5]` prende le prime 5 righe |



### Common Functions - Data Frame methods:


| **Funzione** | **Descrizione** |
|-------------|---------------|
| `count()` | Numero di valori non NaN |
| `sum()` | Somma dei valori |
| `mean()` | Media dei valori |
| `mad()` | Deviazione assoluta media |
| `median()` | Mediana |
| `min()` | Minimo |
| `max()` | Massimo |
| `mode()` | Moda (valore più frequente) |
| `abs()` | Valore assoluto |
| `prod()` | Prodotto dei valori |
| `std()` | Deviazione standard |
| `var()` | Varianza |
| `sem()` | Errore standard della media |
| `skew()` | Asimmetria della distribuzione |
| `kurt()` | Curtosi (appiattimento della distribuzione) |
| `quantile()` | Quantile di un valore % |
| `cumsum()` | Somma cumulativa |
| `cumprod()` | Prodotto cumulativo |
| `cummax()` | Massimo cumulativo |
| `cummin()` | Minimo cumulativo |


### Aggregazione e Riduzione in un DataFrame  


I metodi del DataFrame (alcuni sopra elencati) prendono **due** parametri: 
1. `axis`: può valere `0` o `1`:  
    - axis=0 $\rightarrow$ operazione verticale 
    - axis=1 $\rightarrow$ operazione orizzontale

2. `skipna`: esclude dati che hanno nel loro campo `NA`, può essere impostato a True o False.   

Se sono richieste funzioni lungo un asse del df si può usare `apply()`, se invece si vogliono fare più aggregazioni ma farle nella stessa istruzione, si usa `aggregate()`.  

Aggregare significa applicare una funzione su un gruppo di dati per ottenere un dato riassuntivo, può avvenire su righe o su colonne.    



```python
df = pf.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6],
    'C': [7, 8, 9]    
})

# --- Res

'''
   A  B  C
0  1  4  7
1  2  5  8
2  3  6  9
'''

# --- Aggregazione Righe 

df.sum(axis=0) 

'''
A 6
B 15
C 24
'''
```

_Aggregare lungo le righe_ significa che le righe vengono percorse **verticalmente** e i valori vengono sommati per **colonna**.   
L'operazione percorre tutte le righe di una colonna e le riassume in un solo valore (aggregandole quindi).  


```python

# --- Aggregazione colonne 

df.sum(axis=1)

'''
0 12 
1 15
2 18
'''
```

_Aggregare lungo le colonne_ significa che le colonne vengono percorse orizzontalmente e i valori vengono sommati riga per riga.  
L'operazione attraversa tutte le colonne di una riga e le riassume in un solo valore.  

Notare la differenza di indice usato dall'operazione di aggregazione, per axis=0 si usano gli indici di colonna, per axis=1 si usano gli indici di riga.  


## Map e metodi di stringhe 

Esistono due varianti del metodo `map`:
- `applymap` funziona element wise su un Dataframe
- `map` funziona element wise su una Series 

es:
```python
def dollarConvert(x):
    return x*1.14
df['dollar'] = df['salary'].map(dollarConvert)
```

I metodi di stringhe sono i seguenti:
- `str.lower(), str.upper(), str.len(), str.split()`  
- `df['sex'] = sf['sex'].str.lower()` $\rightarrow$ trasforma le entry della colonna sex in minuscolo.  

## Groupby: 

Il metodo `groupby` permette di splittare i dati del dataframe in gruppi secondo specifici criteri (singoli o multipli)

```python
# raggruppo usando il 'rank' (ruolo)
df_rank = df.groupby(['rank'])

# raggruppo secondo 'rank' e 'sex'
df_rs = df.groupby(['rank','sex'])
```

Questo metodo restituisce un oggetto **`groupby`**, possiamo enumerare i gruppi presenti sotto forma di dizionario con `df_rs.groups`.  

Una volta che abbiamo creato un gruppo, risulta facile ottenere statistiche su di essi:

```python
# fare un gruppo per tipologia di prof e calcolare la media del loro salario

df_rank = df.groupby(['rank'])
df_rank[['salary']].mean()
```


## Concat e Merge

Permette di concatenare due o più dataframe verticalmente o orizzontalmente, il verso è specificato dalla variabile `axis` 

Sintassi $\rightarrow$ `pd.concat([list of DataFrames],join=)`

- **concatenazione verticale**:  
    `axis=0` $\rightarrow$ se si hanno 2 DataFrame con le stesse colonne, concat li unirà aggiungendo le righe del secondo DataFrame sotto quelle del primo.

- **concatenazione orizzontale**:  
    `axis=1` $\rightarrow$ concatena lungo le colonne. Usa questo quando vuoi aggiungere colonne da un DataFrame all'altro.  

- **Inner join** Vengono concatenati i df solo in corrispondenza di indici uguali, vengono lasciate fuori le righe con indici presenti in una e assenti nell'altro df (Simile a un **intersezione insiemistica**)

- **Outer join** (default) Tutti gli indici vengono inclusi nella concatenazione, nella tabella che non ha un indice presenta nell'altra vengono inseriti NaN lungo quella riga (Simile ad un **unione insiemistica**)

Comportamento default: concatenazione verticale ed include tutti gli indici (unione insiemistica) inserendo NaN dove necessario  
- `pd.concat([df1,df2], axis=0, join='outer')`


---

<br>

La funzione **merge** è utilizzata per unire due DataFrame basandosi su una o più **colonne** comuni, simile a un'operazione di join in SQL.  

Sintassi $\rightarrow$ `pd.merge(left, right, on=, how=, ...)`  

- Colonne di unione:  
    - `on`: Specifica le colonne su cui unire i DataFrame. Se non specificato, merge utilizza l'intersezione dei nomi delle colonne.

- Tipi di join:  
    - `how='left'` Mantiene tutte le righe del DataFrame di sinistra e aggiunge i dati corrispondenti dal DataFrame di destra.
    - `how=right` Analogo ma verso il dataframe di destra.  
    - `how=outer|inner`La prima opzione mantiene le righe da entrambi i df (mettendo NaN) mentre la seconda opzione mantiene le righe con valori corrispondenti in entrabi i df.  


Es:  
students_df:

| StudentID | Name | Age |
|-----------|---------|-----|
| 1 | Alice | 20 |
| 2 | Bob | 21 |
| 3 | Charlie | 22 |  

grades_df:  
| StudentID | Course | Grade |
|-----------|---------|-------|
| 1 | Math | A |
| 1 | Science | B |
| 2 | Math | B+ |
| 3 | Science | A- |
| 4 | Math | C |

```python
merged_df = pd.merge(students_df, grades_df, on='StudentID', how='inner')

print(merged_df)

```
Risultato:  
| StudentID | Name    | Age | Course  | Grade |
|-----------|---------|-----|---------|-------|
| 1         | Alice   | 20  | Math    | A     |
| 1         | Alice   | 20  | Science | B     |
| 2         | Bob     | 21  | Math    | B+    |
| 3         | Charlie | 22  | Science | A-    |


<br>

---


