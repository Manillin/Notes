# Pandas Cheat Sheet


### Table of Contents:

1. [Creazione DataFrame](#creazione-dataframe)
2. [Visualizzare tipo di dato in un DataFrame](#visualizzare-tipo-di-dato-in-un-dataframe)
3. [Informazioni sulle colonne](#informazioni-sulle-colonne)
4. [Operazioni di count](#count)
5. [Statistiche di base ](#statistiche-di-base)
6. [Metodi del DataFrame](#metodi)
7. [Sampling di un DataFrame](#sampling-di-un-dataframe)
8. [Filtri](#filtri)
9. [Loc & Iloc](#loc--iloc)
10. [Aggregazioni, Map e Metodi di stringa](#aggregazioni-map-e-metodi-di-stringa)
11. [Group-by](#group-by)
12. [Concat e Join](#concat-e-join)
13. [Merge](#merge)
14. [Casi interessanti](#casi-interessanti)
15. [Bar Plots](#bar-plots)
16. [Box Plots](#box-plot)
17. [QQ-Plots](#qq-plot)
18. [Histograms](#histogram)

### Creazione Series:  

```python
s = pd.Series([valori_colonna],index=[indici],name='')
```

Crea una Series con i valori specificati nella colonna, se indici omessi partono da 0 (ricordarsi che valori ed indici devo avere la stessa dimensione).   
Per accedere a una Series si usa la notazione degli array, si può fare anche slicing sugli indici: `s[1:3]` 

<br>


### Creazione DataFrame:

```python
df=pd.DataFrame({'foo':[0., 4., 8., -12.,16],
                 'bar':['x', 'y', 'z', 'w','a']
                 },
                index=['A','B','C','D','E'])
print(df)
```

Si possono manualmente specificare le colonne (che sono delle Series) oppure definire in un dizionario e passare il dizionario al costruttore.  Gli indici si possono scegliere manulmante o lasciare di default.  

<br>

### Visualizzare tipo di dato in un DataFrame:

```python
df['salary'].dtypes
df.dtypes 
```

Il primo metodo restituisce il tipo di dato di una colonna; Mentre il secondo metodo restituisce tutti i tipi di dato delle colonne che compongono il dataframe.  

<br>

### Informazioni sulle colonne

```python
df.shape 
df.size
df.columns 
df.dtypes
```

1. Restituisce una tupla (x,y) con il numero di righe e il numero di colonne, se vogliamo solo uno dei due valori accediamo alla tupla `df.shape[index]`.  
2. Restituisce il numero totale di celle (righe x colonne)  
3. Restituisce il nome delle colonne
4. Restituisce il nome delle colonne insieme al loro tipo di dato (es: int64, object,...)  


<br>

### count 

```python
1. df['disciplines'].nunique()
2. df.nunique()
3. df.value_counts()
4. df['disciplines'].value_counts()
5. df.isna().sum()
6. df['col'].isna().sum()
7. df['disciplines'].count()
8. df['col'].values
9. crosstab = pd.crosstab(df['col1'], df['col2']) #.reset_index() if plot needed

```

1. Restituisce il numero di occorrenze uniche in un Object, contiamo gli elementi unici in una colonna  
2. Contiamo i valori unici in tutte le colonne del df 
3. Conta le occorrenze di combinazioni uniche di tutte le righe, se applicato a un dataframe dice quante righe univoche ci sono (se il dataframe ha tutte righe diverse restituisce il dataframe originale con un '1' dopo ogni riga)
4. Applicato a una colonna, conta quante occorrenze univoche ci sono in quella colonna 
5. Conta il numero di `NaN` che appaiono in un dataframe  
6. Conta i `NaN` che appaiono in una singola colonna 
7. Conta il numero di elementi in una colonna 
8. Restituisce gli elementi sotto forma di array numpy monodimensionale
9. `crosstab` crea una matrice di contingenza che mostra la ferequenza di combinazioni di 2 o più variabili categoriali
    ![crosstab](../../images/crosstab.png)  
    nota: se si vuole poi plottare la tabella di contingenza ricordarsi di resettare gli indici: `.reset_index()`

<br>


### Statistiche di base:

```python
colonne_target = ['col1','col2',...]
df[colonne_target].describe()
df[colonne_target].std()
df[colonne_target].mean()
```

1. Seleziono manualmente colonne su cui filtrare 
2. Restituisce un sommario delle statistiche di base che comprendono il conteggio di elementi, la media, la std, ... 
3. Calcolo la standard deviation sulle colonne numeriche scelte 
4. Calcolo la media sulle colonne numeriche scelte  


<br>

### Metodi 

```python
df.info()
dir(df)[226:]
```

1. Restituisce un sommario delle info di un dataframe (valori nulli, tipo degli elementi)  
2. Lista di tutti i metodi applicabili su un dataframe

<br>

### Sampling di un DataFrame

```python
df.sample(n)
df.sample(frac=0.1)
df[0:n]
```

1. Prende un numero casuale pari a `n` di righe dal dataframe
2. Prende la percentuale indicata di righe random dal dataframe
3. Prende le prime `n` colonne (non random)



<br>

### Filtri


```python
df['colonna'] > 100000
filtro = df['colonna'] > 10
df[filtro]
df[df['colonna'] > 10]
df[df['sex'] == 'Female']['colonna']
mask = (df['sex'] == 'Female') & (df['rank'] == 'Prof') & (df['salary'] > 120000)

```

1. Crea un sottoinsieme booleano dal dataframe dove le righe che rispettano il filtro ritorneranno `True` e le altre `False`  
2. Crea un filtro specifico con la sintassi succitata
3. Riporta solo le entry nel dataframe le cui righe hanno `True` in corrispondenza del filtro applicato, restituisce comunque tutte le colonne del df
4. Uguale a sopra ma in modalità compatta
5. Esempio con filtro per stringa, ma proietto solo la colonna indicata dopo
6. Esempio di applicazione di più filtri 


<br>

### Loc & Iloc 

```python
df.loc[s:e]
df.loc[s:e,['col1','col2',...]]
df.iloc[s:e]
```


1. Restituisce le righe del dataframe che hanno gli indici indicati tra `s` ed `e`, **Attenzione:** Loc funziona con indici, restituisce gli indici [s->e]  
2. Come sopra ma visualizzando solo le colonne specificate
3. Restituisce le righe che vanno da [s->e), indice iniziale incluso, indice finale escluso **Attenzione:** Iloc non ragiona con indici **posizionali**, non con gli indici del dataframe (se un df ha inidici che iniziano da 10... iloc[0:3] prenderà i primi 3 elementi indipendentemente dal loro indice)  

<br>

### Aggregazioni, map e metodi di stringa

```python
1. df['colonna'].agg(['min','max','sum'])
2. df['salary'].map(lambda x:x*1.14)
3. df['colonna'].sum(axis=0,skipna=True)
4. df[['colonna1','colonna2']].sum(axis=0,skipna=True)
5. df['colonna'] = df['colonna'].str.lower()

```


1. Aggrega le funzioni specificate ai valori di una colonna
2. Mappa e applica una funzione a tutti i valori di una colonna, la funzione può essere una lambda o normalmente definita. 
    - Può essere passato anche un dizionario al posto di una funzione e in tal caso map sostituisce ogni valore della colonna con il valore corrispondente nel dizionario.  
3. Somma i valori in base all'asse specificato, `axis=0` somma verticalmente lungo la colonna specificata, saltando i NaN
4. Somma i valori orizzontalmente in quanto `axis=1` **Attenzione:** Ricordarsi che quando si selezionano più colonne manualmente bisogna metterli in una lista!! (_doppie quadre dopo df_)  
5. Trasformo le stringhe della colonna in minuscolo

<br>

### Group-by

```python
df_rank = df.groupby(['rank'])
df_rank_sex = df.groupby(['rank','sex'])
df_rank.groups
df_rank
```

La clausola `groupby` permette di raggruppare i dati secondo specifi criteri (per singolo gruppo o per gruppi): Per ogni gruppo, conta tutte le combinazioni uniche di tutte le colonne.  

Restituisce un oggetto `groupby` e possiamo enumerare i gruppi con `df_rs.groups`.  
Una volta creati gruppi risulta facile fare e ottenere statistiche su di essi!   

1. Creo nuovo df in cui sono raggruppati gli oggeti per valori di 'rank'
2. Come sopra ma raggruppando per 'rank' e 'sex'
3. Restituisce come sono i gruppi e i loro indici  
    `{
        'AssocProf': [1,4,6,8,...],  
        'Prof': [9,12,45,49,...],
        ...
    }` 
    Ogni indice corrisponde alla riga del df che appartiene a tale gruppo.  
    Se raggruppati per più colonne si avrà qualcosa del tipo  
    `('gruppo1','gruppo2'): [1,5,7,9,...] ` 


<br>

### Concat e Join


```python
df_concat1 = pd.concat([df1,df2],axis=0,join='outer')
df_concat1.reset_index(drop=True)
df_concat2 = pd.concat([df1,df2],axis=1)
df_concat3 = pd.concat([df1,df2],axis=1,join='inner')
```



Permette di concatenare due o più dataframe verticalmente o orizzontalmente in base agli **indici**, il verso è specificato dalla variabile `axis`.  
L'attributo `join` definisce il comportamento della concatenazione, se `inner` allora risulterà come una intersezione insiemistica, si concatenano i df solo in corrisppndenza di indici uguali ; se `outer` allora tutti gli indici vengono inclusi nella concatenazione, vengono inseriti valori NaN dove necessario.  



1. Sintassi per la concatenazione, i parametri che si vedono sono quelli di default che comporta l'aggiunta di righe al dataframe
2. Resetto gli indici per evitare duplicati  
3. Concatenazione orizzontale, si aggiungono colonne al dataframe 
4. Concateniamo come su ma aggiungiamo le colonne del df2 sono in corrispondenza delle entry del df1 che hanno gli stessi indici; se df1 ha 100 righe e df2 ne ha solo 5, e considerando per entrambi indici che partono da 0 incrementalmente, il risultato sarà un df con solo 5 righe, **Attenzione:** Se ci sono colonne comuni a entrmbi i df quete verranno duplicate

<br>

### Merge  

```python
merge1 = pd.merge(df1,df2,on='',how='inner')
merge1 = merge1.dropna()

```

La funzione merge permette di unire due DataFrame basandosi su una o più **colonne** comuni, simile a un operazione di join SQL.  
La clausola `on` specifica le colonne su cui unire i dataframe, se non specificato verrà fatta la merge su tutte le colonne presenti in entrambi i df.  
Esistono due tipi di merge, `inner` fa il merge solo quando tutti i valori delle colonne su cui si sta facendo il merge corrispondono, se non ne trova non verrà fatto nessun merge.  
Il tipo `outer` invece fa il merge senza cercare corrispondenza tra le colonne, corrisponde a un concat sia in verticale che in orizzontale.  


1. Sintassi per il merge, con i parametri di default
2. Eliminiamo tutte le righe che contengono `NaN`


<br> 

---

<br><br>

## Casi Interessanti:  


<br>


### Uso di `apply`: 


```python
def map_list(lst):
    return pd.Series(lst)

data = pd.DataFrame({'Patient':['A','B','C'],
    'measurements': [ # occhio che è una lista di liste!
        [10, 20, 30, 40, 50],
        [15, 25, 35, 45, 55],
        [12, 22, 32, 42, 52]
    ]
})

new_columns=data['measurements'].apply(map_list)

print('\nOriginal Data')
print(data)
print(f'new col\n{new_columns}')

'''
Original Data:
  Patient          measurements
0       A  [10, 20, 30, 40, 50]
1       B  [15, 25, 35, 45, 55]
2       C  [12, 22, 32, 42, 52]

new col:
    0   1   2   3   4
0  10  20  30  40  50
1  15  25  35  45  55
2  12  22  32  42  52
'''

new_columns.columns=[f'measure_{i}' for i in range(1,new_columns.shape[1]+1)]
data=pd.concat([data[['Patient']],new_columns],axis=1)
print('\nNew Data')
print(data)

```

**Spiegazione:**  
- Si crea un dataframe Data con due colonne come illustrato dall'output
- Si vogliono trasformare gli elementi della colonna measurements in $5$ colonne separate, dove $5$ è il numero di elementi di ogni lista 
- Si crea la funzione `map_list` per trasformare una lista (elemento della colonna 'measurements') in una Series
- Quando si trasforma una lista in Serie, gli elementi della lista divenano righe, è come prendere la lista e metterla in verticale:   
    ```python
    lista = [1,2,3,4]  
    serie = pd.Series(lista)
    print(serie)
    '''
    0    1
    1    2
    2    3
    3    4
    '''
    ```
- **Attenzione:** La funzione `.apply()` prende il risultato di ogni trasformazione e lo usa come riga nel dataframe risultante  
    - Per questo motivo invece di avere la lista in verticale la avremmo sviluppata elemento per elemento in orizzontale.  
    - È una caratteristica specifica di apply: quando restituisce oggetti Serie, li combina in righe di un DataFrame invece di mantenerli come colonne separate.
    - Se si volesse mantenere l'orientamento verticale, occorrebbe fare la trasposizione (`.T`)

- Avendo creato colonne nuove con apply() queste non hanno un nome associato specifico, modifichiamo i nomi di default (RangeIndex: indici numerici da 0 a n) con nomi più significativi
    - Usiamo il metodo `.columns()` per modificarle e applichiamo una list comprehension per fare questa operazione in modo elegante.  

- Concateno il dataframe selezionando solo la colonna 'Patient' dall'originale e concatenando in orizzontale (`axis=1`) (aggiungendo colonne) le nuovi colonne, in questo modo otteniamo il dataframe nella struttura che desideravamo.  


<br><br>

---


## Bar Plots

Crea un grafico a barre, ideale per visualizzare la frequenza/conteggi di categorie discrete (es: maschio,femmina).  
- Asse $x$: mostra le categorie 
- Asse $y$: mostra la frequenza di ciascuna categoria (es:577 per maschi e 314 per femmine - ds titanic)  

```python

# Contenuto colonna 'Sex'
df['Sex'].head(2)
'''
0      male
1    female
Name: Sex, dtype: object
'''

# Value counts su colonna 'Sex' -> returns dati discreti
df['Sex'].value_counts()
'''
Sex
male      577
female    314
Name: count, dtype: int64
'''

# Creo barplot sui dati discreti per visualizzare frequenza 'Sex'
df['Sex'].value_counts().plot.bar(title='Sex')
```

![barplot sex](../../images/barplot_sex.png)


<br><br>

---


## Box Plot


Riassume la distribuzione di un insieme di dati attraverso 5 statistiche:
1. Minimo: Il valore più basso, esclusi gli outlier.
2. Primo Quartile (Q1): Il 25° percentile.
3. Mediana (Q2): Il 50° percentile, o valore centrale.
4. Terzo Quartile (Q3): Il 75° percentile.
5. Massimo: Il valore più alto, esclusi gli outlier.
 
Viene visualizzata una scatola che contiene il Q1 e Q3 con una linea che indica la mediana (Q2).  
I baffi sono gli estremi max e min esclusi gli outlier, che vengono rappresentati come cerchi.  

La lunghezza della scatola rappresenta la dispersione dei dati centrali (se la mediana è al centro della scatola i dati la distribuzione è simmetrica)  

**Raggruppamento nei Boxplot:**  
Si indica a pandas che vuoi creare boxplot separati per ciascun gruppo definito dalla colonna specificata 

```python
df['Fare'].boxplot('Fare',by='Survived')
```

![boxplot grouped by survived - fare](../../images/boxplot_fare_survived.png)

Spiegazione: `by='Survived'` crea boxplot serparati per ciascun valore unico nella colonna Survived (0 e 1), ogni boxplot rappresenta la distribuzione dei prezzi dei biglietti (Fare) per i passeggieri che sono sopravvissuti (1) e non (0).  

**Note-Osservazioni:**  
boxplot non mostrano il numero di osservazioni in ciascun gruppo. Non puoi determinare quanti passeggeri sono sopravvissuti o non sopravvissuti solo guardando il boxplot.  
I boxplot sono ideali per analizzare la distribuzione e la variabilità dei dati all'interno di ciascun gruppo.



<br><br>

--- 

## QQ-Plot

Un QQPlot confronta i quantili di due distribuzioni per verificare se hanno la stessa forma:
- se i punti seguono la retta diagonale ($y=x$) le due distribuzioni sono identiche 
- se i punti deviano dalla diagonale allora le due distribuzioni differiscono in modo sistematico

Per costruire un QQPlot si ordinano i dataset in modo crescente e si calcolano gli stessi quantili per entrambe le distribuzioni.  
I punti si generano prendendo i quantili delle due distribuzioni:
- Ascissa: quantile della prima distribuzione 
- Ordinata: quantile della seconda distribuzione  
Si tracciano i punti e la linea di riferimento e si osserva come tali punti seguono la diagonale.  

**Interpretazione dei punti:**  
- Se il punto è sulla diagonale allora le due distribuzioni hanno lo stesso valore in quel quantile 
- Se il punto è **sopra** la diagonale, allora la seconda distribuzione ha valori maggiori in quel quantile  
- Se il punto è **sotto** la diagonale, allora la seconda distribuzione ha valori minori in quel quantile  

Serve principalmente per confrontare se due distribuzioni sono simili, ma può anche essere utilizzato per vedere se una distribuzione segue una normale (si confronta la distribuzione con una distribuzione normale -> QQPlot vs Normale).  

Esempio concreto:  

![qqplot titanic tariffe uomini e donne](../../images/qqplot.png)  

Notiamo che i punti (rossi) in basso sono vicini alla linea, di conseguenza possiamo dire che per i biglietti di costo ridotto uomini e donne pagano tariffe quasi uguali.  
I punti nelle zone più alte dove le tariffe sono maggiori invece sono molto sopra la linea, il che vuol dire che le donne tendono a pagare **molto di più** per i biglietti più cari.  

Il QQplot ci permette di avere più informazioni rispetto a una semplice media (che in questo caso sarebbe males: 25.5 e females: 44.4), infatti ci fornisce informazioni anche nelle code e sugli outlier.  
Ogni punto rosso risponde alla domanda: "A parità di posizione nella distribuzione (es. mediana), chi paga di più?"


<br> <br>


--- 



## Histogram


bla bla bla placeholder per istrogramma


```python
df['col'].plot.hist()

df.plot.hist(column=['Col'], by='Col2', density=True)
```


- `by = 'Col2'`: Divide i dati in base alla colonna specificata e crea un istogramma seprato per ciascun gruppo (es. male vs females)
- `density=True`: L'asse $Y$ rappresenta la densità di frequenza, ossia normalizza l'istogramma in modo che l'area totale sia uguale a 1. Utile per cofrontare distribuzioni con diverse dimensioni di campione (es. campione1 molto maggiore rispetto campione2), e mostra la proporzione relativa di dati in ciascun intervallo.


