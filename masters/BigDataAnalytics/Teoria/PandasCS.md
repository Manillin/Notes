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
df['disciplines'].nunique()
df.nunique()
df.value_counts()
df['disciplines'].value_counts()
df.isna().sum()
df['col'].isna().sum()
df['disciplines'].count()

```

1. Restituisce il numero di occorrenze uniche in un Object, contiamo gli elementi unici in una colonna  
2. Contiamo i valori unici in tutte le colonne del df 
3. Conta le occorrenze di combinazioni uniche di tutte le righe, se applicato a un dataframe dice quante righe univoche ci sono (se il dataframe ha tutte righe diverse restituisce il dataframe originale con un '1' dopo ogni riga)
4. Applicato a una colonna, conta quante occorrenze univoche ci sono in quella colonna 
5. Conta il numero di `NaN` che appaiono in un dataframe  
6. Conta i `NaN` che appaiono in una singola colonna 
7. Conta il numero di elementi in una colonna 

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
df['colonna'].agg(['min','max','sum'])
df['salary'].map(lambda x:x*1.14)
df['colonna'].sum(axis=0,skipna=True)
df[['colonna1','colonna2']].sum(axis=0,skipna=True)
df['colonna'] = df['colonna'].str.lower()

```


1. Aggrega le funzioni specificate ai valori di una colonna
2. Mappa e applica una funzione a tutti i valori di una colonna, la funzione può essere una lambda o normalmente definita  
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





