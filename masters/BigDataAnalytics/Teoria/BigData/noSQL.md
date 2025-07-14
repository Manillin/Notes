# No SQL



### Background


Elementi positivi dei tradizionali RDBMS:
1. Possono rappresentare relazioni nei dati 
2. Disk oriented storage
3. Indexing Structures 
4. Multi-Thread per nascondere la latenza 
5. Meccanismo di locking per garantire consistenza

Collegare RDBMS ad applicazioni web-based diventa problematico (a partire dall'esplosione dei social media che richiedono grandi quantita di dati).  

Sappiamo che i tradizionali RDBMS assicurano le garanzie ACID, ossia:
- A: **Atomicity** $\rightarrow$ Ogni operazione vinene eseguita interamente o per niente, se una parte della transazione fallisce si annulla tutto.  
- C: **Consistency** $\rightarrow$ Una transazione deve portare il DB da uno stato valido ad un altro stato valido, rispettando tutte le regole e vincoli pre-stabiliti
- I: **Isolation** $\rightarrow$ Le transazioni concorrenti non interferiscono tra di loro, come se ogni transazione fosse eseguita in isolamento  
- D: **Durability** $\rightarrow$ Una volta che una transazione è stata confermata i suoi effetti sono permanenti, anche in caso di crash del sistema.  

Il modo migliore per garantire i principi ACID e un modello di query ricco (e complesso) è quello di avere l'intero dataset su una singola macchina.  
Questo comporta limiti sulla scalabilità verticale, scalare in verticale quando si hanno moli di dati giganti è troppo costoso e non abbordabile.  

La scelta ottimale sarebbe quella di scalare in orizzontale, aggiungendo macchine meno potenti e poco costose per distribuire il carico di lavoro.  

**Sharding:**  
Lo Sharding o Partitioning consiste nello scalare orizzontalmente porzioni del database, scala bene per le read e per le write, _tuttavia_:
- I tradizionali RDBMS sono difficili da scalare orizzontalmente in quanto bisogna garantire i principi ACID e diventano complicate alcune operazioni:
    1. Fare una Join attraverso partizioni differenti è costoso 
    2. Difficile garantire l'integrità referenziale tra shard diverse 
    3. Non è trasparente, le applicazioni devono essere 'partition-aware'

<br>

## NoSQL:

Key-Feature dei modelli NoSQL:
- Non-relazionali 
- Non richidono uno schema 
- I dati sono duplicati in molteplici nodi (dati identici e fault tolerant) e possono essere partizionati: non c'è un SPOF e i nodi sono facilmente rimpiazzabili
- Scalabile orizzontalmente
- Poco costoso e di facile implementazione (OS)
- Performance massive per le operazioni di write 


Svantaggi:
- Non supportano completamente le feature relazionali: Operazioni di `join`, `group by`  e `order by` sono possibili solo all'interno della stessa partizione.
- Non sono supportati da un linguaggio dichiarativo, è richiesto un linguaggio più programmatico
- Le clausole ACID sono rilassate, si hanno meno garanzie (CAP theorem)
- Non facilmente integrabili con altre applicazioni che supportano in partenza SQL

<br>

### NoSQL Models

1. **Key-Value:**  
    Il focus è quello di scalare per grandi quantità di dati, il design di questo paradigma è stato pensato per gestire quantità massive di dati (basato su paper Dynamo di Amazon) -> progettato per assicurare performance predicibili e scalabili per carichi di lavoro (workload) non predicibili.  
    Composto da:
    - Tabelle $\rightarrow$ collezione di data items
    - Items $\rightarrow$ gruppo di attributi univocamente identificabile dalla primary key
    - Key $\rightarrow$ esistono due tipi di key:  
    **simple key (partition key)** è il valore che dato a una funzione hash mi retituisce la partizione in cui verranno salvati i dati;   
    **composite key (partition + sort key)** tutti gli items con stessa simple key vengono salvati insieme e ordinati in ordine numerico dal key value. 
    - Attributi $\rightarrow$ ogni item è composto da uno o più attributi  
        Attribute = (name,value) - corrisponde a una entry nel  'dizionario' k:v
    PRO: Non richiedono uno schema, hanno il concetto di PK, estremamente veloci e scalabili  
    CONTRO: Non supportano Join cross-table, ne aggregazione; supportano un numero limitato di data types (int,string,binario) e hanno un supporto di indicizzazione limitato (solo hash-range).   

2. **Document Based:**   
    Offrono più flessibilità per modellare i dati e le query, il data model è una _collezione di documenti_.  
    Documenti supportati: {JSON, XML, altri formati semi-strutturati}  
    Sono più appropriati per dati semi strutturati, gli Objects sono documenti che possono avere diversi tipi di dato, hanno un indicizzazione oltre le funzioni hash e infine supportano analitiche di query avanzate come join e aggregazione.  
    
3. **Column Based o Row Based:**    
    Una tabella relazionale è serializzata come righe/colonne e viene flushata al disco, buoni principi di località spaziale e ottimizzato per caching; gli aspetti negativi sono le operazioni sulle colonne (se serializzato a righe) o righe (se serializzato a colonne).  
    Pensato principalmente per search engines e usato molto anche per applicazioni di messaggistica (Facebook usa Hbase)  
4. **Graph Based:**  
    Il focus è sul modellare la struttura dei dati in termini di interconnettività.  
    Ispirato dalla graph theory matematica $G=(E,V)$ e scala bene con la complessità dei dati.  
    - Nodi: possono avere proprietà (incluso ID)
    - Archi: possono avere ruoli e proprietà
    - Proprietà: sono coppie key:value
    - Labels: raggruppa nodi in sets (insiemi)
    - Invece di calcolare relazioni a query time, i graph DB salvano in memoria collegamenti (sottografi) per essere prontamente visitati e attraversati.  


### Svantaggi dei modelli noSQL: 

- Paradigma nuovo $\rightarrow$ il mondo era abituato a SQL
- Non ci sono standard $\rightarrow$ diventa difficile switchare database
- Mancanza di maturità $\rightarrow$ in quanto relativi nuovi c'è ancora spazio per nuovi tool 
- Linguaggi di query deboli $\rightarrow$ Bisogna fare un maggiore sforzo programmatico che potrebbe ferire le performance


I noSQL sono da utilizzare quando: 
1. La scalabilità è una priorità
2. Se essere schemaless è importante
