# neo4j - lezione 11 marzo 

- gestisce in modo efficiente grafi di grandi dimensioni  
- Ogni modifica sui dati è persistente (resiste a mal funzionamenti del sistema)  
- lingugaggio ad hoc dichiarativo $\rightarrow$ Cypher 
- come tutti i modelli no sql supporta un modello flessibile   
- ottime performance e soprattutto scalabile - le performance non degradano in base alla quantità di dati gestita (ha un accesso ottimizzato ai dati).  


componenti grafo :

**nodo** - self expl  
**relationship** - arco : potrebbe avere una direzione.  
**walk** - sequenza ordinata che esprime un ordine di attraversamento, nodi ed archi possono ripetersi  
**trail** attraveramento in cui non si ripetono gli stessi archi (i nodi si)
**path**  un trail in cui non si ripetono i nodi  

--- 

nei graph db non esiste un modello standard di rappresentazione dei dati, si spiegherà il modello neo4j  



## Property Graph Data Model 

le componenti di un property graph di neo4j includono:
- nodi (entita)
- relationships (archi tra entita)
- properties 
- labels 

si creano grafi come risposta a importanti domande/quesiti all'intero del nostro dominio

si possono assegnare più etichette allo stesso nodo. 

la stessa cosa vale per le connessioni (archi), le etichette sulle connessioni servono per capire il tipo di collegamento,inoltre bisogna controllare il dominio applicativo tra archi ed entita (tipo casa -> married -> person | rappresentazione insensata).  

In neo4j gli archi sono direzionali, ma nell'interrogazione posso non interessarmi della direzione .  
es pA -> married -> pB [relazione simmetrica ma arco orientato da A verso B]; da tenere in considerazione quando si fanno le interrogazioni sul db.  


proprietà negli archi rappresentati come coppie 'key:value'. 
alcune proprietà le posso salvare su un nodo ma posso non volerle salvare per un altro nodo con la stessa etichetta (infatti node4j è **schemaless**)  

-> esempio  
nodo1: label-persona, mi salvo il colore degli occhi   
nodo2: label-persona non salvo questa informazione 


