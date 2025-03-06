# Appunti 06 marzo 

### Row Based Storage 

In un DBMS si seralizzano appendendo (una dietro l'altra) le varie righe della tabella.  
il contro è che le operazione sulle colonne (tipo i group-by) sono costosi in quanto devono caricarsi tutta la tabella e prendere dati extra inutili per la query.  



### Column Based Storage 
Si serializzano le tabelle appendendole per colonne una dietro l'altra. 
Le operazioni sulle colonne (come i group-by & **aggregazioni**) sono estremamente veloci e cheap, ma allo stesso tempo le operazioni su riga diventano più complesse.  


### Column Storage with Locality Groups - Google Big Table

È simile al column storage ma con gruppi di colonne che ci aspettiamo di accedere insieme.  
Inoltre non si salvano i valori null (salvo valore in maniera **semistrutturata**) per risparmiare spazio -> wide column store.  

chi progetta un db di questo tipo deve decidere le column family!



### Data Model 

una tabella è sparsa, distribuita, e una mappa multidimensionale ed ordinata.  
(row:string, column:string, time:int64) $\rightarrow$ uninterpreted byte array  



# Graph Based:

nascono con intenzione di mettere in evidenza i collegmaneti tra i dati, l'idea è di gestire milione/miliardi di archi e nodi.
Nasce dalla teoria dei grafi e sono asmpiamente utilizzati dai social network.  

label property graph model -> possiamo applicare etichette ai nodi che ci dicono il tipo di nodo e di arco.

Questo permette di salvare costo nelle operazioni di tipo join che calcola le relazioni a tempo di query, memorizzo le relazioni direttamente !