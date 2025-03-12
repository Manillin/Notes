# Appunti lezione


integrità referenziale: quando cancelliamo un elemento voglio che scompaia completamente dal DB e da altri collegamenti $\rightarrow$ tipo delete cascade in sql, non garantito per db no sql, deve essere fatto a livello applicativo, devo andare a cancellare tutti i riferimenti a quel dato in tutto il resto dei dati.  

Rischio di avere un DB inconsistente se applicativamente non gestisco bene questa dipendenza.  


The Perfect Storm $\rightarrow$ è un movimento 

CAP $\rightarrow$ consistency - availability problem 


key value :  
scalabile e predicibile per workload non predicibili 