# recap on SQL queries


**Order by:**  
select * from ordine 
where importo < 100
order by data [ASC | DESC]

**query aggregate:**  
- sum, avg, min, max, count 

select max(importo) as max-imp 
from ordine   

select sum(importo) as somma-imp 
from ordine 
where cod-cliente=1

**query con raggruppamento:**  
- group-by (raggruppamento): usando groupby il risultato della select è un unico record per ciascun gruppo  
- having (seleziione dei gruppi)

select ... from ... where ... [groupby | having] 






MOVIE: ID, title, yr, director, budget, gross 
ACTOR: ID, name 
CASTING: movieID, actorID, ord



## Anatomia Query SQL:

Il database esegue le operazioni in un ordine logico specifico, che **NON** è sempre l'ordine in cui scriviamo le operazioni.   

1. FROM/JOIN: costruisce il set di dati iniziali 

2. WHERE: filtra le singole righe, viene eseguito prima di qualsiasi raggruppamento

3. GROUP BY: raggruppa le righe filtrate e cambia le regole per la SELECT, una volta usata la select  può contenere soltanto:
    - le colonne su cui hai raggruppato 
    - funzioni di aggregazione che operano sui gruppi 

4. HAVING: filtra i gruppi creati da GROUP BY, questo è l'unico punto in cui puoi filtrare in base al risultato di una funzione di aggregazione, viene invocato dopo il GROUP BY, agisce su gruppi di righe

5. SELECT: seleziona le colonne da proiettare 

6. ORDER BY: ordina le righe del risultato finale, è una delle ultime operazioni (ASC è il default, DESC va specificato)

7. LIMIT: prende un sottoinsieme del risultato finale  

8. UNION: permette di unire i risultati di due query

9. EXCEPT: permette di fare la differenza tra due query (selezionare .... ma che non siano presenti in .... )  

10. INTERSECT: permette di unire query da due tabelle che contengono gli stessi risultati, l'analogo del AND logico (selezionare .... E che siano presenti in ....)  



left join 