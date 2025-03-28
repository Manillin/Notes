# Appunti lezione 28 Marzo 



Agile unified process (AUP):
- deriva dal unified process, adattato al modello agile 
- aumenta il numero di cicli, da 6-8 passano a 18-22; e ogni ciclo si accorcia da 2-3 mesi a 3-5 settimane.  


Principi di AUP:
- lo staff (sviluppatori) sanno quello che stanno facendo: si fornisce documentazione e manuali ma non vengono imposti
- semplicità: tutto è descritto in modo conciso e usando una manciata di pagine 
- Agilità: La agile unified process si conforma ai principi e valori della Agile Alliance.  
- focus su attività high value: il focus è sulle attività che contano, non su tutte le possibili cose che potrebbero accadere nel progetto.
- Indipendenza dai tool: Si possono usare qualsiasi tool, ma si consigliano tool semplici e open source
- 



## Scrum 

l'idea è che tutti gli sviluppatori spingano nella stessa direzione (analogia con il rugby).  

Lo sviluppo è composto da **sprint**, ossia piccole diverse iterazioni.  
Ogni iterazione deve produrre un working product.  
È privilegiato il rapporto face to face.


prevede 3 ruoli:
- product owner: figura del team che si interfaccia con il cliente.   
- team: composto dai dev che sviluppano 
- scrum master: legata al team (ma non è ne il capo ne un tecnico), si occupa di gestire i problemi organizzativi (non tecnici) del team.  

si produce un product backlog, con ordine di priorità per i suoi elementi.  
il team fa uno sprint planning meeting e scelgono delle task (il numero dipende dalla complessità, bisogna prenderne tante quante servono per terminarle durante lo sprint) di maggiore priorità dal backlog e cercano di terminarlo entro uno sprint.  

si fa lo sprint, che dura circa 2 settimane.  
Ogni 24 ore ci si trova per fare un daily scrum meeting per fare il punto della situazione.  
Si fa anche uno sprint review in cui si illustra al cliente quello che è stato fatto mentre lo sprint retrospective: si cerca da un punto di vista organizzativo (non tecninco) un moodo per ottimizzare i prossimi sprint su quanto imparato dall'ultimo sprint.  


in depth:

product owner: 
- interagisce con gli stakeholders 
- prende le decisione strategiche (accettare il progetto, a quanto accettarlo, ...)
- considerato come il capo

Scrum master:
- gestisce il team non dal punto di vista tecnico ma che si occupa dei problemi di gestione e organizzazione. È quello che difende il team 

Team:
- Composti da 5-9 persone 
- sono persone multifunzionali, tutti responsabili di tutto, nessuno è uno specialista
- auto-organizzante (scelgono dal product log cosa fare in autonomia, sono aspetti tecnici) 
- prendono le decisione operative (usare un certo fw, struttura dati, lib, ecc...)



## I meeting:  

1. Sprint Planning:  
    andare a definire quali funzionalita possono essere implementate all'interno del prossimo spring (cioe fare il passaggio dal product backlog allo sprint backlog).  
    Vengono coinvolti tutti - product owner, scrum master e il team.  
    Questo meeting indicativamente dura 8 ore

2. Daily Scrum:  
    Partecipano anche qui product owner, scrum master e il team.  
    Ha una durata di 15 minuti, si fa stando in piedi e si fa il punto della situazione, l'obiettivo è identificare i problemi non risolverli in quel momento.  

3. Backlog Grooming:  
    Partecipa solo il team, si prendono i requisiti e si trasformano in taks.  
    L'obiettivo è valutare e revisionare le requirements del product backlog.    
    La durata dovrebbe essere meno del 10% della durata totale dello sprint.  

4. Scrum of Scrum:  
    È un meeting inter-team in cui partecipa una persona per ogni team.  
    questo meeting si fa giornalmente o con meno frequenza.  
    serve per evitare sovrapposizioni o divergenze tra team diversi.  


5. Sprint Review:  
    Si presentano i risultati dell'ultimo sprint  
    Tutti sono invitati, Product owner, Scrum master e team, ma anche stakeholders, customers e utenti  
    La durata è di 2 ore ed è un meeting informale (no slide!!)   
    Il gruppo decide come procedere (gettano le fondamenta per il prossimo sprint - iterazione), chiedendo al cliente se vada bene quello sviluppato per ora, cosa si possa migliorare e cosa buttato ecc...


6. Sprint Retrospective:   
    Il team insieme allo scrum master (senza product owner e stakeholders) fanno una autovalutazione per capire come è andato lo spring -  a livello organizzativo, di realazioni personali, pianificazione dei miglioraenti.  
    Sono suggerite 3 ore per uno sprint di 4settimane, questo meeting è l'ultima cosa che va fatta dopo aver concluso uno sprint  


### Artefatti scrum 

1. Product backlog:  
    Elenco dei requisiti, casi d'uso, user stories, ecc...  
    relative a tutte le funzionalita che il clinete vuole, organizzate in base a priorità.  
    La priorità viene chiesta al cliente ma poi viene interpretata dal product owner (valutando anche i rischi) che compila il product backlog (ogni elmemento è un PBI - prod backlog item).  

2. Spring Backlog:  
    deriva dagli item superiori del product backlog  
    il team si auto-organizza, è qualcuno che prende il task, non il task che viene assegnato.  
    Consiste nella quantità di lavoro che il team pensa di poter portare a termine durante quello sprint (senza neanche finire in anticipo)  
    Si usano spesso delle task board 

3. Diagramma burndown  
    asse x giorni dello sprint, asse y c'è lo sforzo rimanente (potrebbero essere 'ore di lavoro rimanenti') e sempre su y ci sono i task completati.  
    idealmente la riga verde deve scendere linearmente (slide p27), ma solitamente non è cosi, infatti guardare la linea blu con i rombi, all'inizio è normale che lo sforzo sia maggiore ma a una certa dovrebbe scendere !  



--- 

## Feature Driven Development (FDD)

si hanno 5 attivita di sviluppo, chiamati processi, e si ha un grande focus sulle feature richieste dal cliente.  



Attività:  

1. Develop an overall model  
    Si basa sul concetto DDM (domain driven modelling), l'obiettivo è avere un modello snello ma sufficiente per capire cosa sviluppare (JEDI)  
    Un team composto da 3 figure: chief architect, domain expert e chief programmers (solo i capi).  
    Gli artefatti di queste attività sono: diagrammi UML(di classe e sequenze) e note


2. Build feature list:  
    lavora dagli artefatti precedentemente prodotti e si traducono in feature (in base al business value) inserite in un backlog  
    Ogni feature è specificata da una sentenza nella forma \<action\>\<result\>\<object\>  


3. Plan by feature:  
    Attivita di pianificazione -> si identifica come procedere con la fase di sviluppo.  
    Ogni feature è assegnata a un chief programmer che poi suddivide il lavoro nel team (si organizza gerarchicamente).  

4. Design by feature:  
    si fa il design della feature selezionata  
    coinvolti in chief programmmer e gli altri programmatori  
    si produce un design package, un documento composto in diverse parti: {introduzione , requisiti, diagrammi di sequenza, alternative al design, modello ad oggetti, strumenti da utilizzare, un calendario temporale con una to-do list per ogni membro}  


5. Build by feature:  
    Si passa dal progetto delle classi alla traduzione con un linguaggio di programmazione.  
    ogni programmatore produce le classi di cui è responsabile e produce anche lo unit test, si fa anche code inspection.  
    Il chief programmer decide quali classi vengono promosse e quali no, quelle promosse vengono integrate, quelle bocciate 

