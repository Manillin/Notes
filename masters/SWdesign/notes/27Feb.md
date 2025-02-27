# appunti 


modello composto da diverse fasi , tipicamente 5:
- specifica
- fare il designa (capire cosa deve avere il sistema)
- implementazione 
- testing e validazione 
- manutenzione e update


volendo potrebbero essere un po mescolate tra di loro. 



fasi:
**specifica:**  
definire i requisiti del nosto sw, capire cosa deve fare e i constraints. 
esistono tecninche di ingegneria dei requisiti.
È importante capire di cosa ha bisogno il cliente, più di capire 'cosa vuole'.  


ogni sottoattività (immagine pagina 6) produce artefatti, che possono essere documenti testo o diagrammi ecc... che alla fine vanno a confluire nel pacchetto finale che serve al progettista per implementare il sistema software. 

**design/progettazione:**  
la fase di progettazione server per capire quali sono i componenti di cui abbiamo bisogno e scegliamo l'architettura, intesa come struttura che realizza le specifiche.
Deve essere universale a livello di tecnologie, nel senso che i linguaggi e fw vengono scleti nella fase successiva.


**implementazione:**  
Solo al terzo step dopo una buona fase di progettazione si inizia a scrivere il codice, che deve essere **pensato**.  
Design e implementazione sono strettamente collegati anche se per rendere flessibile ed adattabile il progetto, si dovrebbero tenere separati la fase di progettazoìione e implementazione.


metodologie (specificano come fare la fase di design) di design:  
approccio sistematico allo sviluppo sw, tipicamente i progetti sono documentati da modelli grafici (DFD, ER, UML,...).  

fase di implementazione e testing  ++

fase di verifica e validazione:
si usano casi di test dalle specifiche per vedere se il sistema nel suo complesso risponde alle specifiche che ci eravamo dati.    
le operazioni di testing può essere divisa in : testing in the small (singoli componenti per volta) e testing in the large (il sistema intero)  



coverage test: coprire le diramazioni dell'esecuzione del sw, cioè coprire tutti gli if, else bìecc....



## Process Models 

waterfall model:  
mette in cascata le fasi, una per volta, quindi una volta che si termina una fase si procede alla successiva, e non prima!  
ha i suoi vari pro (crea buona documentazione, molto comodo per problemi noti es: creare un nuovo gestionale ) e contro (processo rigido e poco flessibile, gestisce con difficoltà i cambiamenti e il ciclo sw è fortemente condizionato da cambiamenti).  


si ripetono passi durante le iterazioni e i requisiti cambiano leggermente (slide 31), guardare l'immagine a pagina dopo per vedere il ciclo che comprende l'evoluzione dei requisiti.  

pro: è flessibile ed adattabile a cambiamenti nei requisiti.  e molto utili per sviluppo di sistemi sconosciuti (es: io sono esperto di gestionali ma mi chiedono un app).  



