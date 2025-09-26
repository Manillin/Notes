molti problemi np-hard hanno importnati applicazioni e non possono essere ignorati  

soluzioni: 
- risolvere efficientemente questi problemi solo per istanze piccole 
- utilizzare euristiche 
- utlizzare algoritmi paralleli o distribuiti
- utilizzare algoritmi di approssimazione: algoritmi che in tempo polinomiale restituiscono una soluzione ammissibile (non necessariamente ottima) ma con garanzia di discostamento della soluzione ottima.  


TSP: 
input : G = (V,E) non diretto, completo e pesato.    
output: ciclo hamiltoniano di costo minimo.   


Ciclo Hamiltoniano: (NP-completo)
input: G =(V,E) non diretto
output: booleano che mi dice se esiste un ciclo hamiltoniano in G.  

trasformiamo il problema del ciclo hamiltoniano in un problema adatto a TSP, aggiungiamo archi necessari per rendere il grafo di partenza completo e aggiungiamo costo 1 ad ogni arco aggiunto, il resto degli archi ha peso zero.   
risolviamo il problema tsp con questo grafo modificato che ci restituirà il costo minimo del ciclo hamiltoniano individuato. Il costo 0<= k <= |V| , e se k=0 allora esiste un ciclio hamiltoniano nel grafo di partenza (non abbiamo percorso nessuno degli archi nuovi aggiunti a costo 1), se invece k >= 1 allora abbiamo percorso almeno un arco extra che non era nel grafo di partenza, di conseguenza non esiste un ciclo hamiltoniano nel grafo originale.  

