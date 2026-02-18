# Fundamentals of Deep Learning 


Le reti neurali esistevano già dagli anni 50' ma sono esplose solo dopo il 2012 grazie alla convergenza di due fattori:

1. Data: I modelli di AI e le reti neurali richiedono enormi quantità di dati per essere addestrate. Con l'era di internet, IoT e dei social media abbiamo generato come società petabyte di dati etichettati, senza di questi il motore dell'AI non sarebbe stato in grado di girare.     

2. Computing Power: Non basta avere solo i dati, serve anche  potenza di calcolo enorme, che al giorno d'oggi esiste.    


Il Deep Learning capovolge il paradigma di programmazione tradizionale.  
Invece di fornire istruzioni specifiche, si forniscono esempi per fare in modo che il sistema trovi da solo pattern e impari a riconoscerli.   

**Schema $\rightarrow$ Dati + Risposte = Modello** 


- **Dati + Risposte**: forniamo al sistema un immagine (o dati) e l'etichetta corretta 
- **Training:** il computer pova a indovinare una regola interna, all'inizio fallirà ma con un algormito diremo al modello di aver sbagliato e di correggere i suoi parametri.  
- **Apprendimento:** Dopo aver visto milioni di esempi e correzioni, il sistema crea da solo un modello matematico che mappa i dati alle risposte corrette.   


Il DL si distigue dal ML per:

1. **Profondità:** Si chiama 'deep' perchè ha molti strati di neuroni uno dietro l'altro; più strati ci sono più la rete è capace di imparare concetti astratti (es. strato1 riconosce line, strato10 riconosce forme, strato100 riconosce facce).  

2. **Parametri:** I parametri sono i pesi che la rete deve calibrare durante il training, le reti neurali hanno spesso miliardi di parametri!   


<br> <br>


## Concetti fondamentali 

I seguenti concetti spiegati sono stati incontrati nei laboratiri, in particolare [in questo Notebook](./notebooks/mnist.ipynb)

### 1. Tensori  

Un **Tensore** è un contenitore strutturato di numeri.  
Permette di definire strutture a N-dimensioni, in 1D abbiamo array, in 2D matrici in N-D abbiamo tensori.  

Il tensore è la struttura, al suo interno può contenere tipi di dato come int, float, ... 


### 2. Immagini e Computer   

Un computer non è in grado di vedere un immagine come la intendiamo noi, vede solamente luminosità.  

Vede un immagine come una sequenza di valori che rappresentano l'intensità di un colore in una scala specifica.  


La proietta sullo schermo come una matrice composta da N righe ed M colonne.   
Es: immagine 28x28 (mnist) a scala di grigi $\rightarrow$ le celle della matrice possono assumere valori tra 0 e 255, dove lo zero rappresenta il nero totale e il 255 rappresenta il bianco.    


### Casting di immagine a tensore: 

Per permettere a pytorch di lavorare e addestrare un modello tramite immagini dobbiamo assicurarci che questa venga trasformata in un *tensore* (ossia in una griglia di numeri), per permettere alla GPU di fare le operazioni matematiche velocemente.  
Se l'immagine fosse salvata in un formato come JPEG la GPU non saprebbe farci i calcoli matematici sopra.  


Una volta trasformata in tensore, il valore minimo e massimo che possono avere gli elementi diventa normalizzato, e vanno da un minimo di `0.0` a un massimo di `1.0`.  
Questo per evitare che i calcoli matematici diventino troppo coomplicati.  

Pytorch, una volta trasformata un immagine in un tensore, organizza sempre i dati in un **ordine preciso**: **`C x H x W`**  

- C è il color channel, nel caso MNIST essendo foto in scala di grigi è 1
- H è il numero di righe (pixel verticali)
- W è il numero di colonne (pixel orizzontali)   

