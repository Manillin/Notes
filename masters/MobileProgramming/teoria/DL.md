# Efficient DL for mobile devices  

L'uso di AI in dispositivi edge ed embedded viene generalemente implementata dal modello di cloudAI: il dispositivo edge registra i dati, li impacchetta e li spedisce a un server potente sul quale vive il modello AI, il server esegue il calcolo complesso e restituisce il risultato al dispositivo edge.  

È un modello che funziona bene ma che presenta problemi di privacy, memoria, bandwidth e latenza, la soluzione di tutti questi problemi è spostare l'intelligenza dal cloud direttamente sul dispositivo!  


Dispositivi embedded $\rightarrow$ risorse limitate !  
Programmare per questi dispositivi richiede di ottimizzare le risorse e trovare compromessi, gli stessi ragionamenti andranno fatti per spostare l'AI su questi dispositivi.  

In particolare:  
- Training del modello: averrà sul cloud, in quanto consiste in una parte che richiede cluster di GPU e un consumo enorme di energia.  
- Inferenza: consiste nell'uso del modello già addestrato, vogliamo fare in modo che questa avvenga in locale nel nostro dispositivo, quindi dovrà essere veloce e consumare poca energia!  


### Problema pratico:  

I modelli di DL stanno diventando esponenzialmente più complessi e più pesanti in termini di memoria. La dimensione di un modello è data dai supi parametri (pesi) e negli ultimi anni si è passati da modelli con milioni di pesi a modelli con centinaia di miliardi di pesi, occupano sempre più memoria e non possiamo farceli stare tutti nei dispositivi edge così come sono.  


Dobbiamo ridurre drasticamente le dimensioni e la complessità di questi modelli per poterli fare etrare nei nostri dispositivi edge, perdendo meno accuratezza possibile!  

le principali tecniche sono:
1. Model Optimization: tecniche per creare architettura di rete neurale che siano intrinsecamente più piccole e veloci 
2. Efficient Training Algorithms: non interessa particolarmente al contesto di edge in quanto l'addestramento si farà in cloud
3. Efficient Deployment: tecniche per prendere un modello già addestrato e restringerlo prima di installarlo sul dispositivo.  



