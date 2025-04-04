# appunti 4 aprile


sw unit -> piu piccola parte di un sw che può essere eseguita, in linguaggi procedurali è una funzione in linguaggi a oggetti un metodo di classe o statico. 

Lo unit testing è un processo del ciclo di sviluppo sw che mira a collaudare le sw unit!  


Asserzione verifica se una relazione tra un oggetto restituito e un oggetto di riferimento coincide (es: asserzione di uguaglianza, esistono altri tipi di asserzione)    


Se ho funzionalità ricche e complesse queste risultano in test composti da molte asserzioni, si preferisce fare vari test composti da un numero limitato di asserzioni piuttosto di un test con tante (troppe) asserzioni, si vuole spaccare il test 'infinito' in più funzioni di test.   


un **test case** è un raggruppamento di tutti i test relativi ad uno specifico sottosistema / sottocasi.  

I metodi di test all'interno diu n test case possono essere organizzati secondo criteri diversi:
- verificare il funzionamento dei metodi di una classe 
- collaudare uno specifico caso d'uso  

Un test case quindi fonde queste due cose.  


**set-up e teardown**:  

Per alcuni test occorre eseguire codice preparatorio, come collegamento a servizi esterni, config dei socket ecc...
- si creano operazini che eseguono queste cose in automatico
- set-up: crea i collegamenti e risorse necessarie per poter far funzionare il test (questa parte viene data per scontata come funzionante)
- tear down: ripulisce lo spazio e chiude le connessioni che sono state aperte per il test.  

```python
class AssetCase(TestCase):
    
    def test_asset_f1():
        <SETUP>;
        <TEST F1>;
        <TEARDOWN>;
```


occhio: se setup e teardown sono troppo grossi -> si sta sbagliando qualcosa, va spezzettato.  


**TestSuite**:  

È un raggruppamento di test che possono essere eseguiti nella stessa sessione di test (in parallelo o sequenzialmente).  

<br>

--- 

**Ciclo di vita di uno unit test**:  

Il test deve rispettare l'api dell'applicazione che stiamo testando (se il test è scritto male allora l'api sarà mal documentata, l'api è corretta se i test passando {attenzione: non perfetta, non si possono eliminare completamente gli errori e i bug})    

1. progetto, scrittura, exec codice applicazione 
2. Analisi report test
3. Correzione degli errori codice applicazione 
4. Correzione errori test
5. Rifattorizzazione codice applicazione 
6. Rifattorizzazione del test


Vantaggi:
- se il codice è rofattorizzato correttamente, il progetto e la scrittura dei testo sono banali -> si scrive un test per ogni metodo/funzione 

Svantaggi:
- il progetto e la scrittura dei test sono influenzati dal codice applicativo preesistente (che potrebbe non essere dei migliori)  
    - es: "Questo già so che funziona, non lo testo", “Questo il cliente non me lo richiede, non lo testo.”, “Questo non è funzionale alla demo, non lo testo.”, ”Testo solo l’input critico per la demo.”  
    - la mentalità sopra è estremamente sbagliata, a volte non dipende completamente dal programmatore ma anche dai superiori.  


Nasce quindi per superare questi limiti il **ciclo di vita test-driven**, si fanno prima i test, li si fanno fallire e poi si implementa il codice per superare il test.  
Se l'api immaginata dovesse risultare sbagliata allora ci si accorge subito, si migliora l'api e si rifattorizza il codice esistente

Vantaggi:
- la qualità dei test prodotti con questo approccio è nettamente superiore all'approccio classico -> non esistendo il codice sorgente non si può barare, si è costretti a ragionare e immaginare l'api. 
- il sw lo si scrive immaginando prima l'api: immagino l'api, scrivo i test e poi la implemento.     

