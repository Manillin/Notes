**Domande Orale ML**  



- Metodi del gradiente per funzioni senza vincoli
    
    - Formula della successione
        
    - Scelta dello step length
        
    - Regola di Armijio
        
    - Caso particolare in cui si può cercare l’ottimo esatto senza backtracking (Caso quadratico)
        
    - Come si connette il concetto di ottimizzazione alla metodologia di learning
        
- Qualche esempio di funzione loss
    
    - Quadratica, caso di classificazione binaria
        
    - Cosa valuta la funzione loss
        
- Differenze tra gradiente deterministico e stocastico
    
    - Quando possiamo applicare lo stocastico
        
    - Filosofia e struttura di base dell’algoritmo stocastico
        
    - Come si ovvia al problema che il singolo esempio potrebbe essere non significativo? (Momentum e Adam cercano di ridurre la varianza del metodo)
        
- Convergenza metodo stocastico: come si comporta?
    
    - Perché è più lento andando avanti?
        
    - Di che tipo di convergenza abbiamo parlato? (In media)
        
    - Che valore ha il limite del valore atteso di E |Fk – F*| con k -> infinito? Va ad una costante e non a zero
        
    - Il comportamento asintotico per metodo del gradiente stocastico con steplength fisso o decrescente cambia? Si
        
- Esempio di class. Binaria o regressione con tanti dati, che software cerco in rete? Qual è l’idea di base con la quale lavorano?
    
- Scrivere il metodo del gradiente proiettato (X)
    
    - Come si sceglie lambdak (1, prendo tutta la direzione di passo)? Armijio in questo caso come funziona? (X)
        
- Classificazione binaria SVM, classi linearmente separabili: Come si costruisce il classificatore lineare
    
    - E nel caso di classi non linearmente separabili?
        
    - Ruolo del parametro C (regolarizzatore/generalizzatore)
        
- Regolarizzazione in Reti Neurali, contesti e metodologie (X)
    
- Funzione loss del gradiente stocastico
    
- Teorema di dualità, cosa le viene in mente?
    
- Backtracking non monotono (X)
    
- Andamento non monotono della decrescita nel gradiente stocastico (Convergenza in expectation quindi non è garantita discesa ad ogni passo)
    
    - Convessità forte, steplenght fisso -> risultato di convergenza stocastica (X)
        
        - La convergenza dipende da alfa, alfa piccolo più preciso ma più lento
            
    - Convergenza in expectation -> non monotona
        
    - Convessità forte, steplength variabile -> risultato di convergenza stocastica (X)
        
    - Convergenza nei casi non convessi (X)
        
    - Perché analizziamo il gradiente e non l’optimality gap?
        
- Nelle reti convolutive cosa sono il layer convolutivo e il max pooling
    
    - Nella convoluzione su che parametri ottimizziamo? I parametri del kernel
        
- Cosa esprimono le condizioni KKT?
    
- Descrivere i problemi di regressione e l’approccio SVM per la loro risoluzione
    
    - Funzione loss
        
- Decomposizione nelle SVM (X)
    
    - Calcoliamo il kernel e lo teniamo in memoria perché è costoso, come mai? Dipende dal kernel
        
    - Esempi di kernel, scrivere il polinomiale e il gaussiano
        
    - Parametro della varianza nel kernel gaussiano, come lo setto? Dipende dall’ordine di grandezza della norma al numeratore
        
- Iperparametri delle reti neurali
    
    - Quali sono gli iperparametri del gradiente stocastico?
        
    - Gradiente stocastico con un valore solo o in minibatch, come funzionano? Quali sono gli iperparametri
        
    - Iperparametri nelle reti convolutive
        
- Come si definiscono le direzioni di discesa e quali abbiamo visto?
    
- Strategia di barzilai-borwein, che strategia è stata usata?
    
- Definizione dell’alfa nel caso quadratico con minimizzazione esatta
    
- Ben posizione della regola di armijio
    
- Come si sceglie lo steplength in Adam (non rispetta la dimostrazione di convergenza perché non è decrescente né costante)
    
- Classificazione multipla con SVM e con reti
    
    - Funzione loss per fully connected layer (logistic regression che in questo caso si chiama cross entropy)
        
    - Softmax
        
- Vincolo attivo che ruolo ha per per le condizioni KKT
    
- SVM per classificazione binaria non lineare
    
    - Funzione obiettivo e loss
        
    - Enunciare duale
        
    - Collegare i valori di alfa al concetto di vettori di supporto e supporto al bound (bound alfa = C, non al bound 0 <= alfa < C)
        
    - I vincoli del problema duale che regione ammissibile definiscono? È semplice?
        
- Passaggio al kernel nelle reti (reti multilayer)
    
- Percettrone modello matico
    
    - Base semplice con funzione di attivazione lineare
        
    - Funzioni di attivazioni più complicate (Relu, leaky relu (strati intermedi), sigmoide e tanh( per layer di output)), sono differeneziabili?
        
- Condizioni di ottimalità per l’ottimizzazione vincolata necessarie e sufficienti
    
- Loss function per classificazione binaria e regressione in SVM formule
    
- Approcci quasi newton per ottimizzazione non vincolata, metodo e vantaggio


Pomeriggio:

1:

- Metodi del gradiente per ottimizzazione non vincolata
    
    - Definizione di direzione di discesa
        
    - Barzilai borwein per l’accelerazione del metodo del gradiente steepest
        
- Differenza tra steplength fisso e decrescente nel gradiente stocastico (Non è necessario ricordarsi tutte le costanti nelle formule!!!)
    
    - Altri metodi per migliorare le prestazioni dello stocastico (momentum e adam)
        
- Evitare overfitting nelle reti neurali
    
    - Come si ottimizzano gli iperparametri, tipo lamda (cross validation)
        

2:

- Problema duale dell’SVM, Kernel
    
- Percettrone e reti multi layer
    
    - Tipologia di livelli (convoluzione, pooling e fully connected)
        
    - Come si introduce la non linearità
        

3:

- Gradiente proiettato e come si usa in SVM
    
    - Pseudocodice gradiente proiettato
        
    - Duale del SVM
        
    - Con Q troppo grande cosa faccio? Decomposizione
        
- Limiti del gradiente deterministico e perché conviene usare lo stocastico
    
- Misure di prestazioni per algoritmi di machine learning formule (precision, recall e accuracy)
    
- Logistic regolarizzata per classificazione binaria formula
    

4:

- Legame tra ottimizzazione e machine learning e regolarizzazione
    
- Non ha studiato e tenta di aiutarlo con argomenti semplici
    

5:

- Loss function viste nel corso, in particolare dell’SVM
    
- KKT (imparare bene come si scrive il gradiente della lagrangiana)
    
    - I gradienti che intervengono sono solo quelli attivi, se non sono attivi i moltiplicatori sono zero
        
- Teorema di Wolfe
    
- Percettrone e come si estende (studiare le formule di attivazione dei percettroni multilayer (Relu et.)
    
- Classificazione multipla
    

6:

- Metodo del gradiente proiettato
    
    - Backtracking non monotono
        
- Problema di machine learning con gradiente proiettato (SVM)
    
    - Scrivere il duale
        
- Decomposizione per SVM
    
- Tecniche di regolarizzazione e a cosa serve
    
- Strati convolutivi nelle convnet
    
    - Che parametri ottimizzo? Pesi del kernel
        
- Ottimalità con le KKT  

DOMANDA NOSTRA DA CHIEDERE

Come scegliere s nella regola di armijio?



Domande fatte all’orale di venerdi 08/01 nei turni del mattino. Ad ogni numero corrisponde una persona interrogata. In generale al pomeriggio e negli orali successivi ha chiesto sempre questi macroargomenti, andando, in alcuni casi, su domande più particolari, elencate in fondo.

1

- Strategie dei metodi del gradiente per funzioni prive di vincoli.
    
    - Regola di Armijo
        
    - Caso speciale in cui è possibile utilizzare la regola di minimizzazione esatta
        
- Richiamare brevemente perché serve minimizzare una funzione quando si addestra una metolodogia machine learning
    
    - Qualche esempio di funzione loss
        
- Gradiente stocastico. Differenze con il deterministico e quando lo possiamo applicare
    
    - Convergenze dei metodi
        
    - Su questo comportamento asintotico (convergenza) ha un influenza la scelta dello step length?
        
- Se abbiamo un training set di un problema di classificazione binaria o regressione abbastanza corposo, e vuole affrontare con la metodologia SVM il problema di learning, che software andrebbe a cercare in rete?
    

2

- Approcci del gradiente che si possono utilizzare in ottimizzazione con vincoli
    
    - Per il metodo del gradiente proiettato, che caratteristica devono avere i vincoli?
        
- SVM: Abbiamo un training set composto da xi, yi con i=1,…,n. yi appartiene a {-1,1}. Le classi sono linearmente separabili. Come le SVM costruiscono un classificatore lineare?
    
    - Cosa modifichiamo invece per i set non linearmente separabili? Cosa rappresenta il parametro C?
        
- Nelle reti neurali abbiamo parlato di regolarizzazione per evitare overfitting (dropout e early stopping).
    
- Quando abbiamo visto il gradiente stocastico e abbiamo fatto gli esperimenti, come funzione loss cosa abbiamo utilizzato?
    
- Se io cito il termine “teoria di dualità” in ottimizzazione vincolata, cosa le viene in mente?
    

3

- Metodi del gradiente proiettato, illustrami la generalizzazione del metodo del gradiente ad un problema vincolato
    
    - Descrivi Armijo in questo caso
        
    - Nel gradiente proiettato quale lamba iniziale prendiamo (1)
        
- Cosa si intende con backtracking non monotono?
    
- Andamento non monotono in gradiente stocastico, relativo ai risultati di convergenza.
    
    - Il gradiente decresce? Cosa possiamo dire sulla convergenza?
        
- Strati convulativi di max-pool
    
    - Nella convulazione su quali parametri ottimizziamo?
        
- Cosa esprimono le condizioni KKT?
    

4

- Problema di regressione e come affrontarlo con SVM
    
- Passaggio dal primale al duale. Come si scrive la lagrangiana?
    
- Tecniche di decomposizione.
    
- Esempi di Kernel con formula (lineare, polinomiale, gaussiano) – cosa rappresenta la varianza nel gaussiano
    
- Concetto di iperparametri nelle reti. Quali sono gli iperparametri e cosa cambia da un parametro e un iperparametro.
    
- Differenze gradiente stocastico classico e con mini batch
    
- Iperparametri nelle reti convoluzionali
    
- Quando si minimizza una funzione senza vincoli, per fare gli spostamenti ci servono delle direzioni di discesa. Come si definiscono le direzioni di discesa e elencami alcuni metodi.
    

5

- Scelta parametro di lunghezza di passo per il metodo del gradiente in ottimizzazione non vincolata.
    
- Calcolo a con minimizzazione esatta nel caso di f quadratica strettamente convessa
    
- Differenze monotono e non monotono
    
- Gradiente stocastico con steplength fisso o decrescente
    
- Classificazione multipla svm e in reti neurali
    
    - Descrivi softmax
        
- Ruolo vincoli attivi in KKT
    

- SVM per classificazione binaria
    
    - Passaggio al duale
        
    - Collegare il concetto di alfa ai vettori di supporto e bound support
        
- Tecniche di decomposizione
    
- Il passaggio da complicato a leggero come si fa nelle rete neurali?
    
- Modello matematico del perceptrone, funzioni di attivazione
    
- Idee di Barzilai Borwein per suggerire uno steplength appropriato
    

7

- Ottimalità per condizione vincolata. Condizione necessaria del primo ordine e Condizioni necessarie e sufficienti del secondo ordine.
    
- Perché si usa un parametro di regolarizzazione
    
    - Loss function usata nella svm binaria
        
- Regolarizzazione in reti neurali. Parametro e tecniche (dropout, early stopping)
    
- Approcci quasi newton
    

  
  

Domande particolari:

- Logistic regression nel caso di classificazione binaria
    
- Scelta di componenti da mettere in base nella decomposizione (kkt-violating)
    
- Gradient related
    
- Convergenza lineare metodi steepest nel caso quadratico (formula)
    
- Formule per accuracy, precision e recall