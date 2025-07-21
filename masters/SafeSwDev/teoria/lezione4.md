# Contenuti 

Contenuto per l'esame
- Protostar: 4,5,6 (guardare a 1 $\rightarrow$ 6, prime 4 dal sito)
- Pentester: SQL 1,2,3,8; 4,5,6,7,9 non le chiede per permessi elevati
- XSS: 1 
- Code/Command injection: Guardarle tutte (sono facili)
- File include: 1,2 (e guardare la teoria delle mitigazioni)

Extra: 
- XSS: 1,2,3
- Code injections: 1,2,3,4
- SQL injections: 1,2,3,8


Errori tipici che verranno considerati:
1. Mancata sanitizzazione input 
2. Eccesiva verbosità output
3. Eccessiva esposizione di asset 
4. Esecuzione con privilegi eccessivamente e inutilmente elevati 
5. Corse critiche



### Permessi sui file
I permessi di accesso al file sono 3 terne di azioni:
- Prima terna: Creatore del file 
- Seconda terna: Gruppo di lavoro del creatore del file 
- Terza terna: Tutti gli altri utenti 

Permessi tipici 

```bash
-rwxr-wx-r-x 2 root root ... zsh
-rw-r--r--   1 root root ... passwd 
```

Rappresentazione ottale:
- `R` $\rightarrow$ 4
- `W` $\rightarrow$ 2 
- `X` $\rightarrow$ 1 
rwxrwxr-w $\rightarrow$ 4+2+1, 4+2+1, 4+1 $\rightarrow$ 0775 (ottale)

Rappresentazione simbolica:
- `R` $\rightarrow$ r
- `W` $\rightarrow$ w
- `X` $\rightarrow$ x
- Creatore $\rightarrow$ `u`; Gruppo $\rightarrow$ `g`; Altri $\rightarrow$ `o`
rwxrwxr-w $\rightarrow$ ug+rwx, o+rx (simbolica)

**Bit di permessi `SETUID, SETGID`**

- SETUID: 4 | s (ottale, simbolica)
- SETGID: 2 | s (ottale, simbolica)


Un processo normale in esecuzione assume le credenziali dell'utente (userid e groupid) che lo ha lanciato.  
Se il bit SETUID o il bit SETGID è attivo allora il processo assume le credenziali del creatore del file (userid o gruopid primario) $\rightarrow$ In questo modo diventa possibile cambiare la persona 'effettiva' che esegue il file.  


Il descrittore di un processo memorizza una prima coppia di credienziali, userID e groupID reali, che sono sempre gli ID di chi ha lanciato il processo.  

Il descrittore di un processo memorizza una _seconda_ coppia di credenziali: **user ID e group ID effettivi**.  
Se il bit SETUID è attivo $\rightarrow$ userID effetivo diventa userID del creatore del file. Stessa cosa per SETGID.  


**Algoritmo di controllo dei permessi**:  
Le credenziali _effettive_ del processo sono messe a confronto con i permessi di ogni elemento del percorso di un file, se le credenziali sono sufficineti per ogni elemento del percorso allora viene accordato il permesso di accesso, altrimenti si nega il permesso.  


**Abbassamento e ripristino dei privilegi**:  
Nei SO moderni si usa una terza coppia di credenziali: **userID e groupID salvato**  
Quando un processo parte, le credenziali salvate sono una copia delle credenziali "effettive".  
- Se l'applicazione non svolge operazioni critiche abbassa i propri privilegi a quelli dell'utente che ha eseguito il comdado $\rightarrow$ _priviledge drop_ 
- Quando l'applicazione svolge operazioni critiche, ripristina i privilegi ottenuti tramite l'elevazione automatica $\rightarrow$ _priviledge restore_
- È possibile ripristinare i privilegi a quelli effettivi andando a pescare il valore dell'ID salvato.  


```bash
# chiamata di sistema che ritorna l'ID reale del processo invocante
getuid() 

# chiamata di sistema che ritorna l'ID effettivo 
geteuid()

# permette il cambio degli ID utente di un processo al valore uid
setuid(uid) # userid effettivo

# abbassamento permanente vs temporaneo:
setuid(getuid());
seteuid(getuid());
```



# Corse Critiche

La sincronizzazione di un processo o thread è l'atto di un processo o thread di bloccare la propria esecuzione nell'attesa di un evento; non appena l'evento si verifica il processo o thread riprende la prpopria esecuzione.  

Sincronizzazione sul cambio di stato: la chiamata di sistema `wait()` blocca il processo invocante fino a quando uno dei suoi figli non cambia stato (es: processo figlio esce normalmente o con errore, processo figlio stoppato o ripristinato da un segnale).    

L'ordine di schedulazione di processi e thread impatta _pesantemente_ sul risultato delle operazioni su un asset condiviso (come una variabile).
Quando il risultato finale di una elaborazione su un asset condiviso *dipende* dall'ordine di esecuzione delle istruzioni che la modificano si è in presenza di una **Corsa Critica (race condition)**.  

Siano dati $n$ task eseguiti concorrentemente e cooperanti tramite dati condivisi, diremo che ciascun $T_j$ ha una porzione di codice che accede ai dati condivisi, tale porzione prende il nome di **Sezione Critica**, tale sezione critica deve essere eseguita al _massimo_ da al più un processo/thread alla volta.  

Approcci al problema:
- Accesso in mutua esclusione: si bloccano gli accessi alla variabile condivisa se questa è già in uso
- Esecuzione atomica: si fa in moodo che ogni tentativo di modifica alla variabile condiisa sia eseguito integramente oppure per nulla.  


Nei SO moderni le sezioni dii ingresso e di uscita da una zona critica sono implementate tramite il modello di **`lock`**.  

- `spinlock` $\rightarrow$ implementazione ad attesa attiva, un ciclo while eseguo fino a quando non se ne vanifica la condizione, la CPU continua a ciclare sul while bloccando l'accesso  
    Da usare per attesa brevi, nell'ordine dei ns, diventa disastroso per attese lunghe in quanto il ciclo while brucia il processore  e impedisce ad altri processi di avanzare aumentando la latenza.  

- `attesa passiva` $\rightarrow$ l'attesa nel `wait()` è di tipo passivo. il processo viene messo in attesa (sleep) che si verifichi l'evento che consente l'accesso alla sezione critica, quando si verifica il processo è risvegliato ed inserito in coda.   
    Ideale per attese lunghe, dai ms in su, in quanto si blocca un processo che altrimenti impallerebbe la macchina. Diventa disastroso per attese brevi in quanto la sospensione ed il ripristino del processo introducono un ritardo inaccetabile.   






