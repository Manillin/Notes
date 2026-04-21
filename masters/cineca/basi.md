# Concetti Linux Utili 




### Scaricare pacchetti (debian) e apt vs apt-get  

Su Debian il sistema che gestisce l'installazione dei pacchetti si chiama `dpkg`, ma poichè tale tool è complesso è stato creato `apt` (Advanced Package Tool).   
- `apt-get`: comando storico, stabile e retrocompatibile e **non si aspetta interazione umana**
- `apt`: comando più recente, progettato per interazione umana   

Quando si scrivono script che devono scaricare pacchetti bisogna usare sempre `apt-get`.  
Quando si fa debugging a mano si usa `apt`

Per scaricare software, linux usa `/etc/apt/sources.list` come riferimento, dentro a tale file ci saranno gli URL dei server (repository) che contengono il software.  
`apt` usa la crittografia asimmetria per proteggere il processo di installazione.  
1. chi compila il sw possiede una chiave privata segreta, che usa per firmare digitalmente i pacchetti 
2. dobbiamo avere la loro chiave pubblica sul nostro server 
3. quando scarichiamo un pacchetto, `apt` usa la chiave pubblica per verificare la firma, se combaciano il pacchetto è autentico 


ES: Scaricare Kubernetes   

```bash
curl -fsSL https://pkgs.k8s.io/core:/stable:/v1.33/deb/Release.key | sudo gpg --dearmor -o /etc/apt/keyrings/kubernetes-apt-keyring.gpg
```   
- curl -> apre una connessione https e scarica la chiave pubblica di k8s. I parametri -fsSL dicono a curl di essere silenzioso (s), ma di mostrare gli errori in caso di fallimento (f, S), e di seguire eventuali reindirizzamenti web (L). Il file scaricato è in formato testo ASCII.
- gpg --dearmor -o -> `apt` richiede che le chiavi siano in formato binario compresso, gpg --dearmor prende la chiave in ASCII, lo trasforma nel formato richiesto e lo salva nel percorso che specifichiamo 

```bash
echo 'deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/v1.33/deb/ /' | sudo tee /etc/apt/sources.list.d/kubernetes.list
```
- echo -> stampa la stringa di configurazione, dove dice esplicitamente ad atp di accettare il file da tale indirizzo SOLO se hanno la stessa chiave crittografica (quella creata prima da gpg) 
- tee -> prende quello che stampa echo e lo scrive dentro al file `kubernetes.list`


```bash
sudo apt-get upadate
```
- apt andrà a leggere que nuovo file, si connetterà allURL e scaricherà la lista di pacchetti disponibili verificando con la chiav gpg.   


---


### Congelare pacchetti:   

```bash
sudo apt-mark hold kubelet kubeadm kubectl
```  

apt-mark hold congela la versione di quei pacchetti, quando lancremo apt-get upgrade, apt aggiornerà tutti i pacchetti SENZA toccare quelli marcati dal hold.  
nota: In k8s è FONDAMENTALE congelare questi pacchetti, in quanto se eseguiamo un aggiornamento automatico gobale su un nodo kubernetes, apt aggiornerà kubelet all'ultima versione. K8s ha regole di compatibilità rigidissime (skew policy). se il kubelet viene aggiornato ma l'api-server no, il nodo si disconnette dal cluster e causa outage.  

---


### IPv4

Un indirizzo IP è una sequenza di 32 bit, per renderlo leggibile si divide in 4 blocchi da 8 bit, ogn blocco va da 0-255.  

Maschera di sottorete:   
Per sapere come instradare un pacchetto un router deve sapere:
1. quale parte dell'IP identifica la rete/network
2. quale parte identifica l'host  

Questa divisione viene decisa dalla *Subnet Mask*, che è unna sequenza di 32 bit con una regola semplice:
1. dove la maschera ha un 1, quella parte dell'IP è bloccata e identifica la rete 
2. dove la maschera ha uno 0, quella parte dell'IP è libera e identific l'host 

Si usa la notazione CIDR (classless inter domain routing) per semplificare la notazione.  
- con 24 bit a 1 -> /24 
- con 16 bit a 1 -> /16
- con 8 bit a 1 -> /8  

Per sapere quante macchine possiamo fare stare in una sottorete si usa la formula : 32 - CIDR  

ES: 10.244.0.0/16  
- la maschera CIDR blocca i primi 16 bit, identifica la rete 
- rimangono 32-16 = 16 bit liberi per le macchine (o pod in k8s)
- il numero di IP disponibili è: 2^16 = 65536 
  - il primissimo IP: 10.244.0.0 identifica la rete e non può essere assegnato 
  - l'ultimo IP: 10.244.255.255 è l'indirizzo di *broadcast*, usato per comunicare con tutti i nodi contemporaneamente 
  - abbiamo quindi 65534 IP liberi!  


---

### SSH

Il collegamento SSH avviene in due fasi  

Prima Fase: **Key Exchange**  
Prima ancora di fare richieste di autorizzazione, client e server si accordano su come criptare i dati, per costruire il secure tunnel.   

Si scambiano chiavi a vicenda (generate con Diffie-Hellman) per creare una **Session Key** che verrà usata per cifrare la comunicazione tra i due (critt simmetrica).   

Solitamente il collegamento avviene tramite tcp sulla porta 22 del server.   



Seconda Fase: **Authentication**   

Il client mostra la sua Pubblic Key, il server controlla se tale chiave è tra quelle autorizzate (/.ssh/authorized_keys) e se lo è genera un `Nonce` (number once).  

Cifra questo numero con la chiave pubblica del client e lo spedisce al client stesso, questa cifratura è sbloccabile solo con la chiave privata, che possiede solamente il client.  
Una volta decifrato il messaggio il client manda al server il numero che si aspetta e si crea la connessione.  


Il resto dei messaggi e scambio di dati tra client e server avviene con la sessione key, la crittografia asimmetrica viene usata solo per l'autenticazione.  

--- 

### Block Storage vs Object Storage   

**Block Storage**:   
L'equivalente di un hard disk fisico, viene presentato al OS come un device grezzo (/dev/vdb).  

I dati sono divisi in blocchi di dimensioni fisse, l'OS della VM deve formattarlo con un file system prima di usarlo.    


caratteristiche: 
- bassa latenza, ideale per carichi di lavoro intensivi
- accesso esclusivo, un volume può essere attaccato (normalmente) a una sola VM per volta
- ideale per i PV di K8s 


**Object Storage**:  

I dati vengono salvati come *oggetti* (file + metadati) dentro a dei container chiamati Buckets.  

Non si monta un disco ma si interagisce con lo storage tramite API HTTP (get, put, post, delete) e ogni oggetto ha un URL univoco.  

Caratteristiche:  
- scalabilità: teoricamente infinita, si possono salvare PB di dati 
- accesso distribuito: chiunque abbia le chiavi API può leggere l'oggetto via web
- alta latenza! non è adatto per farci girare carichi di lavoro storage intensive 
- Si usa per i backup -> es. salvare immagini dei container o per i file i log in k8s.   
- 

---


## Networking  

### 1. Forward Proxy (Proxy Tradizionale)
Un Forward Proxy è un server intermediario che risiede sul lato **Client** della comunicazione. Agisce per conto degli utenti di una rete privata che tentano di accedere a risorse su una rete pubblica (Internet).

*   **Scopo Principale:** Anonimizzazione del client, filtraggio del traffico in uscita (Egress), auditing e caching delle risposte per risparmiare banda.
  
*   **Meccanismo Tecnico:** Il client configura il proxy come suo gateway applicativo. Il proxy riceve la richiesta, esegue un SNAT (Source Network Address Translation) applicativo nascondendo l'IP originale del client, e la inoltra al server di destinazione. Il server web esterno vedrà solo l'IP del Forward Proxy.
   

*   **Esempio:** Un proxy aziendale (es. Squid) in ascolto sulla porta 3128. Un dipendente tenta di raggiungere `[http://facebook.com](http://facebook.com)`. Il proxy intercetta la richiesta, verifica le sue Access Control List (ACL), constata che il dominio è bloccato dalla policy aziendale e scarta il pacchetto restituendo al client un errore HTTP 403 (Forbidden).


---

## 2. Reverse Proxy
Un Reverse Proxy è un server intermediario che risiede sul lato **Server** (Infrastruttura). Agisce per conto dei server di backend, intercettando le richieste provenienti dall'esterno prima che queste raggiungano i servizi interni.

*   **Scopo Principale:** Protezione dell'infrastruttura (nasconde gli IP interni), Load Balancing, TLS Termination (decriptazione SSL centralizzata) e compressione.
  
*   **Meccanismo Tecnico:** I client esterni risolvono tramite DNS l'IP pubblico del Reverse Proxy, ignorando totalmente l'architettura sottostante. Il proxy riceve la richiesta TCP, stabilisce lui stesso la connessione con il client e apre una seconda connessione indipendente verso il backend scelto per recuperare la risorsa.
  
*   **Esempio:** Un server NGINX perimetrale esposto su Internet. Riceve traffico crittografato HTTPS sulla porta 443. Esegue la *TLS Termination* decifrando il traffico grazie al certificato SSL a bordo, e inoltra la richiesta HTTP in chiaro sulla rete privata (porta 8080) a uno dei 3 server backend Tomcat (es. `10.0.0.10`, `10.0.0.11`, `10.0.0.12`) utilizzando un algoritmo di *Round Robin*.


---


## 3. Ingress Controller (Il Reverse Proxy di Kubernetes)
In un cluster Kubernetes, un **Ingress Controller** (es. Traefik, NGINX Ingress) è un Reverse Proxy specializzato e dinamico. L'oggetto K8s `Ingress` è solo il file manifesto YAML con le regole di routing, mentre l'Ingress Controller è il demone (Pod) che concretamente le esegue.

*   **Problema Architetturale Risolto:** Il *Multiplexing*. In un cloud provider (come OpenStack), ogni IP pubblico costa. L'Ingress permette di esporre decine o centinaia di servizi diversi utilizzando un singolo IP pubblico e una singola porta (es. la 443).   
  
*   **Meccanismo Tecnico (Host-based / Path-based Routing):** L'Ingress Controller opera rigorosamente a **Livello 7 (Applicativo)** del modello OSI. Poiché tutto il traffico in ingresso arriva sull'unica porta 443, il Load Balancer L4 non sa come smistarlo. L'Ingress Controller prende il pacchetto, decifra il layer TLS e ispeziona l'**HTTP Host Header** (l'URL digitato dall'utente) e l'**URI Path**. In base a queste stringhe di testo, applica le regole di routing e inoltra il traffico verso l'IP della rete Overlay (ClusterIP) associato al Pod di destinazione corretto.
*   **Esempio:**
    *   Unico IP Pubblico assegnato al cluster: `131.175.x.x` (porta 443).
    *   Traffico A: Il client invia un pacchetto con Host Header `api.miosito.it`. L'Ingress lo legge e instrada il traffico all'IP Overlay del Pod Backend (`10.42.1.15`).
    *   Traffico B: Il client invia un pacchetto verso `131.175.x.x` con Host Header `web.miosito.it`. L'Ingress lo ispeziona e lo instrada all'IP Overlay del Pod Frontend (`10.42.2.33`).  



--- 

### HTTPS & TLS  

Si usano certificati SSL/TLS per criptare la comunicazione, avviene in due fasi.  

1. Handshake:  
  Il server target possiede due chiavi: Pubblica e Privata.  
  Quella pubblica la condivide con tutti e quella privata serve per decriptare i messaggi.  
  - Quando il client chiede di connettersi, il server gli lancia la sua chiave pubblica  
  - Il client (spesso il browser) crea una chiave temporanea: **Session Key** e la chiude con la chiave pubblica del server e la spedisce. Solo il server che ha la private key può decifrare questo messaggio.  
  - Il server riceve il pacchetto, usa la sua chiave privata e scopre la session key. Ora client e server hanno un segreto condiviso!  

2. Conversazione (crttografia simmetrica):
  Una vlta che client e server si sono scambiati la session key, usano solo quella per comunicare e decriptare il traffico velocemente.  


---

### X.509

La X.509 ha il solo scopo di **legare un'identità a una chiave crittogrfica**.  
Serve per verificare che una certa chiave pubblica appartenga con certezza a qualcuno di fidato.  

Viene usata dal client per capire se il server con cui vuole parlare è veramente lui.  
Se il certificato è diverso da quello che si aspetta abbatte la connessione.  

Viene usata ESCLUSIVAMENTE nella fase di handshake asimmetrico iniziale -> permette al client di verificare l'identità del server e di avere la ccertezza che la chiave pubblica contenuta al suo interno sia autentica.  
Una volta che client e server hanno concordato la session key parlano usando la crittografia simmetrica. il certificato x.509 non viene più usato da quel punto in poi.  


### Router != ReverseProxy  

Il router ha un IP pubblico ma NON è un reverse proxy. Lavorano a due livelli separati dello stack OSI.  


- Router: lavora a livello 3 - rete, è velocissimo ma stupido, guarda soloIP e porta dei pacchetti che riceve, non ha idea di cosa ci sia dentro il pacchetto

- Reverse Proxy: lavora a livello 7 - applicativo, è software che gira su una macchina (virtuale), riceve il pacchetto DAL router e guarda il contenuto (del messaggio HTTP), guarda l'URL richiesto e in basa a quello decide a quale VM della subnet inoltrarlo 


Ruter e ReverseProxy hanno bisogno entrambi di indirizzi pubblici raggiungibili da internet.  
ENTRAMBI gli indirizzi verrano messi sul router! si crea perà una regola speciale per il floatingIP usato per il reverse proxy = appena arriva traffico che ha come destinazione il floatingIP, il ruouter lo inoltra immediatamente verso la VM a cui è mapato quel IP, e tale VM fa il lavoro del reverse proxy


### Load Balancer L4 vs Reverse Proxy L7  

- *Livello 4 (TCP Proxy)*: Legge solo IP e Porta (es. 6443). Prende i byte e li inoltra al backend senza ispezionarli. Consuma poca CPU ed è vitale quando il traffico è cifrato end-to-end (come l'API di K8s) perché non altera i certificati SSL. 
- *Livello 7 (HTTP Reverse Proxy)*: Apre il pacchetto, ispeziona l'header HTTP, legge l'URL, i cookie e i certificati. Può fare routing intelligente (es. "Se `/api` vai al server A, se `/web` vai al server B"). Consuma più risorse. 

### NAT:  

Il NAT serve a mascherare gli IP privati (es. 192.168.x.x), che non sono instradabili su Internet, sostituendoli con IP pubblici/instradabili. Il destinatario di un pacchetto non vede mai l'IP privato originale del mittente, altrimenti non saprebbe come fargli recapitare la risposta.   

### Interfacce Virtuali vs Fisiche:  

Un'interfaccia come tun0 (VPN) è virtuale: è un software che crittografa i pacchetti. Tuttavia, fisicamente, il pacchetto crittografato deve uscire dal computer attraverso la scheda di rete hardware (es. enp0s31f6 per il cavo Ethernet, o enx... per i dongle USB/Docking Station).  


  