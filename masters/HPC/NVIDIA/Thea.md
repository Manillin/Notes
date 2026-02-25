# NVIDIA Thea:

Singolo entry point (login node con arch x86) con ssh.  
Due partizioni diverse: 
- **GH(Grace Hopper)**:  
- **GG(Grace Grace)**: 

Va specificata la partizione sulla quale vuoi lavorare (`-p gh` o `-p gg`).  

Si tratta di un cluster al quale hanno accesso anche altri utenti, tutti entrano nel login node (se lanci comando pesante qui rallenti tutti).  
Per accedere a GH o GG bisogna prenotare la risorsa (che viene fornita in modo esclusivo) 

### Hardware: Compute 

Stiamo accedendo a un cluster, un insieme di computer collegati tra loro.  

1. CPU+GPU: 6x Quanta S74G-2U GH200  
    - 72 Core: (tantissimi, un pc high end 16-24 mentre un server 32-64), Architettura ARM 
    - 3.4GHz: valore standard per server (CPU gaming 5-6GHz), la tengono bassa per non fondere il chip, la potenza la fa il parallelismo con il numero di core.  
    - 480 GB LPDDR5X: È la RAM (essendo un sistema GH la memoria è accessibile anche dalla GPU)  
    - Hopper GPU con 96GB HBM3: scheda video H100 con 96GB di VRAM (una RTX4090 ne ha 24) e la memoria HBM3 è velocissima 
    - Infiniband NDR400, 1 HCA: I 6 sistemi GH sono collegati con Infiniband (fibra ottica dedicata ultraveloce); HCA è il host channel adapter, ossia la scheda di rete, ogni nodo ne ha una.  

2. CPU Only: 8x Supermicro ARS-221GL-NR (Grace Grace)  
    - Sono 8 nodi Grace-Grace (due processori Grace saldati insieme).  
    - Dual Grace CPU 72 core x2 = 144 core 
    - HCA sul socket 0: il socket è l'alloggiamento fisico sulla scheda madre dove si incastra il processore, avremo Socket0 (CPU0) e Socket1 (CPU1) 
        - La HCA è collegata al socket0 ossia alla CPU0, se CPU1 deve mandare dati in rete deve chidere il permesso alla CPU0 di usare la scheda di rete.  
    

### Hardware: Storage

**Home: `/global/home/users/$USER`**   

- I dati sono forniti da NFS (network file system), il disco rigido non è dentro il computer, è esterno e collegato via cavo.  
    Il vantaggio è che se cambi nodo (da nodo0 passiamo a nodo1) vediamo gli stessi file, il costo è che è più lento di un disco locale.   
- Quota enforced di 20GBs: spazio di archiviazione massimo di 20GBs sulla directory home, sono pochi perchè la home serve solo per file di configurazione, script e codice sorgente.  
    I dati grossi vanno messi in /scratch o in /local (i dati qui si cancellano a fine lavoro).  

**Scratch: `/global/exafs/users/${USER}`**  

- Directory dedicata per lavori pesanti come: parallel I/O (scrivere su disco da tanti core contemporaneamente), checkpoints (file di checkpoint per addestramento AI), container execution (file .sif che sono grossi)  
- Velocissimo (DDN Exascale e NVMe SSD) e senza backup, serve farli manualmente.  


**Local: `/local`**  

- Memoria più veloce (è un disco SSD NVMe dentro il nodo di calcolo) ma effimera, viene purgata quando un job finisce 
- workflow: prenoti un nodo, copi i dati da $SCRATCH a /local, lanci il calcolo (che leggerà e scriverà su /local a velocità altissima).  
    Finito il calcolo dobbiamo copiare i dati importanti da /local a $SCRATCH in quanto scaduto il tempo di prenotazione sul nodo, /local viene riformattata. Se non si importa su $SCRATCH si perde il lavoro   


**64K pages enabled**:  
- La RAM non viene gestita byte per byte ma a blocchi chiamati pagine 
- in un pc standard le pagine sono: 4KB
- in questo sistema le pagine sono da: 64KB
- avendo pagine più grandi l'indice per fare le cache hit è più piccolo e riduce i TLB miss.  


<br>


### Compilation Node e Partizioni in Slurm:  

Il login node ha una CPU standard x86 mentre i nodi potenti hanno CPU Grace con architettura ARM aarch64.  
I programmi compilati sul login node non saranno eseguibili sui compute node perchè parlano lingue differenti!  

Per questo serve il **Compilation node** chiamato `ggcompile`, è un nodo ARM configurato per fare entrare più utenti contemporaneamente (oversubscription) dedicato solamente per la compilazione.  

In un cluster gestito da Slurm, una **partizione** è un gruppo di computer (nodi) etichettati per uno scopo specifico.  
Nel caso Thea avremo:
- Parizione `gh`: nodi grace hopper 
- Partizione `gg`: nodi grace grace 
- Partizione `ggcompile`: dedicata alla compilazione 

La partizione è il nome logico del gruppo, mentre il nodo è il computer fisico vero e proprio (gh001, gg005, ggcompile001)  

<br>

### Modulefiles: 

I cluster HPC sono di default vuoti, perchè solitamente su tali cluster ci lavorano più persone diverse e per evitare conflitti di versioni di sw diverse (tipo librerie) di tiene tutto vuoto.  
I module files sono dei file che quando vengono caricati, modificano al volo l'ambiente (variable `$PATH`) per usare il sw che ci serve, sono interruttori software che accendono o spengono la visibilità di compilatori e librerie.  

Se serve installare una libreria specifica, va installata in `${SCRATCH}` dal codice sorgente o facendo **Spack local chaining**.  

Spack è una sorta di appstore per i supercomputer, invece di scaricare le app giù pronte scarica le ricette e compila il sw da 0 ottimizzandolo per il processore che stiamo usando.  


<br>

### SW Environment 

Tutte le partizioni del cluster fanno boot della stessa immagine 
- Ubuntu Linux 24.04
- CUDA driver compatibili con CUDA 13 (solo su gh)
- MLNX_OFED_LINUX-24: sono i driver della scheda di rete inifiniband, senza questi la rete ultra veloce 400Gb/s funzionerebbe come una ethernet normale, questi attivano la RDMA (remote directory memory access) che consente al nodo0 di scrivere nella RAM del nodo1 senza disturbare la CPU del nodo1 [driver mellanox e open fabric enterprise distro]  
- GNU GCC 13.3.0 (altri compilatori a disposizione da modulefiles)


<br>

### Submit and Run Jobs  

Slurm è lo schedulatore di sistema, ogni job ha un walltime massimo di 12H e i nodi sono allocati di default in modalità exclusive, la GPU è sempre visibile una volta allocato un nodo, non c'è bisogno di usare `gres`.  
(`gres` serve in Slurm standard per prenotare esplicitamente la GPU, si imposta come un flag che segue questa sintassi `--gres=gpu:1`; serve perchè normalmente i cluster hanno nodi misti, alcuni con GPU altri senza o alcuni con 4 GPU e magari tu ne vuoi solo 1).  

Esampio di interactive Job:

```bash
# allocate a GH200 node
salloc -n 72 -N 1 -p gh -t 1:00:00

# allocate 2 GH200 nodes
salloc -n 144 -N 2 -p gh -t 1:00:00

# allocate two specific GH200 nodea
salloc -n 144 -N 2 -p gh -w gh002,gh004 -t 1:00:00

# allocate 4 GH200 and exclude a specific one
salloc -n 188 -N 4 -p gh -x gh001 -t 1:00:00

# allocate a GG node 
salloc -n 144 -N 1 -p gg -t 1:00:00

# allocate 4 GG nodes 
salloc -n 576 -N 4 -p gg -t 1:00:00
```


`mpirun` è lo starter dei programmi paralleli MPI.  
Quando si crea un programma in C/C++ con MPI si crea un eseguibile unico, per farlo girare su 100 core diversi usiamo `mpirun -np 100 ./programma`.  
mpirun si occupa di: 
1. clonare il programma su tutti i nodi 
2. assegnare a ogni clone un ID (rank)
3. Mettere i nodi in comunicazione tra loro  

Esempi di script con mpirun:

```bash
#!/bin/bash -l
#SBATCH --ntasks=4  [processi totali MPI che voglio]
#SBATCH --ntasks-per-node=2 [di questi processi, ne voglio 2 per computer]
#SBATCH --cpus-per-task=36 [1*]
#SBATCH --nodes=2 [voglio 2 computer fisici]
#SBATCH --partition=gh 
#SBATCH --time=1:00:00
#SBATCH --exclusive 

module purge # [pulisce ambiente da vecchi moduli]
module load openmpi/5.0.8_gcc-13.3.0 #[carica MPI]
mpirun -np 4 --map-by ppr:2:node:PE=36 \ 
    --report-bindings uname -a  
# [2*]
```
- [1*]: voglio 36 cpu per task perchè ogni nodo gh200 ha 72 core, quindi se metto 2 task per nodo $\rightarrow$ 72 / 2 = 36; assegno 32 core a ogni processo MPI
- [2*]: lancia 4 processi totali con mapping:   
    ppr:2:node $\rightarrow$ metti 2 processi per ogni nodo  
    PE=36 $\rightarrow$ 36 Processing Elements (core) per ogni processo   
    --report-bindings uname -a $\rightarrow$ stampa a video dove ha messo i processi (utile per debug, il comando da eseguire stampa il nome del PC).  

<br>

Es2: Usare `srun` invece di `mpirun`:  

```bash
#!/bin/bash -l
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=36
#SBATCH --nodes=2
#SBATCH --partition=gh
#SBATCH --time=1:00:00
#SBATCH --exclusive
srun --mpi=pmi2 uname -a
```  

`srun` è il comando nativo di Slurm. fa la stessa cosa di mpirun ma è integrato meglio con il cluster, Slurm leggendo le righe in alto sa già cosa fare (non serve load openmpi ne mpirun).  
srun --mpi=pmi2 dice a Slurm di lanciare i processi usando l'interfaccia PMI2 per farli parlare tra loro.  

<br>

Es3: Hybrid MPI + OpenMP  
Spesso lanciare 144 processi MPI è inefficiente (troppa comunicazione), per questo si preferisce l'approccio ibrido: 
- pochi processi MPI (es. uno per socket o uno ogni tot core)
- ogni processo MPI lancia tanti thread OpenMP

```bash
#SBATCH --ntasks=64 
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=9
#SBATCH --nodes=4
#SBATCH --partition=gg
#SBATCH --time=1:00:00
#SBATCH --exclusive

module purge
module load openmpi/5.0.8_gcc-13.3.0

export OMP_NUM_THREADS=9
mpirun -np 64 --map-by ppr:16:node:PE=9 \
    --report-bindings uname -a 
```


### Singularity Containers  

Singularity è l'unico engine di container presente, docker o enroot devono essere adattati per poter runnare su Thea.  

Container $\rightarrow$ È una scatola chiusa che contiene tutto quello che serve a fare girare un programma: sistema operativo, librerie e driver.  

NON si usa docker perchè richiede permessi di root e in HPC i cluster sono sempre condivisi tra più persone (evita danni accidentali).  

**Singularity / Apptainer** è un container engine fatto apposta per HPC, non servono i permessi di root (dentroil container sei lo stesso utente che sei fuori).  
Usa un singolo file con estensione `.sif`  

Adattare Docker a Singularity $\rightarrow$ Con il comando `pull` possiamo prendere l'immagine di docker di interesse e convertirla in un file `.sif`.  


### Eseguire container pre-esistente:

Viene mostrato come lanciare un lavoro al volo usando un'immagine già esistente nel cluster.  

Metodo1: Modalità interattiva 

1. Allochiamo un nodo interattivo  
    ```bash
    salloc -n 1 -N 1 -p gh -t 1:00:00
    ```

2. Selezioniamo il container e invochiamo `singularity run`  
    ```bash
    export CONT="/global/exafs/groups/gh/sif/pytorch-25.01-py3.sif"  
    singularity run --nv ${CONT}
    ```
    - run: esegue il container 
    - --nv: attiva il flag che indica la presenza di una GPU sul nodo, il programma dentro il container vede e usa i driver NVIDIA del cluster, senza questo comando non potrebbe vedere la GPU  

Metodo 2: One shot con `srun`  

```bash
export CONT="/global/exafs/groups/gh/sif/pytorch-25.01-py3.sif"  
srun --mpi=pmi2 -N 1 -n 1 --ntasks-per-node=1 gh 4:00:00 \
    singularity -v run --nv "${CONT}" python my_benchmark_script.py
```  

Note:          

Il path corrente è automaticamente esposto, se siamo in `/home/christian/progetto` e lanciamo singularity:
- il programma dentro il container può leggere i file della directory dalla quale è stato lanciato con singularity, inoltre potrà scrivere su $HOME, $SCRATCH e /local.    
- docker non vede i file a meno che non vengano passati esplicitamente con -v 


### Creare una nuova immagine 
Possiamo scaricare un container docker da internet e trasformarlo in un file `.sif`  

**_Problema delle Performance I/O:_** costruire un container è un operazione pesante, significa scaricare, scompattare, scrivere migliaia di file temporanei piccoli e poi ricompattarli nel file `.sif`  
- se lo fai sulla $HOME o $SCRATCH (che sono lenti) intasi la rete e ci metti tanto tempo 
- devi fare questa operazione sul disco `/local` che contiene la NVMe veloce!  

**Passi:**  

1) Prendi un nodo in modalità interattiva:  

```bash
salloc -n 1 -N 1 -p gh -t 1:00:00
```

2) Dirigi i file temporanei su `/local`:  

```bash
export APPTAINER_TMPDIR=/local/${SLURM_JOBID}/_tmp_singularity
export APPTAINER_CACHEDIR=/local/${SLURM_JOBID}/_cache_singularity
rm -rf ${APPTAINER_TMPDIR} && mkdir -p ${APPTAINER_TMPDIR}
rm -rf ${APPTAINER_CACHEDIR} && mkdir -p ${APPTAINER_CACHEDIR}

```

Queste variabili dicono a Singularity di non usare la $HOME o altro per la spazzatura temporanea, ma di usare il disco veloce `/local` dentro una cartella con il nome del job id.  
Inoltre si fa pulizia di eventuali vecchi residui e si creano le cartelle nuove    


3) Usare il comando pull 

```bash
singularity pull pytorch-25.01-py3.sif docker://nvcr.io/nvidia/pytorch:25.01 -p 
```

va sul registro NVIDIA e cerca l'immagine di docker specificata, scarica i layer docker e li converte nel formato Singularity usando le cartelle temporanee veloci in /local precedentemente impostate e infine crea il file finale "pytorch-25.01-py3.sif" nella cartella dove si trovava l'utente che ha lanciato il comando! (nota: conviene farlo da $SCARTCH).  
_Nota:_ Fare sempre `cd $SCRATCH` prima di lanciare `singularity pull` altrimenti si rischia di creare il file `.sif` nella $HOME (e potrebbe saturare i 20GB di quota) o in /local, e tutto ciò che è in local viene formattato a fine job, si perderebbe il lavoro.  


I file `.sif` sono di **sola lettura**, se dentro abbiamo ad esempio Python installato ma ci manca una qualche libreria, non possiamo modificarlo al volo e mantenere le modifiche permanenti, la prossima volta che eseguiremo il container non vedremo nessuna modifica.    

Per aggiungere qualcosa in modo permanente bisogna creare una nuova immagine, e per fare ciò useremo le **Sandbox** e i **Definition File**


**Sandbox e repackage**:  

Sono operazioni che ci permettono di modificare file .sif, utile quando dobbiamo aggiungere librerie o risorse esterne che non esistono nel container attuale.  
Ricorda: I file .sif sono read-only file system; non li stiamo modificando direttamente, li usiamo come base per costruirne uno nuovo e nel momento in cui è in costruzione singularity aggiungerà ciò che abbiamo specificato, il risultato finale sarà un file .sid nuovo con ciò che ci serve.  


Passi: 

**1**: Prendere un nodo del cluster in modalità interattiva:   

```bash
salloc -n 1 -N 1  -p gh  -t 2:00:00
```

**2**: Identificare il container di partenza da estendere e prepariamo l'ambiente:   
```bash
export CONT_DIR = "global/exafs/groups/gh/sif"
export CONT_NAME = "pytorch-25.01-py3.sif"
export APPTAINER_TMPDIR = "/local/${SLURM_JOBID}/_tmp_singularity"
export APPTAINER_CACHEDIR = "/local/${SLURM_JOBID}/_cache_singularity"
rm -rf ${APPTAINER_TMPDIR} && mkdir -p ${APPTAINER_TMPDIR}
rm -rf ${APPTAINER_CACHEDIR} && mkdir -p ${APPTAINER_CACHEDIR}
```

**3**: Facciamo una copia in `/local` del container di base in quanto lettura e scrittura sono molto più veloci in tale directory:  

```bash
cp ${CONT_DIR}/${CONT_NAME} /local/${SLURM_JOBID}/
```

**4**: Creiamo un _Singularity Definition File_  

_nota:_ NGC è il sito web di Nvidia dove mettono i container ottimizzati per le loro GPU (con CUDA, PyTorch, ecc già pronti). L'immagine di base .sif viene da lì; Possiamo partire da queste immagini di base NVIDIA ed estenderle con ciò che ci serve tramite il Singularity Definition File (ciò che vogliamo estendere lo includiamo nella clausola `%post`)  


```bash
cat > custom-pytorch.def << EOF
Bootstrap: localimage 
From: /local/${SLURM_JOBID}/${CONT_NAME}

%post
    apt-get update
    apt-get -y install python3-venv
    pip install --upgrade pip 
    pip install transformers accelerate huggingface_hub
EOF
```

Una volta fatto ciò abbiamo 2 opzioni:  

1. **Creare la sandbox sullo storage del cluster:** 
    Permette di scompattare un container in una cartella normale, essendo una cartella puoi scriverci dentro e modificare i file a piaciemento (aggiungere librerire).  
    Una volta modificata la sandbox a piacimento, dobbiamo ricompattarla in un file .sif per poterla usare in futuro con le modifiche apportate e poterla spostare comodamente.    

```bash
cd ${SCRATCH}
singularity build --sandbox custom-python-sandbox custom-pytorch.def 

# una volta creata la sandbox entriamo in modalità shell 
singularity shell --writable --fakeroot --nv custom-python-sandbox
```

Con questo metodo una volta creata la sandbox vedremo in $SCRATCH una cartella `custom-python-sandbox`, se entriamo vedremo i contenuti come /bin /etc /usr /val ...  
Abbiamo il sistema operativo del container esploso sul disco del cluster.  
Entriamo con `singularity shell --writable --fakeroot`, aggiungiamo ciò che ci serve a mano e quando siamo pronti a sigillare l'immagine faremo:  

```bash
singularity build finale.sif custom-python-sandbox/

# una volta finito e testato puliamo lo storage e cancelliamo la sandbox 
rm -rf custom-python-sandbox
```


<br>

2. **Creare una nuova immagine .sif direttamente**    
    È l'opzione che possiamo usare quando vogliamo creare in un colpo singolo e automaticamente la nuova immagine .sif; Possiamo farlo quando siamo sicuri che tutti gli step in `%post` eseguano senza bisogno di intervento manuale.  

```bash 
cd ${SCRATCH}
singularity build custom-pytorch.def ./custom-pytorch.sif 

# una volta completato possiamo eseguirlo su nodo interattivo e su shell
singularity shell --nv custom-pytorch.sif
```



<br>

<br>

--- 

<br>

### Extra:  


**/global/exafs/groups/...**   

La cartella `/global/exafs/groups/...` è una cartella di sistema gestita dagli amministratori, lì non abbiamo permesso di scrittura!  
Possiamo solo leggere e copiare il contenuto.  
I nostri file e container .sif dobbiamo metterli nella nostra cartella $SCRATCH 

<br>

**/local**:    
Abbiamo la nostra cartella $HOME e $SCRATCH, queste sono visibili **ovunque** (da tutti i nodi del cluster) grazie al fatto che sono montate su un NAS che usa NFS (network file system).  
La cartella /local è privata ad ogni nodo!!! ed è anche per questo che local è velocissimo ed il motivo per il quale viene formattata ogni volta che un job termina (per lasciarlo pulito al prossimo utente).  
Quando facciamo `singularity run --nv container.sif` verrà aperto il container e _automaticamente_ si rendono visibile dentro il container la cartella corrente, la $HOME e la $SCRATCH.  

<br>

**Cosa succede quando apro un container con Singularity?**  



Quando lanciamo `singularity run mio_container.sif` **non avviene nessuno spacchettamento su disco**, non si crea nessuna cartella temporanea con dentro i suoi dati.  
Succede che il Kernel dell'OS di Thea prende il file .sif e lo monta come se fosse un disco virtuale (loopback mount), è un operazione immediata.   
Per il processo dentro il container singularity fa un operazione chiamata **overlay**, sostituisce ciò che vede il processo del container.

- la cartella root `/`: singularity la copre con i file contenuti nel .sif (Ubuntu, librerie, driver), il mio file .sif penserà di essere nella cartella root ma in realtà sarà nel loopback mount.  

- cartelle $HOME e $SCRATCH: singularity da dei 'buchi' nella copertura per permettere al processo .sif di vedere queste directory reali.  
    Tecnicamenete queste 3 cartelle si chiamano **Bind Mounts**.  
    
In un file .sif standard il sistema è **Read-Only** (come masterizzato), se dentro al container provi a scrivere 'touch file.txt' ricevi un errore di read-only file system.  
Non puoi installare librerie o altro al volo dentro il .sif, puoi scrivere solamente nelle cartelle che appartengono all'utente ($HOME, $SCRATCH e /local) perchè quelle sono montate da fuori! non fanno parte del file system del .sif   


<br>

**Ciclo di vita di un container: Dipende dal comando che usiamo!**     

1. `singularity run my_container.sif`  
    Il container nasce, viene eseguita l'azione di default del container (parte subito il programma target) e quando tale programma termina il container **muore**!  

2. `singularity exec my_container.sif python script_diverso.py`  
    Nasce il container, si ignora il comando di default e lanciamo il programma che interessa a noi, finito tale programma muore il container.  

3. `singularity shell my_container.sif`  
    Il container nasce e veniamo trasportati nella sua shell in modalità interattiva!  
    Possiamo avviare il programma che ci interessa e una volta terminato il container rimane in vita! Per uscire dal container dobbiamo scrivere `exit` e solo dopo muore il container.  


<br>


**NAS (hardware) e NFS (protocollo) :**    

- NAS sta per **Network Attached Storage**, è un server che contiene dati, li restituisce a chi li chiede (come un pc dedicato esclusivamente a conservare file)  

- NFS sta per **Network File System**, è il protocollo (set di regole) che permette a un nodo di parlare con NAS; È grazie a NFS se il nodo1 può convincere il suo sistema operativo che una cartella che si trova fisicamente sul NAS sia in realtà 'presente' dentro /home/utente.  

Se sono su un cluster che usa queste tecnologie (come Thea) posso vedere la mia home da qualsiasi nodo! MA i file NON sono distribuiti tra i nodi, i file sono tutti fermi dentro al NAS, i vari nodi montano le cartelle tramite il protocollo NFS.  
Per i nodi è come se i dati fossero locali, ma ogni volta che apri un file, il protocollo NFS corre a chiederlo al NAS in tempo reale.  

<br>


**Rallentare il sistema per tutti:**  

La directory `$HOME` non si trova nel nodo / computer fisico che stiamo usando, si trova su un server di storage centrale (NAS) ed collegato a tutti i nodi tramite la rete infiniband (usando protocollo NFS).  
Il sistema di storage NFS è come un imbuto unico, tutti i nodi devono pasasre per quel singolo cavo/server per leggere la propria home.  

Se lanciamo un programma in `$HOME` che scrive e cancella 10.000 piccoli file al secondo:
- il login node bombarda il server NFS di richieste "scrivi file A", "scrivi file B", ...
- il server NFS ha un limite di operazioni al secondo, e con l'operazione sopra le stiamo consumando tutte.  
- se un utente su un altro nodo prova a fare una `ls` vedrà il suo terminale freezarsi e metterci magari 10 secondi per fare la `ls`
- per questo motivo si usa $SCRATCH che è un sistema parallelo fatto apposta per gestire questo tipo di operazioni.  



**$HOME vs $SCRATCH**:  

- `$HOME`: Usa NAS classico con NFS standard
    - Single-head: un singolo processore (CPU) dentro il NAS che gestisce il traffico 
    - Serializzazione: se devo leggere 10.000 file mando 10.000 richieste a quel singolo processore 
    - anche se il processore è veloce, se le richieste sono troppe si crea una coda 
    - se un altro utente fa `ls` deve aspettare che la coda si liberi (e quindi vedrà un rallentamento notevole e fastidioso!!!) 

- `$SCRATCH`: DDN exascale NVMe, progettato a più server con GPFS(parallel file system)  
    - Multi-head: Non c'à un solo server, ci sono tanti serve che lavorano insieme 
    - **Striping**: quando si copia un file grosso (es.10GB) su $SCRATCH, il sistema lo distribuisce sui vari server (un pezzo su serverA, un altro su serverB, ...)  
    - Accesso Parallelo: quando leggi il file grosso, il nodo scarica i pezzi da tutti server che lo contengono _contemporaneamente_, molto più veloce!  
    - NVMe + RDMA: usano tecnologia NVMe (velocissima) e RDMA (remote direct memory access) su infiniband, permette di fare passare i dati dai dischi alla RAM senza disturbare la CPU.  

**_Nota_**: se rallenti il nodo di login (facendo un lavoro pesante su $HOME), rallenti anche gli altri utenti, anche se loro sono su $SCRATCH e non su $HOME!!  
Il terminale degli altri utenti (shell) dipende dalla $HOME:
- ogni volta che si lancia un comando la shell scrive in `.bash_history` che si trova nella sua home
- il prompt del terminale legge file di configurazione `.bashrc` che si trovano nella home
- le chiavi SSH per connetersi si trovano nella home `.ssh/`  
Quindi se saturiamo la $HOME rallentiamo tutti ovunque si trovino!

