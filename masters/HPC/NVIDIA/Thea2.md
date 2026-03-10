# thea pt 2


Note sui concetti della seconda call:   

### Esploarzione e diagnostica hardware:  


- `nvidia-smi topo -m`: dice come sono collegate le GPU in un sistema

- `ibstat`: controllare lo stato dell'infiniband, in particolare informazioni utili sono:  
    - `state: Active`: dice se è attiva o meno;  `Rate` (es. 200) dice la velocità teorica massima del cavo.  
- `ibv_devinfo`: uguale ad ibstat ma verbose   

Il concetto è assicurarsi che le GPU usino le schede di rete più vicine per parlare con la rete (sullo stesso socket CPU).   


<br>

--- 


### Amministrazione del cluster   

I nodi solitamente non hanno l'OS installato su un disco rigido fisico permanente, si vuole avere un cluster con nodi 'stateless'.    
Al boot time il nodo scarica l'OS dalla rete e lo carica in RAM.  

Vantaggio: se un nodo si incasina o ha problemi, basta riavviarlo (reboot) e tornerà pulito come nuovo. Inoltre se volessimo aggiornare 1000 nodi dobbiamo solo di cambiare l'immagine centrare e riavviarli.   

Una volta  che l'OS è nella RAM, è nudo, si usa **Ansible** o **Puppet** per definire script automatici che settano l'OS come vogliamo (es: installa GCC, crea utente A, config rete, monta cartella condivisa, ...)   
Questo rende la configurazione **Riproducibile**.     


- `pdsh`: parallel distributed shell; permette di lanciare lo stesso comando su un subset di nodi o su tutti i nodi del cluster (`pdsh -w nodo[1-100] date`).  
    Evita di dover dare un comando manualmente su ogni nodo dovendo fare ssh tutte le volte.  


- `pdcp`: parallel distributed copy; permette di copiare un file dal nodo master a tutti i nodi figli slave specificati.   


RPM (Red Hat Package Manager): è il formato dei file di installazione per Linux (distro RedHat, CentOS, RockyLinux - che sono quelle maggiormente usate nei cluster).   


<br>

---


## Benchmarks:        


### Benchmark Memoria RAM 

RAM CPU: Si usa il programma `stream.c` ottenbile qui: [stream.c](https://www.cs.virginia.edu/stream/FTP/Code/stream.c) per misurare la banda della memoria RAM ( in GB/s).  

Per compilarlo:  

```bash
gcc -Ofast -march=native -fopenmp -mcmodel=large -fno-PIC -DSTREAM_ARRAY_SIZE=120000000 -DNTIMES=200 -o stream_openmp.exe stream.c
```

- `-Ofast`: compila per massima velocità
- `-march=native` usa tutte le istruzioni speciali della CPU in uso (es. SVE su ARM)
- `-fopenmp`: abilita parallelismo su CPU 
- `-mcmodel=large` permette di usare piu di 2GB di memoria statica [TODO: in che senso??]
- `-DSTREAM_ARRAY_SIZE=N` definisce quanto è grosso l'array, la regola è che deve essere molto più grande della cache della CPU, altrimenti misuriamo la velocità della cache e non della RAM.   

Il valore più significativo è quello di **Triad** che è la combinazione delle operazioni più complesse, il suo risultato indica i MB/s reali!    


### Benchmark di Rete e GPU  

Testiamo la velocità di comunicazione tra nodi (MPI) e tra GPU  

Bisogna compilare da zero OSU:   

```bash
./configure --prefix=/path/to/destination CC=/home/nvidia/.../hpc_sdk_259/Linux_aarch64/25.9/comm_libs/hpcx/bin/mpicc CXX=/home/nvidia/.../hpc_sdk_259/Linux_aarch64/25.9/comm_libs/hpcx/bin/mpicxx --enable-cuda --with-cuda-include=/home/nvidia/.../hpc_sdk_259/Linux_aarch64/25.9/cuda/12.9/include --with-cuda-libpath=/home/nvidia/.../hpc_sdk_259/Linux_aarch64/25.9/cuda/12.9/lib64   
```

Per testare i benchmarks usiamo `mpirun`:   


```bash
# Latency 
mpirun --allow-run-as-root -np 2 --host lego-cg1-qs-183:1,lego-cg1-evt-06:1 -x LD_LIBRARY_PATH -mca coll_hcoll_enable 0 -mca coll_ucc_enable 0 -x UCX_NET_DEVICES=mlx5_0:1 -x UCX_TLS=rc_x,self,shm  --map-by dist -mca rmaps_dist_device mlx5_0 /home/nvidia/.../osu-25.1-cuda/libexec/osu-micro-benchmarks/mpi/pt2pt/osu_latency >/home/nvidia/.../logs_osu/osu_lat_1836.log

# Bandwidth 
mpirun --allow-run-as-root -np 2 --host lego-cg1-qs-183:1,lego-cg1-evt-06:1 -x LD_LIBRARY_PATH -mca coll_hcoll_enable 0 -mca coll_ucc_enable 0 -x UCX_NET_DEVICES=mlx5_0:1 -x UCX_TLS=rc_x,self,shm  --map-by dist -mca rmaps_dist_device mlx5_0 /home/nvidia/ftartaglione/osu-25.1-cuda/libexec/osu-micro-benchmarks/mpi/pt2pt/osu_bw > /home/nvidia/ftartaglione/logs_osu/osu_bw_1836.log
```


Risultati - (Ideale = High Bandwidth low Latency):  
- **Bandwidth:** Si vogliono avere numeri alti   
    - se `ibstat` dice 200 Gb/s dobbiamo aspettarci circa 23.000 - 25.000 MB/s 


- **Latency:** Si vogliono avere numeri bassi!  
    - Puntiamo ad ottenere risultati sotto i 2ms (microsecondi - us) che sono l'ideale per infiniband  


Mentre eseguiamo un benchmark che usa la GPU possiamo usare il comando `nvidia-smi` per controllare lo stato del nodo.   

- Apriamo due connessioni ssh al cluster 
- entriamo allo stesso nodo di calcolo con salloc 
- su una shell lanciamo mpirun.... 
- sull'altra lanciamo `watch -n 1 nvidia-smi`

 