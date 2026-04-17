
# Comandi Linux Utili 

- lscpu: mostra dettagli sul dispositivo (arch, distro, CPUvendor,...)
  
- ls -l /dev/kvm: se esiste vuol dire che la virtualizzazione è attiva sul host 

Cambiare permessi sui file: 

- chmod: modifica dei permessi di un file [Owner - Groups - Others]
  - valore ottale: 4 = Lettura; 2 = Scrittura; 1 = Esecuzione  
  - Es: 0755 -> User=r,w,x; Groups=r,x; Others=r,x
  - `chmod 0600 file.txt`


Uccidere un processo in esecuzione:  

Per prima cosa dobbiamo cercare il PID del processo:
- `ps` -> process snapshot: restituisce uno snapshot dei processi in esecuzione  
  - opzioni raccomandate `aux` -> a: mostra tutti gli utenti; u: mostra il proprietario del processo; x: mostra anche i processi non attaccati a un terminale   

- `grep` -> cerchiamo un match nell'output che fornisce ps 

Un esempio potrebbe essere: `ps aux | grep ssh`  

Per trovare un processo possiamo anche usare `top` o `htop` e guardare il consumo della CPU e RAM.    

Una volta che troviamo il pid del progetto possiamo usare `kill`, ci sono due modi:
- kill <PID>    : permette al processo di salvare dati prima di ucciderlo
- kill -9 <PID> : uccide il processo all'istante (utile nel caso un processo sia freezato)   

