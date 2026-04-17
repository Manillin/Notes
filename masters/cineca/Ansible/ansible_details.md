## Alcuni dettagli di Ansible 


### File di inventario .ini 

creiamo gruppi con nomi racchiusi tra parentesi graffe es: `[workers]`  
Ne possiamo definire quanti vogliamo, sotto ogni gruppo mettiamo gli IP a cui vogliamo collegarci.   

**Collegamento ad IP privati**: in una infrastruttura protetta, abbiamo un singolo entry point / bastion che ha un IP pubblico mentre tutti gli altri nodi vivono protetti nella rete privata.  
Ci possiamo collegare direttamente in ssh solo all'ip pubblico (non possiamo fare ssh verso un IP privato!)    

Creiamo una var (regola) per il bastion host:  

```py
ansible_ssh_common_args='-o ProxyCommand="ssh -W %h:%p -q ubuntu@131.175.207.131 -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no"'
```

- `ansible_ssh_common_args` -> appende al comando ssh la stringa che segue 
- `-o` -> abilita le opzioni ssh (check man ssh) 
- `StrictHostKeyChecking=no` -> non chiede la conferma quando si collega 
- `ProxyCommand = "ssh -W %h:%p -q ubuntu@131.175.207.131"` 
  - -`W %h:%p`: la W significa Window/Forward, dice al bastion host di prendere i dati e inoltrarli direttamente al host %h sulla porta %p  
- `-q` silenzia l'output del bastion

`%h` (Host) e `%p` (Port) sono variabili segrete di SSH.  

Quando chiediamo ad Ansible di eseguire un task su 100 nodi, Ansible fa un ciclo for. Prende il primo IP privato (es. 192.168.0.56) e compone il comando SSH sostituendo al volo le variabili:
- sostituisc `%h` con l'IP target: `192.168.0.56`
- sostituisce `%p` con la porta `22` (ssh)

Il comando reale che esegue il pc da cui lanciamo ansible quando vogliamo collegarci al worker privato `192.168.0.56` diventa:

```bash
ssh -i ~/.ssh/id_ed25519 -o ProxyCommand="ssh -W %h:%p -q ubuntu@131.175.207.131 -i ~/.ssh/id_ed25519" ubuntu@192.168.0.236
```

La versione con ProxyCommand è obbligatoria dentro ansible per retro compatibilità, se vogliamo farlo da un terminale del nostro pc possiamo usare l'opzione `-J`:  

``` bash
ssh -J ubuntu@131.175.207.131 ubuntu@192.168.0.236
```


### blocco copy  

Dentro una task di ansible possiamo usare la funzione `copy`  
copy prende la stringa che scriviamo e la copia dentro la `dest` che va definita dopo la stringa.  

Per fare stringhe con più linee usiamo `content: |` e tutto ciò che sarà indentato sotto sarà una stringa unica!  

Esempio:   
```yaml
- name: configurazione lb  
    copy: 
    content: |
        upstream my_cluster{
        server 192.168.0.236:80;
        }

        server{
        listen 80;
        location / {
            proxy_pass http://my_cluster;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }
        }
    dest: /etc/nginx/sites-available/default 
```

