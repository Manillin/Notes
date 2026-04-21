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

---


### Percorsi nei Ruoli (Roles):  
Quando si usa la struttura a ruoli (roles/nome_ruolo/), Ansible applica delle convenzioni rigide per facilitare la vita. Se in un `tasks/main.yml` si usa un modulo come `template` o `copy`, non serve specificare il percorso relativo (es. ../templates/miofile.j2). Basta scrivere `src: miofile.j2` e Ansible cercherà automaticamente nella cartella `templates/` o `files/` di quello specifico ruolo. Questo rende i ruoli portabili al 100%.   


### Il modulo template (Jinja2):
A differenza di copy (che copia un file statico), `template` usa il motore `Jinja2` (.j2). Prende il file sorgente locale, sostituisce dinamicamente tutte le variabili (es. {{ groups['proxy'][0] }}) leggendole dall'inventario o dai facts in fase di esecuzione, e carica il file compilato sul server remoto. È lo strumento fondamentale per rendere le configurazioni dinamiche.

### Stato Desiderato (Modulo service):
Ansible non "lancia comandi", ma garantisce uno stato.

`enabled: true` (o yes) -> Dice ad Ansible di configurare il servizio per l'avvio automatico al boot della macchina (equivalente a systemctl enable).

`state: started` -> Dice ad Ansible di verificare se il servizio sta girando in questo esatto momento. Se è spento lo accende. Se è già acceso, non fa nulla e restituisce ok (*Idempotenza*).

