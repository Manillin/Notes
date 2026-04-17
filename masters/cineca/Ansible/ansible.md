# Ansible  

Ansible è agentless, non esiste nessun programma costantemente in esecuzione sulle macchine worker o master di openstack.  

**File di inventario** -> file generato (idealmente) da terraform che contiene gli IP dei server, dice ad ansible dove sono i nodi su cui voglio fare l'automazione.  

**Playbook** -> possiamo dare il nome che vogliamo. è il regista da cui parte l'esecuzione.  

**File delle variabili** -> definiamo le variabili che ci interessano e che faranno override delle defaults definite nel ruolo ansible  

### Gathering facts 
Quando si lancia un playbook, la prima cosa che fa ansible è eseguire un modulo speciale chiamato `setup` su tutte le macchine target.   

Il modulo lancia uno script python sulla macchina remota che interroga l'OS e crea un dizionario json con tutto quello che deve sapere su quella macchina.  

il nome del dizionario è fisso e predefinito da Ansible, si chiama `ansible_facts`.  

Possiamo accedere a questi dati dinamicamente dal playbook, navigandolo come un dizionario py.  
- estrarre l'IP della prima scheda di rete: `{{ansible_facts['default_ipv4']['address']}}`  
- sapere la famiglia dell'OS: `{{ansible_facts['os_family]}}`    

Per vedere tutto il dizionario possiamo usare `ansible -m setup -i inventory.ini all`   


### Ruoli:   

Un ruolo è un 'mini-progetto indipendente' progettato per fare una cosa specifica (es. installare rke2, configurare il firewall, ...).  
Se vogliamo creare un ruolo dobbiamo rispettare una struttura specifica.  

Dentro un ruolo la cartella si deve chiamare `tasks` e il file principale li dentro deve chiamarsi `main.yaml`.  

Dentro la cartella dei ruoli avremo una cartella `defaults/` dove ci sarà un `main.yaml` con lla definizione della variabili di default.  
Queste verranno usate di default, ma se ne definiamo altre nella dir prima o quando passiamo il comando verranno schiacciate e sostituite da quelle nuove.    


Quando si importa un ruolo (es. ruolo lablabs per rke2) non si modificano MAI le variabili default!  
Si definisce un file `my_vars` dove si fa overriding delle variabili che ci interessano, e quando si lancia il playbook si passa questo file per sovrascriverle.  

```yaml
---
- name: Deploy RKE2 Cluster su OpenStack (ADA)
  hosts: all
  become: yes
  
  # Importo il mio file con le variabili per schiacciare i default
  vars_files:
    - mie_variabili_rke2.yaml
  roles:
    - role: lablabs.rke2
```


---



Quando lanciamo un playbook:
1. anible legge il playbook
2. si collega in ssh alle macchine elencate nell'inventario 
3. ogni task lo traduce in uno script py temporaneo 
4. copia questo script sulla macchina remota
5. esegue lo script, prende il risultato e cancela lo script per non lasciare tracce.    

Il concetto di Idempotenza: È la regola d'oro di Ansible. Uno script Ansible non dice "Installa NGINX", ma dice "Assicurati che NGINX sia presente", definiamo quelo che vogliamo essere il nostro **stato finale**. Se lo lanci 10 volte di fila, la prima volta lo installerà (stato Changed), le successive 9 volte vedrà che c'è già e non farà assolutamente nulla (stato Ok).  






### Esempio:
Automatizzare installazione di un server apache.  

```
mio_progetto/
├── inventory.ini             <-- Generato da Terraform
├── le_mie_variabili.yml      <-- File separato per le vars
├── master_playbook.yml       <-- Il tuo copione principale
└── roles/
    └── apache_ruolo/         <-- Il nostro mini-progetto (Ruolo)
        ├── defaults/
        │   └── main.yml      <-- Variabili di base del ruolo (DEVE chiamarsi main.yml)
        └── tasks/
            ├── main.yml      <-- L'indice dei task (DEVE chiamarsi main.yml)
            ├── installa.yml  <-- Task per installare
            └── avvia.yml     <-- Task per accendere il servizio
```


nel file di inventario avremo:
```
[webservers]
10.0.0.50
```

nel file delle variabili avremo:

```
porta_del_sito: 8080
```


Il ruolo `apache_ruolo` ha:

- `roles/apache_ruolo/defaults/main.yaml`
    ```
    porta_del_sito: 80
    ```

- `roles/apache_ruolo/tasks/installa.yaml`:

    ```yaml
    - name: installazione di apche
    apt:
        name: apache2
        state: present
    ```
- `roles/apache_ruolo/tasks/avvia.yaml`:
    ```yaml
    - name: accendere apache 
    service:
        name: apache2
        state: started
    ```

- `roles/apache_ruolo/tasks/main.yaml`:
    ```yaml
    - include task: installa.yaml
    - include task: avvia.yaml
    ```

Il playbook vero e proprio sarà corto grazie alla modularità che abbiamo applicato:


```yaml
- name: Installa il mio server web
  hosts: webservers
  become: yes
  vars_files:
    - le_mie_variabili.yml
  roles:
    - role: apache_ruolo
```

Lo avviamo con: `ansible-playbook -i inventory.ini master_playbook.yml`   

- ansible legge che vogliamo colpire il gruppo `webservers` e si collega all'IP specificato nel file di inventario 
- carica `le_mie_variabili.yaml`
- entra in `roles/apache_ruolo`
- legge i default del ruolo ma la variabile definita prima vince e la schiaccia
- va in `tasks/main.yaml` e include installa.yaml e avvia.yaml
- esegue l'istallazione e lavvio in modalità idempotente  



