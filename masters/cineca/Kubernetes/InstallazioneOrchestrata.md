# Installazione Orchestrata   


### Ansible 

Il tool che useremo è Ansible 

Uno strumento progettato per automatizzare operazioni massive sui nodi, lancia comandi ssh senza installare nulla a bordo dei nodi. (TODO)   

Ansble segue il seguente flusso:  
- Viene definita la lista degli host (file di inventario) (dove vogliamo lanciare la nostra automazione)  
- Viene definito un **Playbook** - un insieme di file in YAML con una lista di ruoli da applicare  
- Viene eseguito il playbook 
  

I **ruoli** sono gli elementi dei playbook e sono costituiti da: 
- Una serie di *task*, lanciati in ordine 
  - task: sono pezzi di ansible (funzioni di libreria aka moduli) che possiamo lanciare 
  - i task possono avere dei modificatori (es. essere resi condizionali)
- una serie di file template



playbook ha la forma di una lista in formato YAML   

i task del playbook vengono eseguiti in parallelo su tutti i nodi, se vogliamo forzare l'esecuzione sequenziale dobbiamo specificare `serial: 1`   


Playbook ha i ruoli, per ogni ruolo si crea un folder.  

Dentro il folder di un ruolo avremo:
- task: folder che contiene il main.yaml -> il laovoro che va fatto (es. scaricare librerie ecc..)
- defaults: contiene un file yaml dove possiamo specificare le variabili di ambiente 
- templates: 