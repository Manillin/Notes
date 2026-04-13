




### Replica set  

Permette di lanciare un pod un certo numero di volte   

Per vedere il replica set:    
`kubectl get replicaset`     

describe ci dà molte più informazioni!  

`kubectl describe replicaset alpine-replicaset`   

kubectl describe daemonsets.apps alpine-daemonset

crictl -> server per amministrare i container dentro un nodo 


Per debuggare un nodo; kubectl exec lancia un comando all'interno di un container passando per l'api-server (se ci sono più container dentro al pod va specificato il container con -c)   

`kubectl exec -it alpine-replicaset-6xl5q -- /bin/sh` -> crea shell dentro container    


kubectl debug -> mi aggancio a un pod con un container di debug  



kubectl events 



Per cancellare possiamo:
- eliminare per nome 
- eliminare a partire dal file minfest che ha creato la risorsa 
    - `kubectl delete -f replicaset.yaml`




il deployment ha sotto un replica set 


i template  dei job sono immutabili, se lanciamo il job e poi modifichiamo il template e proviamo a riavviarlo k8s si arrabbia -> field is immutable  

se vogliamo rischedulare un job, bisogna prima eliminarlo con delete e poi rifare apply.   



Il cron job ha il campo schedule dove si specifica ogni quanto va fatto ripartire il job (con la sintassi di cron) e ha il job template.   



---


Stateless applications:  

Un applicativo deployato su kubernetes è una cartella di file yaml

deployment.yaml  

Sarà un nginx replicato, carica due file di configurazione 
1 a partire da una configmap e uno da un secret

l'app consiste in un deployment con due repliche di un pod con queste specifiche:  
1 nginx che espone la porta 80 che monta due volumi, html-config (path) e html-secret (path).   

questi mount di volumi sono riferiti per nome (i volumi sono comuni a tutto il pod, non vanno definiti sotto al container ma nel campo volumes.)  

sotto al campo volumes definisco i due volumi per nome:
html conig ha una config map (che si chiama nginx-html)
html secret ha un secret (che si chiama nginx-secret)


se io faccio solo il deployment del deployment, i pod rimarranno bloccati in containercreating.  
questo perchè il container vuole montare una config map e un secret, ma non ci sono ancora, qundi non potranno mai partire questi container (il pod parte ma ma rimane in stato di transizione in quanto il container non può partire)  

-> posso accorgermener facendo describe pod  
e troviamo che il pod non è pronto perchè non ha montato i volumi ma è stato inizializzato.  
negli eventi vediamo proprio che i volumi html-secret e html-secret non sono stati creati perche manca il secret e la configmap.  


quindi creiamo la configmap e il secret (il secret è la stessa cosa ma codificata in base64)  

-> creo le due risorse e ricreo il deployment che questa voltà verrà creato con successo.   

una configmap e un secret NON sono container/pod. Sono dei blocchi di testo salvati su etcd.  
Quando facciamo: kubectl apply -f configmap.yaml:
- Il file YAML viene inviato all'API Server (sul tuo nodo Master, passando per l'HAProxy).
- L'API Server prende il testo che hai scritto sotto data: (il tuo codice HTML) e lo salva nel database etcd.
- In questo momento, la ConfigMap è solo inchiostro digitale su un disco del Control Plane. Non fa assolutamente nulla. Sta lì ad aspettare.


kubectl port-forward pod/... 8080:80 -> apre un tunnel tramite l'api server    

---


Il Service permette di assegnare a un insieme di pod una risoluzione dei nomi.   

Servizio clusterIp e servizio LoadBalancer.  

il servizio load balancer ha un Ip e una porta esterna (kubectl get svc per vederlo)  

quindi se voglio espoorre una risorsa solo dentro il cluster faccio un servizio di tipi clusterIP, e se voglio esporlo anche all'esterno faccio un servizio di tipo LoaadBalancer (posto che io abbia un fornitore di lb dentro il mio cluster)   


Quando creiamo app stateful -> allochiamo allocazione e dobbiamo sempre usare statefulset e possiamo metterle dentro a un servizio headleass.  

quando si cancella uno stateful set, rimangono comunque i volumi (per evitare il data loss).  
L'unico modo per cancellare i pvc è farlo esplicitamente con kubectl.  



Ingress: 
l'astrazione del service funziona se parlaimo di tcp e udp, ma al giorno d'oggi si usa molto http/https.   

all'interno delle primitive di rete, k8s aggiunge la primitiva di ingress, progettata per i servizi http e https (dalla versione 1.19)  

-> l'astrazione successiva che rimpiazza ingress è "Gateaway"  


pattern: client entra dafuori, colpisce un lb gestito da ingress, colpisce un terminatore e sulla base dell'host che ha richiesto lo di dispaccia al servizio.    

