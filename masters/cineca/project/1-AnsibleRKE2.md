### RKE2 Installation with ansible   

**Traffico nel cluster:**    

In RKE2 è stato aggiunto un supervisore che ascolta sulla porta 9345, serve per le operazioni di day1(registrazione e scambio certificati) - quando creiamo un nuovo nodo , tale nodo va sulla porta 9345 del master e presenta il `rke2_token` segreto che è nel file yaml.  
Il master valida il token e genera i certificiati tls del nodo, li spedisce al nodo worker.  
Ora il worker ha le chiavi, chiude la conessione verso la 9345 e si gira verso la 6443 (l'api-server) per iniziare a lavorare come un nodo k8s normale.   

Il traffico Kubernetes standard va sulla porta 6443 -> rete underlay 

Il traffico applicativo viagga attraverso plugin di rete (CNI - calico/canal) che usa porte UDP  come la 8472 per tunnel VXLAN. (es. ping tra due nodi) -> rete overlay 

Rete dei servizi -> una rete che usa solo IP virtuali che non esistono, questi vengono assegnati ai Service del cluster e servono a dare un VIP a un Service e disassociarlo dall'IP del suo pod.  

Creiamo un ServiceA a cui viene assegnato un VIP. Questo ServiceA è configurato per puntare al PodA (tramite etichette/label). Il PodB contatta il ServiceA tramite il VIP non tramite l'IP del podA. Sarà il kernel a intercettare il traffico e reindirizzarlo verso il PodA (grazie a kube-proxy che scrive le regole di iptables guardando la mappa `endpoints` aka mappa VIP->IP reali )    
In questo modo se il podA muore, il ServiceA viene fatto rinascere su un altro nodo (cambiando IP), il serviceB non deve aggiornare l'IP di ServiceA, continuerà a comunicarci dal VIP.   

Nota: l'Ip virtuale non è assegnato a nessuna scheda di rete, è responsabilità di kube-proxy tradurre il VIP con l'IP vero del pod di destinazione.   


**Rete Overlay e Underlay**:   
Il traffico in K8s si distingue in base alla porta di destinazione, ma in k8s abbiamo **2 reti sovrapposte** -> una Underlay e una Overlay.   

- Rete Underlay: è la rete SDN vera e propria che creiamo con Terraform (e openstack).  
    Dentro ci vivono le VM e gli IP sono quelli fisici delle VM (10.0.05, 10.0.06, ...)  
    Su questa rete viaggiano in chiaro le porte di sistema e di gestione:
    - la 22 (ssh)
    - la 6443 (traffico API tra kubelet e master) 
    - la 9345 (per il day1 di RKE2)  

- Rete Overlay: è la rete wrapper che crea la CNI del cluster.  
    Se Pod A (su worker1) vuole fare un ping a Pod B (su worker2) succede questo:
    - podA crea il suo pacchetto e lo manda nella sua rete overlay
    - il componente di rete di kubernetes (CNI) prende il pacchetto, capisce che deve mandarlo al worker2 e quindi prende l'intero pacchetto del pod e lo chiude dentro un pacchetto UDP
    - il pacchetto viaggia sulla rete openstack (underlay) come traffico UDP
    - quando arriva al worker2 sulla porta 8472 il kernel sa che tale porta è di K8s, passa il pacchetto alla CNI che scarta il pacchetto UDP esterno e tira fuori il pacchetto originale del podA per consegnarlo al podB. (lo spacchettamento lo fa la CNI).  


Nella rete underlay troveremo -> IP dei nodi   
Nella rete overlay troveremo -> IP dei pod   

Le traduzione di indirizzi e l'instradamento di traffico overlay è responsabilità del plugin CNI (attraverso il daemon set che mette un pod su ogni nodo per trattare queto tipo di traffico).  




**Variabili ansible**:  

Nel file main.yaml va dichiarata l'esistenza di tutte le variabili possibili anche se non verranno utilizzate (per evitare errori).  
Quando ansible legge queste variabili e vede che la principale è spenta, salterà tali task ignorandoli (es. se mettiamo `rke_ha_mode_kubevip: false`, tutti i task che installano o usando kube-vip verranno ignorati).   



**VIP e FIP**:  
Sono soluzioni alternative allo stesso problema: come tenere la connessione viva se un nodo master muore.  

VIP -> caso bare metal 
FIP -> caso cloud ( FIP + Octavia (lb))



main.yaml   

```yaml 
#blocca il downgrade 
rke2_allow_downgrade: false

# guarda in che gruppo dell'inventario si trova la macchina attuale
# se in masters la var diventa server; se in workers la var diventa agent  
rke2_type: "{{ 'server' if inventory_hostname in groups[rke2_servers_group_name] else 'agent' if inventory_hostname in groups[rke2_agents_group_name] }}"

rke2_ha_mode: false # mettere a true se vogliamo HA


# definizione di VIP - nel nostro caso probabilmente sarà false -> useremo lb di openstack  
rke2_ha_mode_keepalived: true
rke2_ha_mode_kubevip: false


# assegna a rke2_api_ip l'IP del primo Mater 
# verrà usata per fare il join degli altri nodi al cluster  
rke2_api_ip: "{{ hostvars[groups[rke2_servers_group_name].0]['ansible_default_ipv4']['address'] | default(hostvars[groups[rke2_servers_group_name].0]['ansible_default_ipv6']['address'] ) }}"

# serve per le operazioni di giorno2 
rke2_api_private_port: 9345


# range di VIP che viene usato per i Service lb
# se usiamo openstack lasciamo vuoto 
rke2_loadbalancer_ip_range: {}

# Subject Alternative Names (SANs) sono nomi di dominio aggiuntivi inclusi in un certificato SSL/TLS
# per permettere il collegamento da fuori - va messo il floatingIP di openstack
# TLS serve come certificazione lato client
rke2_additional_sans: []


# permette di cambiare il dominio radice della rete overlay 
rke2_cluster_domain: cluster.example.net

# la password scambiata sulla porta 9345 per autenticarsi con il master 
rke2_token: defaultSecret12345

# definisce la directory radice dell'intero sistem RKE2
rke2_data_path: /var/lib/rancher/rke2

rke2_architecture: amd64
rke2_install_script_dir: /var/tmp
rke2_channel: stable


# permette di bloccare l deploy di alcune componenti (rke2-canal, rke2-coredns, ...)
rke2_disable: []

# attivarlo se si vuole usare cilium 
disable_kube_proxy: false



# avvisiamo l'api server e il kubelet della presenza di provider esterno - es. openstack
rke2_ldisable_cloud_controler: false
rke2_cloud_provider_name: "external"

# ansible potrebbe copiare dei yaml in una cartella speciale, RKE2 appena si accende li esegue
# comodo per configurazioni di sicurezza
rke2_custom_manifests: []
# funzione di sicurezza per pod statici 
rke2_static_pods: []


# gestione di registry per le immagini 
rke2_custom_registry_mirrors: []
rke2_custom_registry_configs: []
rke2_custom_registry_path: templates/registries.yaml.j2


# indica quale file usare come template per generare il file di configurazione finale 
rke2_config: templates/config.yaml.j2

# serve per disaster ricovery -rke2 fa backup di etcd, se stiamo ricostruendo un cluster \
# ansible pescherà da questa cartella gli snapshot vecchi di etcd. 
rke2_etcd_snapshot_source_dir: etcd_snapshots
# salvataggio locale di etcd 
rke2_etcd_snapshot_file: ""
rke2_etcd_snapshot_destination_dir: "{{ rke2_data_path }}/server/db/snapshots"

# definisce il sw che gestirà la rete overlay 
rke2_cni: [canal]

# se inserito RKE2 al momento dell'avvio applic configurazioni di sicurezza restrittive 
# cis (center for internet security) -> creare un sista hardened da subito.  
rke2_cis_profile: ""

# alla fine dell'installazione rke2 genera sul nodo master il file /etc/rancher/rke2/rke2.yaml
# contiene i certificati da amministratore assoluto del cluster (senza fare kubectl dalla workstation fallirebbe)  
# dalla workstation faremo KUBECONFIG=/tmp/rke2.yaml  
rke2_download_kubeconf: true
rke2_download_kubeconf_file_name: rke2.yaml
rke2_download_kubeconf_path: /tmp



# mappa dell'inventario, nomi devono combaciare con inventory.ini
rke2_cluster_group_name: k8s_cluster
rke2_servers_group_name: masters
rke2_agents_group_name: workers





rke2_node_name: "{{ inventory_hostname }}"

# la rete overlay  (per i pod)
rke2_cluster_cidr:
  - 10.42.0.0/16

# la rete dei Service 
rke2_service_cidr:
  - 10.43.0.0/16

```


L'Ansible role di lablabs per installare RKE2 richiede un file di inventari specifico dove i nodi master/server di kubernetes appartengano al gruppo masters, mentre i worker/agent al gruppo worker.  
Entrambi i guppi devono essere children di k8s_cluster

```yaml
[masters]
master-01 ansible_host=192.168.123.1
master-02 ansible_host=192.168.123.2
master-03 ansible_host=192.168.123.3

[workers]
worker-01 ansible_host=192.168.123.11
worker-02 ansible_host=192.168.123.12
worker-03 ansible_host=192.168.123.13

[k8s_cluster:children]
masters
workers
```

