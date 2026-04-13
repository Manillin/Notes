
### Sviluppo progetto:   



**Definizione della rete**   

Creare una private subnet dove mettere i nodi che crea openstack  

Configurare un FIP per comunicare con la rete pubblica
- FIP su nodo master 
- FIP su nodo dedicato bastion 
Nota su bastion: una VM lightweight che prende il floatingIP (è lunica esposta), i nodi del cluster hanno tutti IP privati.  
Dall'esterno bisogna prima passare sul bastion e una volta dentro il baston si può andare verso i nodi interni -> concetto di jump host.  
Queste considerazoni valgono per il traffico di gestione e per collegarsi con ssh su un nodo master.   


**Crezione dei nodi con terraform**:   


Ci son due modi per definire i nodi:  

1. Metodo default (stampino):  
   Si usano varabili semplici per definire quante copie vogliamo, TUTTE le macchine avranno lo stesso flavour, comodo per cluster omogenei.  

2. Metodo Custom Definition:  
   Se vogliamo fare un cluster eterogeneo usiamo questo metodo.   
   usiamo la variabile `k8s_nodes`, per abilitarlo dobbiamo:  
   - `number_of_k8s_nodes` e `number_of_k8s_nodes_no_floating_ip` a `0` e definire `k8s_nodes`.   
    k8s_nodes = {
        "ing-1" = {...},
    }  


Quando usiaamo il metodo custom dobbiamo creare un dizionario con i nodi e per ogni nodo definire 3 cose:  
1. az (availability zone)
2. flavor
3. floating_ip: boolano a true o false 

Ci sono anche altri parametri opzionali che possiamo definire.  

Lo schema è:  
```terraform
k8s_nodes = {
  "key | node name suffix, must be unique" = {
    "az" = string
    "flavor" = string
    "floating_ip" = bool
  },
}
```

Per esempio:  

```terraform
k8s_nodes:
   node-name:
    az: string # Name of the AZ
    flavor: string # Flavor ID to use
    floating_ip: bool # If floating IPs should be used or not
    reserved_floating_ip: string # If floating_ip is true use existing floating IP, if reserved_floating_ip is an empty string and floating_ip is true, a new floating IP will be created
    extra_groups: string # (optional) Additional groups to add for kubespray, defaults to no groups
    image_id: string # (optional) Image ID to use, defaults to var.image_id or var.image
    root_volume_size_in_gb: number # (optional) Size of the block storage to use as root disk, defaults to var.node_root_volume_size_in_gb or to use volume from flavor otherwise
    volume_type: string # (optional) Volume type to use, defaults to var.node_volume_type
    network_id: string # (optional) Use this network_id for the node, defaults to either var.network_id or ID of var.network_name
    server_group: string # (optional) Server group to add this node to. If set, this has to be one specified in additional_server_groups, defaults to use the server group specified in node_server_group_policy
    cloudinit: # (optional) Options for cloud-init
      extra_partitions: # List of extra partitions (other than the root partition) to setup during creation
        volume_path: string # Path to the volume to create partition for (e.g. /dev/vda )
        partition_path: string # Path to the partition (e.g. /dev/vda2 )
        mount_path: string # Path to where the partition should be mounted
        partition_start: string # Where the partition should start (e.g. 10GB ). Note, if you set the partition_start to 0 there will be no space left for the root partition
        partition_end: string # Where the partition should end (e.g. 10GB or -1 for end of volume)
      netplan_critical_dhcp_interface: string # Name of interface to set the dhcp flag critical = true, to circumvent [this issue](https://bugs.launchpad.net/ubuntu/+source/systemd/+bug/1776013).
```

**Ciclo di Vita**:   

Per inizializzare e scaricare il provider per parlare con openstack facciamo la `init` (occhio alle cartelle).   

Possiamo usare `Cloud-init` per fare la configurazione iniziale delle macchine in cloud.  
-> terraform ordina a openStack di creare una VM, openStack la crea e la accenda, il OS fa il boot per la rima volta e in questa fase legge il file YAML di Cloud-init fornito da terraform per eseguire i comandi.  

ES:  

```tf
#cloud-config
## in some cases novnc console access is required
## it requires ssh password to be set
ssh_pwauth: yes
chpasswd:
  list: |
    root:secret
  expire: False

## in some cases direct root ssh access via ssh key is required
disable_root: false
```

Per debugging -> `OS_DEBUG=1` e `TF_LOG=DEBUG`.   

Destroy -> smonta tutto, e si consiglia di pulire il file ~/.ssh/known_hosts, altrimenti il pc di ricorderà le vecchie macchine e quando ne creerà delle nuove con gli stessi IP, SSH si bloccherà.  


Terraform Output: terraform fornisce in output valori che sono utili per la configurazione del deployment di k8s, come:
- private_subnet_id -> la subnet su cui girano le macchine, serve per traffico interno 
- floating_network_id -> serve per il traffico esterno   


**Ansible**:   
Anible è lo strumento che, una volta che le VM sono accese, ci entra dentro via SSH e installa i sw che servono -> k8s.  

Impostiamo il ssh-agent con la chiave ssh, è uno step richiesto dal terraform proviisioner.  

eval $(ssh-agent -s)
ssh-add ~/.ssh/id_rsa  

Per fare funzionare ansible serve almeno un bastion host (il traffico ssh passerà sempre da questi bastion) o almeno un nodo master con un FIP.  

Fare il test di ping ansible per capire se la configurazione delle macchine è come ce la aspettiamo, se è tutto in regola possiamo iniziare a installare kubernetes.  

