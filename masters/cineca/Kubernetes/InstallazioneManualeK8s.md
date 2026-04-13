# Installazione manuale Kubernetes:   

Ci saranno 5 VMs (node1-node5) e un host su una rete piatta:   

172.16.42.[10-50] -> sono i 5 nodi 
172.16.42.1 -> host 


Tramite uno script bash si crea un servizio DNS che avvia `dnsmasq` sul host e lo mette in ascolto su 172.16.42.1 (IP dell'host) -> tutte le macchine virtuali sanno che se vogliono risolvere un nome devono chiedere a quell'IP  

Conviene usare i nomi DNS invece degli IP per flessibilità:  
Ogni componente K8s comunicca con gli altri usando certificati crittografici TLS, quindi quando si genera un nodo bisogna specificare a chi appartiene (Subject Alternative Name) e:
- se usiamo solo l'IP: il certificato è valido solo per chi ha quello specifico IP (se un giorno dobbiamo cambiare sottorete o l'IP di quel nodo, tutti gli altri nodi rifiuteranno la connessione, bisogna distruggere e ricreare quel nodo!)
- se usiamo il nome DNS il certificato sarà valido per chi ha quel nome (es. node1), se l'IP cambia, basta aggiornare il record DNS. Il certificato rimane valido perchè il nome non cambia.  



Virtual IP: 
Abbiamo un altro record DNS: api-lb.k8scourse.serics.eu -> 172.16.42.5  

è un indirizzo IP che non appartiene a nessun nodo, serve ai nodi worker per comunicare con i nodi di control plane (o master nodes).  
Servirà fare il setup con keepalived e haproxy che vederemo dopo.   


Ulteriori record DNS-SRV (Service):      
Abbiamo definito anche:  

SRV 6443 1 1 _k8s-server._tcp.k8scourse.serics.eu node1.k8scourse.serics.eu  

SRV 6443 1 1 _k8s-server._tcp.k8scourse.serics.eu node2.k8scourse.serics.eu  

SRV 6443 1 1 _k8s-server._tcp.k8scourse.serics.eu node3.k8scourse.serics.eu  

Si tratta di un DNS-SRV che è diverso da un DNS di tipo A:  
- DNS A : risolve solamente il nome in un IP, NON specifica mai una porta (se abbiamo un server web sulla porta 80 e un server DB sulla porta 5432 sulla stessa macchina, il record A non fa distinzioni)  
- DNS-SRV: Risponde a una domanda più strutturata e complessa: "dove trovo il servizioX, su quale protocollo, a quale porta e in quale ordine di priorità devo contattare i server che lo offrono".  
    nel nostro caso avremo:
    - porta 6442 -> il client DNS sa esattamente a quale porta connettersi 
    - Priorità (1) -> Definisce l'ordine di failover (se avessimo node1 con priorità 1 e node2 con priorità 10, i client andrebbero sempre e solo su node1, contatterebbero node2 solo se node1 fosse irraggiungibile)  
    - peso (1) -> definisce il bilanciamento del carico, mettere peso 1 su tutti i nodi distribuisce il carico in modo equo (RoundRobin). se node1 avesse peso 3 e node2 peso 1, allora node1 riceverebbe il triplo del carico.  

NOTA: il kubelet che gira sui nodi si aspetta un URL https, abbiamo definito l'IP virtuale (172.16.42.5) assegnato a api-lb... I worker punteranno esclusivamente a quell'indirizzo per parlare con il control plane (api-lb... sarà un Load Balancer!)    
I consumatori del record DNS-SRV servono ai componenti dell'infrastruttura sottostante, per due scopi:
1. etcd discovery: etcd è il DB distribuito che fa da cervello. Per formare un cluster le istanze di etcd sui nodi master devono trovarsi a vicenda. Invece di passare una lista statica di IP al boot, etcd supporta il DNS discorvery tramite DNS-SRV
2. Haproxy / LoadBalancer dinamico: Si usa un LB che ascolta sul virtua IP (.5). invece di scrivere nel suo file di config (haproxy.conf) l'IP di ogni singolo master, impostiamo HAproxy per interrogare il record DNS-SRV. Quando il DNS restituisce node1, node2, node3, HAProxy aggiorna i suoi backend dinamicamente. Se domani aggiungiamo node4 al DNS, HAProxy inizia a mandargli traffico senza che il bisogno di riavviarlo.
 
Quindi quando un nodo worker vuole parlare con un nodo master avviene questo:  
1. **Kubelet verso virtual IP**:Il kubelet del nodo worker si aspetta un singolo IP -> contatta il load balancer che abbiamo impostato nel IP virtuale (haproxy).
2. **LoadBalancer verso i master (Service Discovery SRV)**:Il pacchetto arriva in mano ad haproxy, il cui compito è prenderlo e inoltrarlo a uno dei nodi master (uno tra node1,2,3 nella porta 6443). Haproxy ha la capacità di interagire con DNS-SRV, quindi interroga il record che gli risponde dicendo dove trovare i nodi e su quale porta specifica è il servizio
3. A questo punto Haproxy inoltra la richiesta al nodo corretto   


NOTA2: l'ip virtuale `172.16.42.5` è di proprietà esclusiva di un nodo. Se in un determinato istante il node1 è il leader, il daemon keepalived aggiunge l'IP .5 alla sua scheda di rete (come IP secondario). Ovviamente serve che questo IP ruoti in caso di failover, altrimenti i worker non saprebbero più chi chiamare per contattare l'api-server.  
Per questo l'IP ruota, keepalived implementa un protocollo chiamato *VRRP(virtual router redundancy protocol)*:
- i daemon keepalived (sui nodi master) si mandano heartbeat
- se il node1 (attuale possessore dell'IP) muore, i node2 e node3 smettono di sentire il heartbeat
- indicono un elezione per eligere un nuovo nodo su cui fare abbordare l'IP. la vince il node2
- il daemon keepalived di node2 aggiunge l'IP .5 alla sua scheda di rete.  


Haproxy è un normale daemon in esecuzione, nell'architettura stacked che faremo, andrà messo in esecuzione su titti e 3 i nodi di control plane.  
Haproxy è configurato per ascoltare il traffico in ingresso sull'IP virtule .5 e poichè nella nostra architettura stacked tale IP virtuale esiste solo su un nodo alla volta, SOLAMENTE l'Haproxy del nodo leader in quel momento riceverà effettivamente il traffico dai worker. (gli Haproxy degli altri due nodi rimarranno in ascolto su un IP che localmente non hanno, aspetteranno che avvenga un failover per entrare in gioco)   

In architetture di granda scala si usa: BGP + ECMP (l'IP lo hanno più macchine che fanno solamente da load balancer, per evitare il bottleneck).  


ulteriore record dns che punta a un registro container:  
registry.k8scours.serics.eu -> 172.16.42.142


---


### Installare la CRI (Container Runtime Interface)  

Si installa `CRI-O`, una container runtime interface progettato espressamente per k8s e minimale.   



### Installare la CNI (Container Network Interface)  

Si installa `kube-router` , che è semplice e privo di add-on rispetto a k8s-vanilla 

### Installare la Container Storage Interface (CSI)  

è un NFS CSI plugin, storage distribuito in rete.  
Non HA e installato sul primo nodo.   

---

### Metodo di installazione (kubeadm)  

`kubeadm` è l'unico metodo ufficilae supportato upstream


### Control Plane endpoint  

Il metodo raccomandato per HA dalla documentazione è: `keepalived` + `haproxy`   

Il Control Plane endpoint è un entità logica composta da tre pezzi:

1. IP floattante: gestito da keepalived. è l'indirizzo IP a cui tutti puntano e si sposta fisicamente tra i 3 nodi di control plane. se il nodo1 si rompe, keepalived sposta l'IP sul nodo2 o sul nodo3 in una frazione di secondo. I client non se ne accorgono 
2. Load Balancer: è il 'vigile urbano' che ascolta sull'IP flottante. prende il traffico in ingresso e lo distribuisce equamente ai 3 nodi di control plane per non sovraccaricarli.  
3. Nome DNS: i kubelet punteranno a quanto nome, non all'IP hardcoded, questo per blindare i certificati ed essere flessibili.  



Fornisce Load Balancing + Failover , ip flottante che si sposata tra i master, se uno muore viene spostato in modo trasparente tra i master.  


Service discovery tramite DNS-SRV record     

Endpoint del control plane definito per nome: rende possibile cambiare gli IP senza invalidare i certificati. In questo modo se il lb vuole sapere dove sono i node master deve fare una query DNS-SRV   

Se si spacca il DNS si rompe tutto!  


--- 


---  

## Installazione Manuale di Kubernetes  

Per installare un cluster K8s a mano dobbiamo: 

1. Preparare i Nodi
2. Installare la CRI
3. Preparare il load balancer per il control plane 
4. Bootstrap del server 
5. Join dei nodi server 
6. Joind dei nodi worker 
7. Installazione di servizi aggiuntivi 


### 1. Preparare i Nodi  

Dobbiamo seguire i seguenti controlli:
1. MAC address univoco per ogn nodo (`ip link`)
2. hardware id univoco per ogni nodo (`sudo cat /sys/class/dmi/id/product_uuid`)
3. Swap disattivato (`sudo swapoff -a`, modificare `/etc/fstab`, `systemd.swap units`)
4. Hostname univoco (`cat /etc/hostname`)
5. Alias a localhost in `/etc/hosts` (localhost deve essere 127.0.0.1)
6. Aidacenza L3 con gli altri nodi ()


### 2. Installare la CRI  
Installeremo CRI-O  
Per installarlo possiamo seguire la guida alla pagina di [cri-o](https://cri-o.io/)  

Impostare le variabili di ambiente + apt-get update + apt-get install -y software-properties-common curl + apt install gpg + curl (sotto) + apt-get update (di nuovo) + apt-get install -y cri-o + systemctl start crio.service + systemctl status crio (per vedere se è attivo)  

CRIO_VERSION=v1.33  
KUBERNETES_VERSION=v1.33  

curl: 
```bash
curl -fsSL https://download.opensuse.org/repositories/isv:/cri-o:/stable:/$CRIO_VERSION/deb/Release.key |
    gpg --dearmor -o /etc/apt/keyrings/cri-o-apt-keyring.gpg

echo "deb [signed-by=/etc/apt/keyrings/cri-o-apt-keyring.gpg] https://download.opensuse.org/repositories/isv:/cri-o:/stable:/$CRIO_VERSION/deb/ /" |
    tee /etc/apt/sources.list.d/cri-o.list

```


### 3. Control Plane Endpoint  

Prima di far partire K8s dobbiamo dare un identità al nostro control plane (perchè il kubelet per far partire i suoi pod contatta l'api-server, ha bisogno di avere un identità subito!)   

Quindi dobbiamo pensare da subito a come fare il load balancer.   
In ambiente bare metal metteremo:
- un lb su ogni nodo server
- un indirizzo IP flottante condivio tra i nodi server 
- un nome DNS associato all'IP flottante  

Usiamo `keepalived` (per avere IP flottante) che permette di condividere indirizzi IP tra più host, usando il protocollo VRRP (virtual router redundancy protocol)

apt install keepalived 

Configurare keepalived.conf 
il file di config definisce degli indirizzi IP e come sincronizzarli tra più nodi  
Contiene al suo interno una serie di blocchi che si chiamano VRRP instance  

keepalived lavora a livello 2/3, il suo unico scopo è manipolare i MAC address e gli indirizzi IP fisici sulle schede di rete, sposta solamente l'indirizzo VIP da un server all'altro.  


Usiamo `Haproxy` come load balancer, ossia colui che ridistribuirà il carico agli API-Server  

apt install haproxy 

La configurazione di hapoxy si divide in :
- frontend: sono dove ascolta il lb 
- backend: che servizi deve colpire  

Per controllare se il file di configurazione è giusto: `haproxy -c -f /etc/haproxy/haproxy.cfg`



Haproxy lavora a livello 6/7, è un daemon intelligente. ascolta inn attesa di traffico TCP sulla porta 8443 (in questo esempio specifico). è la componente configurata ad interrogare il DNS locale alla riceva del record SRV. haproxy legge quella lista, aggiorna la sua tabella di routing interna e smista i pacchi.  


### Installazione di Kubernetes  

- installazione di gpg
- dowload delle chiavi di firma 
- repository di k8s 
- apt update 
- apt install kubeadm kubectl kubelet
- apt-mark hold kubelet kubeadm kubectl
- systemctl enable --now kubelet.service 


Bootstrap kubernetes:  

Useremo kubeadm, che è solamente uno strumento di bootstrapping. Dobbiamo lanciarlo con quanti più parametri possibili per evitare brutte sorprese.   

Da fare solo sul primo nodo di control plane!  

kubeadm init 
    --apiversion-bind-port 6443
    --apiserver-cert-extra-sans {{k8s_api_loadbalancer_ip | 172.16.42.5 }}
    --control-plane-endpoint {{k8s_api_loadbalancer_dns | api-lb.k8scourse.serics.eu}}:{{k8s_api_loadbalancer_port | 8443}}
    --node-name {{fqdn | node1.k8scourse.serics.eu}}
    --upload-certs
    --pod-network-cidr 10.244.0.0/16
    --service-cidr 10.245.0.0/16
    --service-dns-domain cluster.local


I parametri in ordine:  
- `--apiversion-bind-port 6443`: configura il servizio kube-apiserver, apre un socket tcp in ascolto sulla porta 6443 e si aggangia all'IP locale della macchina host.  
- `--apiserver-cert-extra-sans 172.16.42.5 `: 
- `--control-plane-endpoint api-lb.k8scourse.serics.eu:8443`: kubeadm prenderà quell'URL e quella porta per scriverli nei file di configurazione del cluster. In questo modo, nessuna componente del cluster saprà mai dell'esistenza della porta 6443 o dell'ip fisico del node1 (ossia dove viene veramente servito l'api-server). Sapranno solo che per parlare con l'api-server devono collegarsi alla porta 8443 di api-lb.  
- `--node-name node1.k8scourse.serics.eu`: è l fqdn, l'dentificativo con cui i nodo si registrerà in etcd. quando il control plane dovrà mandare un comando a questo nodo, userà questo nome.  
- `--upload-certs`: serve per potere mandare certificazione (ed evitare di dover copiare fisicamente su ciavetta i file .crt e .key). In questo modo kubeadm prende le chiavi, le cripta con una psw temporanea e le carica in etcd. Quando lanceremo il comando join su un altro nodo, questo scaricherà le chiavi direttamente dalla memoria del cluster in modo sicuro.  
- `--pod-network-cidr 10.244.0.0/16`: questi sono gli IP che vengono assegnati ai singoli container(Pod), quando un Pod nasce il plugin di rete (la CNI) prende un IP da questa lista e lo assegna al Pod. K8s divide questo blocco e assegnerà una sottorete più piccola (/24) a ogni nodo fisico in modo che i nodi noon si rubino gli IP dei Pod 
- `--service-cidr 10.245.0.0/16`: K8s usa i Service ossia bilanciatori di carico interni al cluster che fungono da IP stabili. l'AI Server prende gli IP per questi bilanciatori da questo blocco 
- `--service-dns-domain cluster.local`: definisce la rete dentro il cluster.
  - K8s ha un suo server DNS interno in esecuzione come Pod (**coreDNS**) che serve per far parlare le applicazioni tra di loro.  
  - Se abbiamo un pod che fa da FE e uno che fa da DB, il FE non userà mai l'IP del DB perchè è in continuo cambiamento, usa il nome del suo *Service* 

### Installazione del PlugIn CNI:   

Installeremmo kube-router seguendo la [guida](https://github.com/cloudnativelabs/kube-router/blob/master/docs/kubeadm.md)  

```bash
KUBECONFIG=/etc/kubernetes/admin.conf kubectl apply -f https://raw.githubusercontent.com/cloudnativelabs/kube-router/master/daemonset/kubeadm-kuberouter.yaml  

kubectl -n kube-system rollout restart deployment/coredns  

kubectl -n kube-system scale deployment/coredns --replicas=3

# dovremo vedere:
root@node1:/home/user# kubectl get pod --all-namespaces
NAMESPACE     NAME                                                READY   STATUS    RESTARTS   AGE
kube-system   coredns-7c754bbf8d-5fw7r                            1/1     Running   0          2m9s
kube-system   coredns-7c754bbf8d-pfn55                            1/1     Running   0          3s
kube-system   coredns-7c754bbf8d-v4sl8                            1/1     Running   0          2m9s
kube-system   etcd-node1.k8scourse.serics.eu                      1/1     Running   0          5h35m
kube-system   kube-apiserver-node1.k8scourse.serics.eu            1/1     Running   0          5h35m
kube-system   kube-controller-manager-node1.k8scourse.serics.eu   1/1     Running   0          5h35m
kube-system   kube-proxy-hv7jc                                    1/1     Running   0          5h35m
kube-system   kube-router-w9c4l                                   1/1     Running   0          3m54s
kube-system   kube-scheduler-node1.k8scourse.serics.eu            1/1     Running   0          5h35m
```

Abbiamo installato una CNI quindi va riavviato per prendere l'indirizzo della CNI  
`kubectl -n kube-system rollout restart deployment/coredns`  

Possiamo aumentare il suo numero di repliche, si consiglia max(2, numero_nodi/50)   


### Join di un nodo.   

kubeadm join  
    {{}}:{{}}
    --control-plane
    --certificate-key {{}}
    --token {{}}
    --discovery-token-unsafe-skip-ca-verification
    --node-name {{}}

Nel nostro caso dobbiamo andare sul node1 e generare 
```bash
root@node1:/home/user# kubeadm token create --ttl 1h
29urrx.mj2vzofzg1f5u3if
root@node1:/home/user# kubeadm init phase upload-certs --upload-certs
I0326 16:30:12.527749    8956 version.go:261] remote version is much newer: v1.35.3; falling back to: stable-1.33
[upload-certs] Storing the certificates in Secret "kubeadm-certs" in the "kube-system" Namespace
[upload-certs] Using certificate key:
ff40811298fe9ce482442e9b2750fabf7c0222f41aea72aa5f1ce5ddcfd59f7b
```

Poi faremo la join del node2:
```bash
kubeadm join api-lb.k8scourse.serics.eu:8443 \
  --control-plane \
  --token 29urrx.mj2vzofzg1f5u3if \
  --certificate-key ff40811298fe9ce482442e9b2750fabf7c0222f41aea72aa5f1ce5ddcfd59f7b \
  --discovery-token-unsafe-skip-ca-verification \
  --node-name node2.k8scourse.serics.eu
```


### MetalLB  

è un software che permette di allocare IP virtuali annunciati via ARP o BGP (L2 mode)  

