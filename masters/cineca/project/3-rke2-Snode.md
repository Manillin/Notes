### Set up di RKE2 a singolo nodo master  


Infrastruttura creata con le API OpenStack di Terraform  

- subnet privata 
- due VM flavour: xs
  - una VM con floatingIP che funge da bastion 
  - una VM che fa da nodo k8s nella rete privata 
- un router e un floating IP
- un secgroup comune e 4 sec rules 
- ansible per l'installazione di Kubernetes RKE2 con un ruolo creato ad hoc


Per prima cosa si fa il deploy dell'infrastruttura con terraform (terraform plan + terraform apply)  

La struttura della cartella di lavoro sarà:  

```
terraform_ansible/
├── ansible
│   ├── ansible.cfg
│   ├── inventory.ini
│   ├── playbook.yml
│   └── roles
│       └── rke2_master
│           ├── tasks
│           │   └── main.yml
│           └── templates
│               └── config.yaml.j2
└── terraform
    ├── ada_ssl_2026.pem
    ├── bro.md
    ├── clouds.yaml
    ├── cluster.auto.tfvars
    ├── inventory.tpl
    ├── main.tf
    ├── modules
    │   ├── compute
    │   │   ├── main.tf
    │   │   ├── outputs.tf
    │   │   └── versions.tf
    │   ├── loadbalancer
    │   │   ├── main.tf
    │   │   ├── outputs.tf
    │   │   └── versions.tf
    │   └── network
    │       ├── main.tf
    │       ├── outputs.tf
    │       └── versions.tf
    ├── outputs.tf
    ├── terraform.tfstate
    ├── terraform.tfstate.backup
    ├── variables.tf
    └── versions.tf
```

Rispetto al deploy del bastion + private VM dobbiamo:  

- Usare il bastion come loadbalancer lv4 (non più lv7) 
- Aggiungere sec group per aprire l'ingress sulla porta 6443 di entrambe le VM  
- Installare RKE2 con ansible sulla VM privata 


### Aggiunta della security rule:

```hcl
resource "openstack_networking_secgroup_rule_v2" "test_secgroup_rule4"{
    direction = "ingress"
    ethertype = "IPv4"
    protocol = "tcp"
    port_range_min = 6443
    port_range_max = 6443
    remote_ip_prefix = "0.0.0.0/0"
    security_group_id = openstack_networking_secgroup_v2.test_secgroup.id 
}
```

Applichiamo i cambiamenti con terraform apply.   


### Organizziamo la cartella ansible per definire il ruolo e template jinja2  

Creiamo `ansible/roles` e dentro tale cartella creiamo:  

- `tasks/main.yml`: il ruolo per installare RKE2 nella VM privata 
- `templates/config.yaml.j2`: file di configurazione con templating jinja2 per fare configurazione dinamica  


In `tasks/main.yml` installeremo rke2:  


```ansible
---
- name: Forza API a usare IPv4 
  copy: 
    dest: /etc/apt/apt.conf.d/99force-ipv4
    content: 'Acquire::ForceIPv4 "True";'
    mode: '0644' 

- name: Aggiorna cache di APT 
  apt:
    update_cache: true 
    cache_valid_time: 3600 

- name: Assicurare curl 
  apt: 
    name: curl 
    state: present 

- name: Crea la directory di configurazione per RKE2
  file:
    path: /etc/rancher/rke2
    state: directory
    mode: '0755'

# USIAMO TEMPLATE JINJA2 
- name: Genera il file config.yaml da Jinja2
  template:
    src: config.yaml.j2
    dest: /etc/rancher/rke2/config.yaml

- name: Scarica ed esegui lo script di installazione di RKE2
  shell: curl -sfL https://get.rke2.io | sh -
  args:
    creates: /usr/local/bin/rke2  # Evita di riscaricarlo se esiste già

- name: Abilita e avvia il servizio rke2-server
  service:
    name: rke2-server
    enabled: yes
    state: started

- name: Scaricare il file kubeconfig dal master al PC locale 
  fetch:
    src: /etc/rancher/rke2/rke2.yaml 
    dest: ~/.kube/config_rke2
    flat: true 

- name: Modifica IP nel kubeconfig locale per puntare al bastion 
  delegate_to: localhost 
  become: false 
  replace:
    path: ~/.kube/config_rke2 
    regexp: '127\.0\.0\.1' 
    replace: "{{ groups['proxy'][0] }}"
```

### Note:   

Genereremo il file `config.yaml` dal template jinja2 in templates/config.yaml.j2 (usando espressamente 'yaml' in quanto rke2 si aspetta quel suffisso)   

dentro tale file avremo:

```jinja2
# /etc/rancher/rke2/config.yaml
write-kubeconfig-mode: "0644"
tls-san:
  - "{{ groups['proxy'][0] }}"
```

Scaricheremo il file kubeconfig dalla VM privata al pc host locale per poter controllare il cluster da remoto.  
Useremo quindi la `fetch` di ansible e sceglieremo come destinazione nel pc locale `~/.kube/config_rke2`    

Modifichiamo il file kubeconfig che abbiamo appena scaricato localmente per fare in modo che punti al bastion (il singolo entry point che fa lb!)  
Per farlo usiamo la `replace` con `regexp` in particolare avremo: 
- regexp: '127\.0\.0\.1' -> matchiamo questo IP
- replace: "{{ groups['proxy'][0] }} -> usiamo il file inventory.ini per prendere il FIP del bastion 


### Impostiamo il bastion come lb lv4   

Nel playbook principale, ossia dove scarichiamo ciò che ci serve sulle due VM, avremo le seguenti istruzioni ansible per imposstare il bastion come lb lv4.  

```ansible
# altre istruzioni... (forzare IPv4, APT update, ...)
- name: Configura Nginx come TCP proxy per k8s API
      blockinfile: 
        path: /etc/nginx/nginx.conf 
        insertafter: EOF 
        block: | 
          stream {
            upstream k8s_api{
              server {{groups['masters'][0]}}:6443;
            }
            server {
              listen 6443;
              proxy_pass k8s_api;
            }
          }
      notify: Riavvio Nginx 
  
  handlers:
    - name: Riavvio Nginx 
      service: 
        name: nginx 
        state: restarted  
```


Siccome nostro file inventory.ini sarà stato generato dinamicamente da terraform non dobbiamo preoccuparci di mettere gli IP giusti e useremo le variabili di ambiente di ansible che leggono direttamente da tale file.   

A questo punto potremo runnare il playbook.  

Una volta fatto ciò, avremo il cluster K8s RKE2 pronto e per comandarlo dall'esterno basterà fare:  
- `export KUBECONFIG=~/.kube/config_rke2 `  

dal terminale del pc host locale faremo `kubectl get nodes` e vedremo qualcosa del genere:  

```
cvonwaldorff@NCVONWALD204771:~$ kubectl get node
NAME            STATUS   ROLES                AGE   VERSION
test-worker-1   Ready    control-plane,etcd   76m   v1.34.6+rke2r3
``` 
