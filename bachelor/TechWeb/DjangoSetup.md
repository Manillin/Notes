# SetUp Progetto Django 


Per iniziare un progetto su Django si usano i seguenti comandi:

### Comandi Shell:

```bash
mkdir ProjectFolder
cd ProjectFolder

pipenv install django 
pipenv shell

django-admin startproject nomeprogetto
cd nomeprogetto

python manage.py runserver #per lanciare il server web

# se si vuole creare un app per gestire particolare necessita:
python manage.py startapp nomeapp 

```

### Comandi dentro il Workspace:

Una volta creato il workspace con i realtivi folder ricordarsi di apportare le seguenti modifiche:


1. Creare i folder dei template in `nomeprogetto/templates` e se abbiamo creato un app anche in `nomeapp/templates/nomeapp`  
2. In `settings.py` $\rightarrow$ aggiungere in `INSTALLED_APPS` il nome dell'app e in `TEMPLATES` aggiungere `DIRS:[os.path.join(BASE_DIR), "templates"]`.  
3. Creare i modelli e applicare le migrazioni con i seguenti due comandi:
    - `python manage.py makemigrations nomeapp`
    - `python manage.py migrate`
4. Se si vuole inizializzare il DB creare uno script `nomeprogetto/initcmd.py` e creare `init_db()` e `erase_db()`.  
5. Creare un **Superutente**:  
    - `python manage.py createsuperuser` e seguire le istruzioni da terminale  
    - In nomeapp/admin.py registrare i modelli `admin.site.register(Modello)`

