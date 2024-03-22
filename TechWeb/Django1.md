# Django

## Cos'è:

Django è un **Web Framework** open source basato su Python che utlizza il pattern architetturale **model-template-view**

## pipenv:

Pipenv ci permette di creare ambienti virutali per mantenere separate le dipendenze tra diversi progetti (e dal sistema) mantenendo i loro ambienti di sviluppo isolati (importazione di librerie).
Creazione di un ambiente:

```bash
mkdir folder_progetto
cd folder_progetto
pipenv --version
pipenv install django
pipenv shell #entriamo nel progetto
```

a questo punto vedremo una cosa del genere:

```bash
(django-N5R2c0ff) chris@host:~/Desktop/TechWeb/django (main)$
```

## Startproject:

il comando per gestire le attività amministrative è: `django-admin` e se seguito da `startproject` e come argument `<nome_progetto>` creerà un nuovo progetto django.
La creazione di un progetto Django crea una gerarchia di directory e file.
I file che verranno creati saranno i seguenti:

```python
main_folder/
    nome_progetto/
        __init__.py
        asgi.py
        settings.py
        urls.py
        wsgi.py
    db.sqlite
    manage.py
    views.py
Pipfile
Pipfile.lock
```

e per lanciaer il webserver locale useremo:

```bash
python manage.py runserver
```

Dopo aver lanciato il comando avremo in http://127.0.0.1:8000 il nostro server web

## File Importanti:

I file creati in automatico da Django ci permettono di modellare il nostro webservice e applicazione, in particolare:

### `views.py`

È un file costituito da 'viste', in questo file mettiamo la nostra **Business Logic** e il nostro codice di **BackEnd**, ossia la logica e la risposta del server in seguito a una particolare richiesta.

```python
from django.http import HttpResponse
def home(request):
    return HttpResponse("Greetings!")
```

### `urls.py`

File destinato a contenere una lista `urlpatterns` nella quale ciascun elemento rappresenta un url _collegato a una view_ specifica e ad un nome (o alias) che può essere usato nei template.

```python
from django.contrib import admin
from django.urls import path
from . import views # file sopra menzionato con la business logic

urlpatterns = [
    path('', views.home, name='home'),
    path('admin/', admin.site.urls, name='admin')
    ...
]
```

nell'esempio `path('', views.home, name='home')` avremo che verrà eseguita la funzione `home` dal file `views.py` e il nome/alias assegnato sarà: 'home'
