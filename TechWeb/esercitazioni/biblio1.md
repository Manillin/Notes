# Esercitazione 1: Biblio

Creazione di una webapp per la gestione di libri per una biblioteca.
Git del progetto: https://git.hipert.unimore.it/ncapodieci/techweb/-/tree/main/django/biblio/biblio?ref_type=heads

## Richiami di teoria:

Ricordiamo che il concetto di `Model` in Django è fondamentale del design pattern `MVT` $\rightarrow$ Model, View, Template

**_Filosofia ORM: [Object Relational Mapping]_**
Django usa questa tecnologia per la gestione di dati strutturati, tramite librerie interne traduce oggetti in entità fruibili come dati in un database.
Il processo di traduzione tra codice Python e comandi inerenti al DB sottostante si chiama **migrazione**.

### Vantaggi:

- Possibilità di creare e interagire con DB senza dover scrivere query
- Possibilità di personalizzare il comportamento delle classi modello con l'aggiunta di metodi `meta`
- Funzioni comunemente usate nei DB $\rightarrow$ import/export/CRUD+L etc... sono implementate gratis tramite il **Django View System**

# Creazione del progetto $\rightarrow$ step by step

Si crea un nuovo progetto, chiamato `biblio`, e la sua relativa app di gestione `gestione` (gestisce DB di libri).

```bash
pipenv install django
pipenv shell
django-admin startproject biblio
python manage.py startapp gestione
```

Una volta avviato il progetto si creano le seguenti cartelle:

1. `templates` nella root directory e `gestione/templates/gestione`.
2. Mettiamo il base.html in `biblio/templates`.
3. Ci assicuriamo che i templates siano raggiungibili in biblio/settings.py

```python
TEMPLATES = [
    {
        'BACKEND': '....',
        'DIRS':[os.path.join(BASE_DIR, "templates")],
        ...
    }
]
```

4. Installiamo l'app `gestione` in biblio/gestione
5. Creiamo `gestione/urls.py` e aggiungiamo l'app name

```python
# In gestione/urls.py:
app_name = "gestione"
urlpatterns = [
    path("listalibri/", lista_libri, name="listalibri")
]

# In biblio/urls.py aggiungiamo:
from django.urls import path, include
urlpatterns = [
    path(),
    path('gestione/',include('gestione.urls'))
]
```

# Creazione del DataBase:

Partiamo con una cosa semplice, una singola tabella Libro che è caratterizzato da: Titolo, Autore, Pagine, e da una data dell’ultimo prestito.
Sappiamo che è l'app models a gestre il nostro DB quindi ci spostiamo lì:

```python
# In gestione/models
from django.db import models

class Libro(models.Model):
    titolo = models.CharField(max_length=200)
    autore = models.CharField(max_length=50)
    pagine = models.IntegerField(default=100)
    data_prestito = models.DateField(default=None)

    def __str__(self):
        out = self.titolo + " di " + self.autore
        if self.data_prestito == None:
            out += " attualmente non in prestito."
        else:
            out += " in prestito dal " + self.str(data_prestito)

        return out

```

## Preparare e rendere effettiva la migrazione App level:

```bash
# Preparare la migrazione (output in file 0001_initial.py)
python manage.py makemigrations gestione
# Output:
Migrations for 'gestione':
   gestione\migrations\0001_initial.py
     - Create model Libro

# Rendere effettiva la migrazione:
python manage.py migrate
```

## Aggiungere elementi nel Database:

Si possono aggiungere elementi al database Programmaticamente | da console Admin | tramite Views e function views.

### Programmaticamente:

Nel nostro caso vogliamo inizializzare il DB e aggiungere elementi manualmente tramite una funzione apposita:

Creiamo un nuovo file `biblio/initcmds.py` e scriviamo le seguenti funzioni:

```python
# In biblio/urls.py

def erase_db():
    ....


def init_db():
    if len(Libro.objects.all() != 0):
        return # controllo se DB è gia inizializzato o meno

    l = Libro()
    ... #istruzioni per creare libro
    l.save()

# ----------------------------------------------------------

# In biblio/urls.py:

# ----------------------------------------------------------

from .initcmds import *

urlpatterns = [
    ...
]

erase_db() # DB wiped
init_db() # DB inizializzato

```

### Gestire DB da console Admin

Per poter sfruttare le funzionalità da Admin bisogna creare un superutente della webapp:

```bash
python manage.py createsuperuser
```

Una volta creato andiamo in `gestione/admin.py` e scriviamo questo:

```python
from .models import Libro
admin.site.register(Libro)
```

Una volta fatto ciò potremmo andare in: http://127.0.0.1:8000/admin/ e aggiungere | eliminare Libri liberamente dal DB!

### Accesso al DataBase da parte dell'utente:

Per ora abbiamo visto come rispondere a richieste `GET` HTTP che si attivano nel momento in cui un utente cerca risorse specifiche all'interno _dell'albero di risorse url_ dinamicamente generato da django.
Abbiamo visto anche come interagire con il DB usando codice Python, è quindi tempo di unire le due cose!

Creaiamo quindi una view e un template in grado di mostrare il contenuto del DB:

**gestione/views.py**

```python
from django.http import HttpResponse
from django.shortcuts import render
from .models import Libro

def listaLibri(request):
    templ = "gestione/listalibri.html"
    ctx = {
        "title": "Lista Libri",
        "listsalibri": Libro.objects.all() #restituisce una lista
    }
    return render(request,template_name=templ, context=ctx)
```

**gestione/template/gestione/listalibri.html**

```html
{%extends "base.html"%} {%block head%} {%endblock%} {%block title%} {{title}}
{%endblock%} {%block content%}
<center>
  <h1>{{title}}</h1>
  {% if listalibri.count > 0 %}
  <p>Ci sono {{listalibri.count}} libri in questa biblioteca</p>
  <ul>
    {% for l in listalibri %}
    <li>{{ l }}</li>
    {% endfor %}
  </ul>
  ...
</center>
{% endblock %}
```

Per rendere effettiva questa view andiamo in `gestione/urls.py`:

```python
from ...
from .views import *
app_name = "gestione"

urlpatterns = [
    path("listalibri/", listalibri, name="listalibri")
]
```

**ATTENZIONE:** Notiamo che la view e il template che abbiamo creato gestisce e interagisce con i `model`, nel nostro caso con il model `Libro`, che è stato ovviamente definito nell'app `gestione/`.
Come possiamo notare l'url per la view è anch'esso definito in `gestione/`, di conseguenza quando avviamo la nostra webapp se abbiamo fatto le cose bene, il path `/listalibri` definito in gestione/urls.py non sarà visibile dall'esterno, in quanto non è definito negli urls della root della webapp ossia in `biblio/`.
Per questo motivo precedentemente abbiamo incluso gli urls di `gestione/` dentro gli urls di `biblio/`, specificamente con l'istruzione:

```python
# biblio/urls.py
:
from django.urls import path, include
urlpatterns = [
    path(),
    path('gestione/',include('gestione.urls'))
    # rende visibili gli urls definiti in gestione/urls.py
]
```

In questo modo sarà possibile accedere agli urls della nostra app che gestisce i models e il DB, ma a **_una codizione_**, ossia che prima di cercare urls appartenenti a gestione/urls.py ci vada il prefisso `gestione`.

Es:
Una volta avviata la webapp notiamo:
[wrong url](../../images/404_wrong_url_gestione.png)
I path che cercati sono quelli definit ovviamente in `biblio/urls.py`
Se inseriamo nella root il path `gestione` rendiamo accessibili anche gli urls definiti nell'app ossia `gestione/urls.py`.
Con questo in mente andiamo in: http://127.0.0.1:8000/gestione/listalibri e vedremo la nostra view definita precedentemente con il template correttament renderizzato
[correct url](../../images/path_con_gestione.png)
