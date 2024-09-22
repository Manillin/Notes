# Django CBV

Abbiamo visto l'intero design pattern di Django:
- Models (DB relazionale tramite ORM + migrations)
- Views (Funzioni che regolano la logica della webapp)
- Templates (per il lato presentazione)


## Policy DRY in Django

Django rimane fedele alla policy DRY (don't repeat yourself), sappiamo che non è sempre comodo sviluppare accesso al DB tramite FV (codice ripetuto).  
Per rispondere a una richiesta del client Django permette anche di usare le **Classi (Class Based View)** con le loro caratteristiche (minimizzazione riutilizzo di codice sfruttando relazioni di ereditarietà e polimorfismo).  

**In breve** $\rightarrow$ In alcune situazione usare una Class Based view è la cosa migliore, in altre conviene invece usare una Function View, sono quindi da intendersi come due meccanismi complementari e non esclusivi.  


### Metodi CRUD sul DB:

I metodi CRUD rappresentano le operazioni base con cui operare sui DB, vedremo come fare tali operazioni tramite CBV.  


## Django Views

Django mette a disposizione delle `View` sotto forma di classi $\rightarrow$ appartengono ai moduli `django.views.generic.*`  

Per sfruttarle quindi si usa una classe che estende dalle View di Django  
Esiste una View per ogni operazione CRUD:
- `ListView`: Elenca le entry delle tabelle
- `UpdateView`: Permette di modificare una entry di una tabella 
- `CreateView`: Permette di creare una entry in una tabella
- `DeleteView`: Rimuove una entry da una tabella
- `DetailView`: Permette di leggere gli attributi di una specifica entry di una tabella

Nella nostra classe CBV si specifica:
- Il **Model** al quale ci si riferisce 
- Il **Template** per renderizzare in HTML/DTL

Nota $\rightarrow$ Nel caso in cui si creino delle CBV risulta necessario specificarlo in urls.py, in quanto di default la funzione `path` si aspetta una funzione e noi stiamo passando una classe!  
Il metodo `as_view()` trasforma questa classe nell'equivalente di una funzione e permette il corretto display della nostra view, la 'funzione' quindi apparirà cosi: `views.NomeClassView.as_view()`


Es: [ProgettoCBV - guardare su git per maggiori dettagli]

```python
#In iscrizione/views.py

from django.views.generic.list import ListView
from .models import *

class ListaStudentiView(ListView):
    model = Studente
    template = 'iscrizioni/lista_studenti.html'
```

Per rendere accessibile la view e far si che risponda alle richieste del client: 

```python
#In root/urls.py
urlpatterns = [
    path('admin/', admin.site.urls),
    path('iscrizioni/', include('iscrizioni.urls')),
]
```

```python
#In iscrizioni/urls.py

from django.urls import path
from . import views

app_name = 'iscrizioni'

urlpatterns = [
    path('listastudenti/', views.ListaStudentiView.as_view(), name='listastudenti')
]
```

### Variabile di contesto: object_list

E il template sarà così strutturato:

```python
#In iscrizioni/templates/iscrizioni

{% extends 'base.html' %}
{% block title %} Dump DB {% endblock %}
{% block content %}

{% for p in object_list %} # object_list (!!!!!)
<ul>
    {{ p }}
</ul>
{% endfor %}

{% endblock %}
```

Poniamo la nostra attenzione sulla variabile di contesto (ottenuta 'gratis') che si chiama **`object_list`**:  
object_list è una variabile di contesto che viene automaticamente fornita da Django quando si utilizza una classe basata su una delle generiche viste, come `ListView` (il nostro caso specifico in questo esempio), questa variabile contiene un `QuerySet` che rappresenta l'insieme di oggetti del modello specificato nella definizione della classe CBV.  
**In breve** $\rightarrow$ Django esegue una Query sul modello specificato e la mette nella variabile di contesto object_list, questa variabile è accessibile direttamente da template 


## QuerySet:

Un QuerySet in Django è una collezione di oggetti provenienti dal database. È il modo in cui Django rappresenta una query al database. Un QuerySet può essere filtrato, ordinato e manipolato in vari modi per ottenere i dati desiderati.

<br><br>


# Metodi fondamentali delle CBV:

Queti metodi sono ereditati da `View`, il capostipite di tutte le view preconfezionate di Django.  
Consistono in metodi che django utilizza di default, e che quindi possono essere sovrascritti per essere modellati secodo le nostre necessità.  

- `get_queyset` $\rightarrow$ permette di fare operazioni sul queryset indicato dall'attributo model definito dentro la classe.    
- `get_context_data` $\rightarrow$ permette di aggiungere varaibili di contesto utili al template.  


Es: [Elenca studenti iscritti almeno ad un corso]

```python
class ListaInsegnamentiAttivi(ListView):
    model = Insegnamento 
    template_name = 'iscrizioni/insegnamenti_attivi.html'

    #uso dei metodi delle CBV 

    def get_queryset(self):
        return self.model.objects.exclude(studenti__isnull=True)

    def get_context_data(self,**kwargs):
        context = super().get_context_data(**kwargs)
        context['titolo'] = 'Insegnamenti Attivi'
        return context
```


### Contesto Ereditato:

Notiamo che in get_context_data() viene chiamata in primis il metodo dalla classe padre per ottenere il contesto di default, e una volta ottenuto lo modifichiamo inserendo i dati che vogliamo, salvandolo in una variabile locale.  

```python
context = super().get_context_data(**kwargs)
print(context.keys())

#output:
dict_keys(['paginator', 'page_obj', 'is_paginated', 'object_list', 'insegnamento_list', 'view'])
```

Notiamo che nel contesto ereditato abbiamo una key **`view`** che ci permette di acceder a metodi e attributi della nostra classe deerivata da ListView $\rightarrow$ Ci consente di definire ulteriori logiche abitrarie e di chiamarle direttamente dal template.  


Es: [Fare una CVB che elenca studenti e restituisca il numero di iscrizioni totali]  

```python
class ListaStudentiIscritti(ListView):
    model = Studente 
    template_name = 'iscrizioni/studenti_iscritti.html'

    def get_model_name(self):
        return self.model._meta.verbose_name_plural

    def get_test():
        return "Test!"

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        ctx['titolo'] = 'Lista Studenti Iscritti:'
        return ctx
    
    def get_totale_iscrizioni(self):
        count = 0
        for i in Insegnamento.objects.all():
            count += i.studenti.all().count()
        return count
```

Sotto riportato il _template:_  

![view variable in template](../images/view_django_variable.png)
  
Notiamo come nel template grazie alla variabile `view` possiamo accedere direttamente a metodi definiti nella classe trasparentemente (ricordare che serve avere il contesto e quindi get_context_data() è fondamentale!).  



