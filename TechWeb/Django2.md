# Django e Primi test

### Ricordiamo gli step per creare un nuovo progetto in Django:

1. Impostazione di un ambiente virtuale con `pipenv`
2. Assicurarsi di essere nell'ambiente viruale `pipenv shell`
3. Installazione nell'ambiente virtuale delle librerie necessarie
4. Lanciare il comando: `django-admin startproject <nomeprogetto>`

## Mini erercizio:

> Dare un messaggio di benvenuto ai visitatori in uno specifico URL:
>
> Steps:

- Andare su views.py e implementare una funzione che prenda in ingresso una `request` e restituisca un oggetto `HttpResponse`
- Inserire in urls.py il corrispettivo url che punta alla funzione cosi definita
- Provare il corretto funzionamento andando su tale URL

```python
# views.py
from django.http import HttpResponse

def home_page(request):
    response = "Hello! Welcome to my webpage"
    return HttpResponse(response)
```

```python
# urls.py
from django.contrib import admin
from django.urls import path
from .views import home_page

urlpatterns = [
    path('admin/', admin.site.urls, name='admin'),
    path('home/', home_page, name='home')

]
```

Andando ora in http://127.0.0.1:8000/home/ vedremo il nostro messaggio!
Se però torniamo ora su http://127.0.0.1:8000 vedremo un messaggio di errore, scopriremo grazie alla modalità **Debug** che il webserver ritorna status code `404`.
Scopriamo cosi che se si prova ad accedere a una risorsa non specificata il nostro programma ritorna errore.

## Uso di regex per definire URL

Possiamo usare **Regex** per facilitare il reindirizzamento in URL, ma bisogna stare attenti in quanto le Regex sono estremamente potenti e potrebbero fare match indesiderati se non definite correttamente.
Risorsa: https://regex101.com .
Per creare URLS espandibili con regex importiamo `re_path`

```python
...
from django.urls import path, re_path
...
urlpatterns = [
    re_path(r"^$|^/$|^home/$", home_page, name="homepage")
]
```

## Com'è fatto l'oggetto `request`

> Scopriamo che request è: `<WSGIRequest: GET '/urlpath/'>`

Le richieste `GET` ammettono parametri specificati tramite URL:
`url_path?param1=1&param2=2&param3=3&...`
Esempio:

```python
# urls.py:
path("elencaparametri/", elenca_parametri, name="parametri")

# la funzione elenca_parametri è specificata in:
# views.py
def elenca_parametri(request):
    response=""
    for k in request.GET:
        response+=request.GET[k]+""

    return HttpResponse(response)

```

### note su `request`:

- È un oggetto della classe `<class 'django.core.handlers.wsgi.WSGIRequest'>`
- Ha un dizionario accessibile con il metodo `.GET` che contiene i parametri che riceve (passati tramite URL)

Si possono passare parametri tramite URL path e fare operazione di **type enforcement**:

- `path('url_path/<int:eta>/', view_func, name='alias')` Prende un solo parametro di nome `eta` e di tipo intero.
- `path('url_path/<str:nome>/<int:eta>, welcome_path, name='alias')` Prende due parametri uno stringa e l'altro intero.

Usando il _type enforcement_ diventa possibile accedere direttamente ai parametri come se fossero argomenti della funzione in views.py:

```python
def welcome_path(request, nome, eta):
    [...]
```

Quando si usano i parametri passati tramite URL la chiave del dizionario GET che li contiene è il nome del parametro:

In questo codice si accede al dizionario con kw `num`.

```python
def pari_dispari(request):
    if 'num' not in request.GET:
        return HttpResponse("Non hai inserito un numero")
    numero = int(request.GET['num'])
    if numero % 2 == 0:
        return HttpResponse(f"Il numero {request.GET['num']} è PARI")
    else:
        return HttpResponse(f"Il numero {request.GET['num']} è DISPARI")
```

Un url valido avrà questa forma: `http://127.0.0.1:8000/paridispari?num=5`.
Un url invalido, dove diamo un nome diverso alla variabile, non chiamerà correttamente la view che gli corrisponde: `http://127.0.0.1:8000/paridispari?parametro=5` $\rightarrow$ il mio valore intero sarà contenuto in request.GET['parametro'], io però dalla view sto accedendo al numero tramite `num`.
