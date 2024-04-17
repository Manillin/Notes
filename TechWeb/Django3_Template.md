# DTL e Risorse Statiche:

Fino ad ora abbiamo visto URL routing con funzione di risposta, che restituisce un `HttpResponse` object sotto forma di stringa che causa la creazione di HTML minimale.
I siti però seguono regole di formattazione HTML e il contenuto è formattato tramite tag.

Risorse utili per HTML:

- https://www.w3schools.com/html/default.asp
- https://developer.mozilla.org/en-US/docs/Web/HTML/Element?retiredLocale=it

Django permette di inserire direttamete stringhe formattate secondo i canoni di `HTML` ma questo presenta vari problemi:

- Non comodo, Genera codice ridondante e ripetuto (viola policy **DRY**)
- Non facilmente riutilizzabile e non estendibile

La soluzione a questi inconveniente sono i **_Template_**

# Template

> I Template sono 'scheletri' riutilizzabili nella parte di presentazione della nostra webapp.
> I Template separano la logica di presentazione dalla logica di business.
> Si basano sul presupposto che diverse pagine si 'somiglino' e quindi codice simile deve essere riutilizzabile.
> Deve dare la possibilità di aggiungere una logica al mero HTML!

Ricordiamo che Django segue la logica **Model-Template-View** e concludiamo quindi dicendo che i **template** sono i responsabili della presentazione dell'interfaccia utente.
In breve sono file HTML che possono includere sintassi specifica di Django nel linguaggio specifico **DTL** (che aggiunge dinamicità alle pagine)

## DTL: Django Template Library

Il DTL è considerato un vero e proprio linguaggio, che si basa su **blocchi** anziche su tag.
Un file DTL appare come file sorgente statico che viene dinamicamente risolto lato server per essere spedito sotto forma di _risultato statico_ all'utente tramite il browser.

## Tipo di linguaggio:

Il DTL è un mix tra linguaggio di markup (html) e logica di programmazione, consente di inserire logica dinamica all'interno dei template.
Oltre ai blocchi che delimitano 'spazi notevoli' tipici dei linguaggi di presentazione abbiamo anche dei _programming construct_ che ci permettono l'accesso in **lettura** e **scrittura** alle variabili di **_contesto_**.
Ci permette molto comodamente di seprare le istruzioni condizionali da quelle di markup.

### Variabili di contesto: dive in

Consistono in dati (variabili, oggetti, strutture dati, query, ...) forniti alla `view` di Django per essere passati al template per essere visualizzati.
Quando un template viene **_Renderizzato_** può accedere a queste variabili (passate tramite un dizionario mappato 'str': 'ctx_var') per visualizzarle dinamicamente all'interno del HTML.

## Uso dei Template e DTL nel progetto:

Per poter sfruttare questo linguaggio servirà aggiungere nella root directory del progetto la cartella `templates` e configurare le impostazioni di Django da `settings.py` in modo che risultino visibili i cambiamenti

```python
TEMPLATES = [
    {
        'BACKEND': '',
        'DIRS': [os.path.join(BASE_DIR), "templates"],
        #specifica la dir che contiene i nostri 'templates'
        ...
    },
]
```

Possiamo considerare i template come oggetti in python, e in quanto oggetti dispongono della logica legata ai principi di **OOP**.
Infatti diremo che un `template` può:

**_Nota:_** Su Vscode se si edita un file DTL è probabile che intervenga un autoformattatore, se succede il file risulterà errato, **assicurarsi** di salvare i file DTL con: `CMD+k + S` per salvare senza formattare il file.

- **Estendere** un template base
- **Includere** un template base  
  note: guardare esempio p.17
