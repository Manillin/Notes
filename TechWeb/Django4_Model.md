# Models

Il _model_ è lo strumento utilizzato da Django per strutturare i idati sul _database_. Solitamente ogni model viene mappato su una singola tabella del database.

- Ogni model è una classe Python che eredita da `django.db.models.Model`
- Ogni attributo della classe Model rappresenta un campo del database
- Django genera automaticamente le API di accesso al DB

```python
from django.db import models

class Persona(models.Model):
    nome = models.CharField(max_length=50)
    cognome = models.CharField(max_length=50)
```

Il modello _persona_ creerà una tabella sul database con il seguente codice:

```SQL
CREATE TABLE soci_persona (
     “id” serial NOT NULL PRIMARY KEY,
     “nome” varchar(50) NOT NULL,
     “cognome” varchar(50) NOT NULL
);
```

_note:_ Il nome della tabella viene generato automaticamente ma può essere sovrascritto se si vuole; un campo ID viene aggiunto con la funzione di primary key (sovrascrivibile anche questo).

## Migrazioni:

Le migrazioni sono il modo in cui Django estende le modifiche fatte sui model direttamente al database, I comandi per le migrazioni sono due e funzionano in sequenza, rispettivamente:

1. `makemigrations`
   Crea nuove migrazioni basate sui cambiamenti rilevati nei model, eseguendo il comando viene generato un file nel quale si elencano le creazioni di nuovi model o le modifiche a model già esistenti.

2. `migrate`
   È il comando che applica le migrazioni. `migrate` esegue i comandi presenti nel file che genera `makemigrations` e applica le modifiche al database.

## Relazioni tra tabelle:

Tramite Django riusciamo ad esprimere comunque le relazioni tra tabelle, in particolare abbiamo i campi:

- ForeignKey:
  Rappresenta una relazione $ n \rightarrow 1$ e richiede due argomenti posizionali :
  - la classe Model al quale si riferisce
  - Il parametro `on_delete` per dettare il comportamento della cancellazione.
  - Il parametro `related_name` permette di definire un nome per l'attributo inverso della relazione.
- ManyToManyField:
  Rappresenta una relazione many to many $n \rightarrow n$ e richiede un argument posizionale: La classe con la quale si relaziona -> Django crea una tabella intermedia per rappresentare la relazione, effettua una _reificazione_.

- OneToOne:
  Rappresenta una relazione one to one $1 \rightarrow 1$, simile al concetto di FK con il parametro `Unique=True`. Viene principalmente usato come PK di un model che estende un altro model.
