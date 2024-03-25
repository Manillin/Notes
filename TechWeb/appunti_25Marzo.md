# Template Engine in Django

Per poter interpretate un file che incapsuli html e logica serve un engine in grado di interpretarlo dinamicamente.
Quindi un mix di linguaggio di markup e linguaggio di programmazione.
Ogni pagina può avere un template diverso

Esistono diversi tipi di template, e possiamo da programmatori scegliere quello che preferiamo.
Il rendering è quella fase dove viene interpretato tale file per 'sputare' la pagina risultante.

## Template Engine

I template engine vengono confiurati in `settings.TEMPLATES`, che è una lista di configurazione:

```python
templates = [
    'Backend': 'django.template.backends.django.DjangoTemplates'
    'Dirs': [] # Una lista di directory dove l’engine deve cercare i file di origine dei template, definiti in ordine di ricerca.
    'App_dirs': True # Definsce se l'engine deve cercare template dentro app installate
    'Options': {
        }
]
```

Si possono overridere funzioni per definire la logica che usa Django per andare a cercare i vari template.
E nel caso venga dato un template inesistente si è in grado di catturare errori logici e di sintassi.

# Template language

```python
variable -> {{variable}}
{{variable.attribute}}
```

Le veriabili vengono sostituite con valori quando il template viene elaborato
variabile = qualcosa che è nata nella business logic di python ma la sto passando ai template.
Possiamo anche accedere ai suoi attributi con la notazione puntata.

Quando il template engine incontra una variabile con punto, prova le seguenti ricerche in questo ordine:

- cerca un dizionario
- cerca un attributo o un metodo
- cerca un indice numerico

se il valore risultante è `callable` allora viene invocato senza parametri

# Grammatica di Base

## Filtri

Consistono in operazioni di preprocessing/filtri, una specie di decoratori in DTL.
Possiamo noi come programmatori definire filtri a nostro piacere.

## Costrutti
Per definire un costrutto di programmazione si usa una sintassi leggermente diversa rispetto a quella per la lettura delle varuabili.
Richiedono tag di apertura e di chousura es: `{% tag %}`
Si possono fare controlli di flusso e loop c
es:

```python
{% for element in elements_list %}
    {{element.detail}} <br>
{% endfor %}
```

Ci dovrebbe essere una netta separazione tra i 3 moduli : model view e template.

Per questo non si dovrebbe 'esagerare' con la capacità logica del DTL.
In particolare esagerare con la logica in DTL pone un rischio per la sicurezza.
