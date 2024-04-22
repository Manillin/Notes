# 22 Aprile

## Diagramma delle classi!

Due concetti importantissimi :
Aggregazione e composizione (sono due cose diverse), vengono direttamente dalla fase di analisi.

Importanza del lifecycle in informatica -> per gestire la memoria e le allocazioni/deallocazioni -> conseguenza macro sulle performance. [es c++]

Si usano piu interfacce per -> Separation of concerns (separazione di concetti)

Dependency injection (vedere filosofia SOLID), le interfacce sono come dei "contratti".

## Template Classes

importanza dei generics e template.

In uml usare Stereotype [<<>>] e Notations (check for docs) per specificare funzionalità e migliorare la readibility.

**Interfacce**: concetto più importante della OOP, ricordano i 'contratti'
quando uso qualcosa mi interessa solo il contratto che ho con tale oggetto! non devo sapere come è implementato !

_interface segregation_
Quado si usa un oggetto lo si usa attraverso un interfaccia, possibilmente la piu alta nella tassonomia $\rigihtarrow$ classificazione gerarchica di elementi .

## PACKAGE - Namespace

Raggruppatori di classi in base a macrofunzioni (salvabile, core, testing, ecc...)

_Fare sempre l'esercizio di DIVIDE ET IMPERA_ -> fondamentale sempre in ogni progetto, se omesso crea code debt.

(c# extensione methods)

## SOLID

consiste in 5 best practices pensate per creare codice pronto ad essere scalabile.

-> come strutturare il codice afficihe sia piu scalabile

Applicabile agli OOP e sono:

1. Single Responsability
2. Open/Close principle
3. Liskov substitution
4. Interface Segregarion
5. Dependency Inversion

Ogni classe dovrebbe avere una sola entità per cambiare -> ogni classe dovrebbe implementare una SOLA funzionalità (diverso da dire "avere un solo metodo") -> esempio del contatore: deve incrementare e 'sapere quante istanze ci sono' -> code smell.
Fa aumentare il numero di classi in un programma.

Aperte per estensioni Chiuse per modifiche -> si usa la visibilità protected per raggiungere questo vincolo

devi essere in grado di lavorare su ogni oggetto della tassonomia delle mie classe -> serve a identificare i code smell.

Many clinet specific interfaces are better than a big one.
An interfaces is a contract! -> conviene averne "il piu possibile" -> risponde a when should i create an interface? as much as you can!
Quanto testo una classe TESTO UNA SUA INTERFACCIA , in quanto testo una funzionalità precisa.
--> LIMITARE dipendenze di libraria: es, classe importa MAX 1 libreria

Il progetto intero non deve dipendere da nulla! le dipendenze devono essere wrappate in classi.
Non dipendere ma essere la radice della dipendenza
