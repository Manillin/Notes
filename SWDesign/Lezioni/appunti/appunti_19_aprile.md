# Appunti 19 Aprile

Perche la OOP è importante ? dal punto di vistas concettuale e umano serve per rappresentare un 'oggetto' in un modo univoco e all'interno dell'oggetto possiamo avre il suo stato e i modi di accesso ad esso.
Possiamo visualizzare il mondo e i problemi al giusto livello di astrazione !

## Classi e relazioni

in OOP parliamo di dati -> rapprensetiamo la realta attraverso i dati!

diagrammi ER -> permettono di generare direttamente un DB (molto potenti per questo motivo).
ps: store procedure -> funzioni che vengono lanciate quando cambia qualcosa nel DB - side effects.

Z diagrams complicati da usare ma formalmente corretti e non ambigui.

Class diagrams:
grafo che definisce classi e interfacce [nodi] e le relazioni [archi], supporta quindi la OOP !
Permettono di modellare il comportamento statico di un sistema.  
Evidenziamo l'importanza di definire cosa succede staticamente a compile time e cosa succede dinamicamente a run - time -> perchè è cosi importante:

linguaggi statici [lavoro lasciato al compilatore, runs fasts ]vs linguaggi dinamici [interprete esamina instruction a run time, runs slower]

**Per modellare classi** bisogna usare il giusto livello di astrazione in base alla situazione da modellare. (filosofia di modellare il mondo e semplificarlo riducendolo alle sole sue qualità impotanti e inerenti in base alla situazione in cui ci troviamo).

OOP RECAP:
quando si creano le interfacce usare la `i` prima del nome dell'interfaccia.
qiando si creano le classi astratte usare la `a` prima del nome.

idea di tesi: tool grafico che dato uno snippet di codice astratto crea le classi in diversi linguaggi.

_static_ -> un campo statico ha tempo di vita pari al programma, nasce indipendentemente dal fattp che una classe sia instanziata o meno.  
È un campo che appartiene a **tutti** gli oggetti della classe, è un informazione comune a tutte le istanze della classe. metterle maiuscole in quanto pericolose e possono causare side effects. formalmente vanno 'sottolineate' negli schemi grafici.

Tipi di operazioni:

- chiamiamo **Query** xle operazioni che servono a recuperare lo stato dell'oggetto (lettura)
- chiamiamo **Modifiers** le op che modificano lo stato dell'oggetto (scrittura)

code smell: se ci sono piu di 4 o 5 parametri in una funzione --> attenzioneeee (un buon upperbound: 3) -> vuol dire che quella classe sta facendo troppo e ha troppe responsabilità, manutenzione diventerbbe difficilissimo.

### Relazioni tra classi

1. semplici
2. aggregazione (es. clinete nell'autonoleggio)
3. composizione

check i TRL! -> es nasa

sla -> service level agreement (serve per testare il prototipo secondo standard)
