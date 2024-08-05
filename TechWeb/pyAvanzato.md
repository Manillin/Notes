# Concetti avanzati di Python

## Duck Typing

Reso possibile dalla natura interpretativa di python, unitamente alla sua gestione dinamica del concetto di "tipo di dato"

```python
class Duck():
    def verso(self):
        print("quack")

class Cow():
    def verso(self)
        print("Moo")

def fai_verso(a):
    a.verso()

fai_verso(Duck())
fai_verso(Cow())

# quack
# Moo
```

Permette di ottenere comportamenti polimorfici anche tra classi che non condividono relazioni di ereditarietà.  

### Argomenti di default

`def stampa_nome(nome="inserisci nome): ...`  

in questo caso possiamo invocare la funzione senza parametri, ed essa userà il parametro di default al posto di quello esplicito. Questa stessa cosa si può fare anche con le classi!  

In breve:
- I parametri di default non vengono instanziati ogni volta che viene invocato il metodo/funzione
- i parametri di default vengono istanziati **una volta sola**
- Se esiste, **NON** sfrutterà un oggetto passato esplicitamente da una precedente invocazione.  

## VarArgs e KwArgs:  

### varargs:
Possiamo passare una generica lista/tupla contenente una lista di parametri ad una funzione, si usa l'operatore unario `*` per esplodere le strutture dati ed adeguarle ai parametri della funzione.  

### kwargs:  

Usato per espandere dizionari come parametri, ogni parametro in questo caso si differenzia semanticamente da quello successivo.  
Si usa l'operatore unario `**` per esplodere strutture dati di tipo dizionario.  

Es: 

```python
d = {"eta": 77, "l_mansioni": [
    "Giardino", "Reception", "Magazzino"],
    "nome": "mario", "cognome": "rossi"}


def dict_varargs1(**kwargs):
    for e in kwargs:
        print(str(e)+": "+str(kwargs[e]))


def dict_varargs2(nome, cognome, eta, l_mansioni):
    print("nome: " + nome + "cognome: " + cognome + "anni: " + str(eta))
    print("si occupa di: ")
    for _ in l_mansioni:
        print(_)


dict_varargs1(**d)
print("\ninvocazione esplicita: ")
dict_varargs2(**d)

```

Entrambe le chiamate funzionano, nel secondo caso (v2) le key del dizionario vengono messe in accordo con i nomi dei parametri che quindi devono essere di numero fisso e coerenti di nome.   

Gli operatori * e ** possono essere usati anche oltre le funzioni, in particolare per concatenare liste e dizionari.  

```python

l = [1,2]
l2 = [3,4]

lista = [*l1,*l2] # -> [1,2,3,4]

d1 = {'a': 1, 'b': 2}
d2 = {'c': 3, 'd': 4}

dizionario = {**d1,**d2} # {'a': 1, 'b': 2, 'c': 2, 'd': 3}

```


# Python Threads

Python mette a disposizione diversi costrutti per il multi-threading, il più semplice deriva dal package `Threading`

- un thread viene creato specificando la sua funzione target e gli argomenti di tale funzione
- una volta creato viene lanciato dal metodo `start`
- per gestirli sui può usare una lista di thread e una volta finito il loro lavoro vengono ricongiunti da `t.join()`, dove t è il thread  



# Asyncio e co-routines: oltre i threads

Oltre ai thread python (>=3.4) permette di lanciare e definire funzioni e metodi asincroni, permette di aggiungere una dimensioni di parallelismo aggiuntivo senza convolgere necessariamente thread paralleli al main thread della nostra applicazione.  

**Parallelism:** Permette di avere diversi flussi di programam in parallelo.  
**Concurrency** Permette diversi flussi in esecuzioni in finestre temporali che si sovrappongono $\rightarrow$ time sharing  

### Python asyncio
Il package asyncio permette di specificare flussi di esecuzione **concurrent e cooperative**, ossia un flusso parallello/concorrente di istruzioni che **promette** di avere un risultato in futuro (simile a future in Java).  

Esempio: 

```python
async def mossa(mio_tempo, avv_tempo):
    mia_mossa(mio_tempo)
    await mossa_avv(avv_tempo)


def mia_mossa(tempo):
    time.sleep(tempo)


async def mossa_avv(tempo):
    await asyncio.sleep(tempo)
    # dormi ma se hai meglio da fare controlla la lista di altre coroutine da eseguire


async def main():  # async necessario per poter essere aspettato
    OPP = 5
    OPP_TIME = 1
    MY_MOVE_TIME = 1
    MAX_MOVES = 2

    l = []
    # riempiamo lista di funzioni di tipo async
    for _ in range(OPP*MAX_MOVES):
        l.append(mossa(MY_MOVE_TIME, OPP_TIME))

    await asyncio.gather(*l)
    # gather schedula ed esegue 'awaitable objects' uno dopo l'altro

if __name__ == '__main__':
    s = time.perf_counter()
    asyncio.run(main())  # aspettiamo che finiscano tutti
    e = time.perf_counter()
    print("Elapsed time: " + str(e-s))
```

### Osservazioni: 

Se inseriamo `threading.current_thread().name` in ogni funzione notiamo che ogni funzione è eseguita dal Main Thread (no thread in parallelo), questa è la caratteristica 'concurrent'.  
La funzione `await asyncio.sleep(x)` dice 'dormi per x secondi e nel frattempo se hai altro da fare fallo pure'.  

- **async def** $\rightarrow$ indica che la funzione è una co-routine
- **await** $\rightarrow$ indica una funzione da aspettare. Tale funzione deve essere una co-routine, ovvero deve essere in grado di sospendersi per lasciare CPU time ad altre funzioni.  
- **`asyncio.gather(*coroutine_objects)`** $\rightarrow$ schedula ed esegue awaitable objects uno dietro l'altro
- **run** $\rightarrow$ esegue ed aspetta il risultato di una o più co-routine.  

