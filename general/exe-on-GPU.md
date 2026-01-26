## Flusso di esecuzione di un'operazione GPU, dal codice Python all'hardware

### Fase 1: Allocazione Memoria (User Space)
*   **Contesto:** Python (User Space).
*   **Codice:** `a = torch.randn(1000, 1000).to('cuda')`.
*   **Azione:**
    1.  L'interprete Python invoca le API di **CUDA Runtime** (libreria utente).
    2.  CUDA richiede l'allocazione di memoria sulla GPU.
    3.  Il Driver Nvidia mappa una porzione di **VRAM** (Video RAM) sulla GPU e restituisce a Python un **puntatore** (un indirizzo di memoria virtuale GPU, es: `0xA000`).
    4.  Python ora possiede solo il puntatore `0xA000`, non i dati fisici (che sono nella VRAM della GPU).

### Fase 2: Preparazione del "Command Buffer" (User Space)
*   **Contesto:** Python / Librerie Framework (PyTorch/cuBLAS).
*   **Codice:** `c = torch.matmul(a, b)`.
*   **Azione:**
    1.  PyTorch non esegue calcoli. PyTorch agisce come un pianificatore.
    2.  Alloca memoria per il risultato `c` (es: puntatore `0xC000`).
    3.  Seleziona il **Kernel CUDA** appropriato (il binario compilato per la GPU che sa fare la moltiplicazione di matrici, es: `sgemm`). *Nota: Qui "Kernel" significa funzione GPU, non Kernel OS.*
    4.  PyTorch scrive in una zona di memoria speciale (chiamata **Pushbuffer** o **Command Queue**) una sequenza di istruzioni binarie:
        *   *Instruction 1:* Carica il Kernel CUDA `sgemm`.
        *   *Instruction 2:* Leggi dati da puntatore `0xA000` e `0xB000`.
        *   *Instruction 3:* Scrivi risultato su puntatore `0xC000`.
        *   *Instruction 4:* Configurazione della Grid/Block size (parallelismo).

### Fase 3: La System Call (Transizione User -> Kernel)
*   **Contesto:** Transizione da User Mode a Kernel Mode.
*   **Azione:**
    1.  La libreria CUDA deve inviare questo Command Buffer alla GPU, ma non ha accesso diretto all'hardware.
    2.  Esegue una **System Call** (solitamente `ioctl` su Linux) passando il File Descriptor del device (`/dev/nvidiaX`) e il puntatore al Command Buffer appena creato.
    3.  La CPU cambia modalità di esecuzione (Context Switch) ed entra nel **Linux Kernel**.

### Fase 4: Il Driver NVIDIA (Kernel Space)
*   **Contesto:** Linux Kernel (Modulo `nvidia.ko`).
*   **Azione:**
    1.  Il Kernel Linux riceve la `ioctl` e la delega al **Driver NVIDIA** registrato per quel device.
    2.  Il Driver esegue controlli di validità e sicurezza.
    3.  **MMIO (Memory Mapped I/O):** Il Driver deve avvisare la GPU che c'è lavoro da fare. La CPU (che sta eseguendo il codice del Driver) scrive un valore specifico in un indirizzo fisico di memoria riservato (Bar Address Register).
    4.  Questa scrittura non va nella RAM, ma viene intercettata dal controller PCIe della GPU. È l'equivalente digitale di "suonare il campanello".

### Fase 5: Esecuzione Hardware (GPU)
*   **Contesto:** Hardware GPU (Processore interno / Firmware).
*   **Azione:**
    1.  **DMA (Direct Memory Access):** Il processore di comando della GPU (GIGAThread Engine o simili) legge il Command Buffer dalla RAM di sistema senza coinvolgere la CPU.
    2.  **Scheduling:** La GPU distribuisce il lavoro ai suoi Streaming Multiprocessors (SM).
    3.  **Fetch & Execute:**
        *   I Core della GPU caricano i dati dalla VRAM (`0xA000`, `0xB000`) nei registri interni e nella Cache L1/L2.
        *   Le ALU (Arithmetic Logic Units) o i Tensor Cores eseguono le operazioni matematiche.
    4.  **Write Back:** Il risultato finale viene scritto nella VRAM all'indirizzo `0xC000`.

### Fase 6: Interrupt e Notifica (Hardware -> Kernel)
*   **Contesto:** Hardware -> CPU.
*   **Azione:**
    1.  La GPU ha finito. Invia un segnale di **Interrupt (IRQ)** sul bus PCIe verso la CPU.
    2.  La CPU sospende momentaneamente ciò che sta facendo ed esegue l'**Interrupt Handler** del Driver NVIDIA.
    3.  Il Driver aggiorna lo stato interno del job segnandolo come "COMPLETATO".

### Fase 7: Sincronizzazione e Lettura (User Space)
*   **Contesto:** Ritorno a Python.
*   **Codice:** `print(c)`.
*   **Azione:**
    1.  Python (che nel frattempo poteva essere in attesa bloccante o polling) riceve la conferma che l'operazione è finita.
    2.  Il puntatore `0xC000` ora contiene dati validi.
    3.  Per stampare a video, i dati devono tornare nella RAM della CPU.
    4.  Python invoca `cudaMemcpy` (Device-to-Host).
    5.  Il Driver ordina alla GPU (via DMA) di copiare i dati da VRAM `0xC000` a un buffer nella RAM di sistema.
    6.  Python legge il buffer in RAM e mostra il risultato.


### metafora del ristorante
Python (Tu): Compili la comanda sul foglietto ("Voglio la Carbonara, tavolo 4"). Non cucini tu.  
System Call (Cameriere): Porta il foglietto in cucina. Non cucina lui, porta solo il messaggio.  
Driver (Capo Partita): Prende il foglietto e lo appende alla bacheca degli ordini. Urla "Ordine in arrivo!". Non cucina lui.  
GPU (Cuoco): Legge il foglietto, prende le uova e la pasta (Dati) dal frigo (VRAM) e cucina. Lui sa come si fa la carbonara (Moltiplicazione), il Driver sapeva solo appendere il foglietto.   
Interrupt (Campanello): DRIIIN! Piatto pronto.  


--- 

