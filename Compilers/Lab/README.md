# Download the latest LLVM version

### 1. Scaricare codice sorgente

Source Code $\rightarrow$ https://github.com/llvm/llvm-project/releases/tag/llvmorg-17.0.6

### 2. Creare una root directory secondo il seguente modello:

```bash
./
../
BUILD/
INSTALL/
setup.sh
SRC/
TEST/
```

### 3. Unzippare il codide sorgente in ./SRC e spostarsi in ./BUILD

### 4. Esportare Variabile di Ambinete

```bash
export ROOT=<directory desiderata>
```

Es:

```bash
export ROOT=/home/users/utente/LLVM
```

### 5. Eseguire il seguente comando

**nota:** Devi eseguire il comando da ./BUILD

```bash
cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$ROOT/INSTALL -DLLVM_ENABLE_PROJECTS="clang"
-DLLVM_TARGETS_TO_BUILD=host $ROOT/SRC/llvm/
```

### 6. Compilare e installare il tutto

```bash
# Compilare con: (n rappresenta il numero di core)
make -j n
# Installare eseguendo:
make install
```

Al termine del processo tutti i tools si troveranno instsallat in:

```bash
$ROOT/INSTALL
```
