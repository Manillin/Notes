import matplotlib.pyplot as plt
import numpy as np

# Definisci i vettori
a = np.array([3, 4])
b = np.array([2, 0])

# Calcola la proiezione di b su a
projection = (np.dot(a, b) / np.dot(a, a)) * a

# Crea il grafico
plt.figure(figsize=(8, 8))
plt.quiver(0, 0, a[0], a[1], angles='xy', scale_units='xy',
           scale=1, color='r', label='Vector a')
plt.quiver(0, 0, b[0], b[1], angles='xy', scale_units='xy',
           scale=1, color='b', label='Vector b')
plt.quiver(0, 0, projection[0], projection[1], angles='xy',
           scale_units='xy', scale=1, color='g', label='Projection of b onto a')

# Annota il grafico
plt.text(a[0], a[1], 'a', color='r', fontsize=12)
plt.text(b[0], b[1], 'b', color='b', fontsize=12)
plt.text(projection[0], projection[1], 'proj_b_onto_a', color='g', fontsize=12)

# Imposta i limiti e l'aspetto
plt.xlim(-1, 4)
plt.ylim(-1, 5)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.gca().set_aspect('equal', adjustable='box')

# Aggiungi la legenda
plt.legend()

# Mostra il grafico
plt.title("Projection of Vector b onto Vector a")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()
