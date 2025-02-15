import numpy as np
import matplotlib.pyplot as plt

# Definiamo i parametri dell'iperpiano (retta in 2D): 2x1 + x2 - 3 = 0
w = np.array([2, 1])  # Vettore normale
b = -3  # Bias

# Definiamo un punto fuori dall'iperpiano
x_i = np.array([1, 4])

# Generiamo punti per disegnare l'iperpiano (retta)
x1_values = np.linspace(-2, 4, 100)
# Ricaviamo x2 dalla formula della retta
x2_values = (-w[0] * x1_values - b) / w[1]

# Calcoliamo la distanza senza normalizzazione
proj_non_norm = np.dot(w, x_i) + b
point_on_hyperplane_non_norm = x_i - \
    proj_non_norm * w  # Spostiamo il punto lungo w

# Calcoliamo la distanza normalizzata
norm_w = np.linalg.norm(w)
distance_signed = proj_non_norm / norm_w
point_on_hyperplane = x_i - (distance_signed * w) / \
    norm_w  # Proiezione ortogonale

# Plot
plt.figure(figsize=(8, 6))
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(True, linestyle='--', linewidth=0.5)

# Disegniamo la retta
plt.plot(x1_values, x2_values, 'b-', label='Iperpiano: $2x_1 + x_2 - 3 = 0$')

# Disegniamo il punto
plt.scatter(x_i[0], x_i[1], color='red', label='Punto $x_i$ (1,4)')

# Disegniamo la distanza non normalizzata
plt.plot([x_i[0], point_on_hyperplane_non_norm[0]],
         [x_i[1], point_on_hyperplane_non_norm[1]],
         'r--', label='Distanza non normalizzata')

# Disegniamo la distanza normalizzata
plt.plot([x_i[0], point_on_hyperplane[0]],
         [x_i[1], point_on_hyperplane[1]],
         'g-', linewidth=2, label='Distanza normalizzata')

# Etichette
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.legend()
plt.title('Distanza di un punto da un iperpiano in $\mathbb{R}^2$')
plt.show()
