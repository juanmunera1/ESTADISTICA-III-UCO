import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Datos
data = {
    'GDP': [4, 3.5, 2.1, 1.1, 4.8, 3.2, 2.5, 2.3, 0.9, 3, 103.5, 5.1, 2.8, 1.2, 0.8, 0.6, 2, 1.3, 0.7, 1.7, 1.5, 1, 7.3, 1.8, 6.2, 3.8, 10.5, 22.4, 16.8, 44.1],
    'Population': [0.52, 0.5, 0.3, 0.15, 0.53, 0.45, 0.35, 0.33, 0.08, 0.49, 7.18, 0.76, 0.47, 0.08, 0.04, 0.01, 0.28, 0.13, 0.01, 0.2, 0.22, 0.05, 0.58, 0.25, 0.48, 0.43, 1.03, 2.23, 1.23, 2.57]
}


#MATRIZ DE COVARIANZA
df = pd.DataFrame(data)
X = df[['GDP', 'Population']].values
X_meaned = X - np.mean(X, axis=0)
cov_mat = np.cov(X_meaned, rowvar=False)



#EIGENVALUES
eigenvalues, eigenvectors = np.linalg.eigh(cov_mat)
eigenvalues = eigenvalues[::-1]
eigenvectors = eigenvectors[:, ::-1]



#VARIANZA EXPLICADA POR EL EIGENVALUE
explained_variance_ratio = eigenvalues / np.sum(eigenvalues)



#EIGENVECTOR
eigenvector = eigenvectors[:, 0]



#MATRIZ PROYECTADA
X_pca = X_meaned.dot(eigenvector)




#ERROR O DIFERENCIA ENTRE LA MATRIZ PROYECTADA
reconstructed_X = X_pca[:, np.newaxis] * eigenvector[np.newaxis, :]
error = np.mean(np.square(X_meaned - reconstructed_X))




print("=============PCA===============\n")
print("1. Matriz de Covarianza:\n", cov_mat, "\n")
print("2. Valores Propios (Eigenvalues):", eigenvalues, "\n")
print("Vectores Propios (Eigenvectors):\n", eigenvectors, "\n")
print("3. Varianza Explicada por el Eigenvalue:", explained_variance_ratio[0], "\n")
print("4. Valor del Eigenvector:\n", eigenvector, "\n")
print("5. Matriz Proyectada:\n", X_pca, "\n")
print("6. Error o Diferencia:", error, "\n")






#CIUDADES EN 1 DIMENCION
plt.figure(figsize=(6, 4))
plt.scatter(X_pca, np.zeros_like(X_pca))
plt.title('Datos proyectados a 1 dimensión')
plt.xlabel('Componente Principal')
plt.yticks([])
plt.grid(True)
plt.show()




#CIUDADES EN 2 DIMENCIONES
plt.figure(figsize=(12, 6))
plt.scatter(df['GDP'], df['Population'])
plt.title('Datos en 2 dimensiones')
plt.xlabel('GDP (USD Billion)')
plt.ylabel('Population (Millions)')
plt.grid(True)
plt.show()




