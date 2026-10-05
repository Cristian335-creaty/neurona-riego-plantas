import numpy as np

X = np.array([
    [80, 18], [70, 22], [65, 28], [55, 25], [50, 32],
    [40, 30], [35, 25], [30, 32], [20, 35], [10, 38]
], dtype=float)

y = np.array([[0], [0], [0], [0], [0],
              [1], [1], [1], [1], [1]], dtype=float)

escala = np.array([100, 50])
X_normalizado = X / escala
print("\nDatos normalizados:")
print(X_normalizado)

def sigmoide(z):
    return 1 / (1 + np.exp(-z))

def entrenar(tasa_aprendizaje, epocas, mostrar=False):
    rng = np.random.default_rng(42)          # semilla fija: siempre da lo mismo
    pesos = rng.normal(0, 0.1, size=(2, 1))  # w1 (humedad), w2 (temperatura)
    sesgo = 0.0
    n = len(X_normalizado)

    for epoca in range(1, epocas + 1):
        # 1) Suma ponderada y activación
        z = X_normalizado @ pesos + sesgo
        probabilidad = sigmoide(z)

        # 2) Error cuadrático medio (ECM)
        error = probabilidad - y
        ecm = np.mean(error ** 2)

        # 3) Gradientes (regla de la cadena)
        gradiente_z = (2 / n) * error * probabilidad * (1 - probabilidad)
        gradiente_pesos = X_normalizado.T @ gradiente_z
        gradiente_sesgo = np.sum(gradiente_z)

        # 4) Actualizar pesos y sesgo
        pesos -= tasa_aprendizaje * gradiente_pesos
        sesgo -= tasa_aprendizaje * gradiente_sesgo

        if mostrar and (epoca == 1 or epoca % 1000 == 0):
            print(f"Época {epoca}: ECM = {ecm:.6f}")

    return pesos, sesgo, ecm

def predecir(datos, pesos, sesgo):
    return sigmoide((datos / escala) @ pesos + sesgo)

pesos, sesgo, ecm = entrenar(0.5, 10000, mostrar=True)
print(f"\nPeso humedad: {pesos[0,0]:.4f}")
print(f"Peso temperatura: {pesos[1,0]:.4f}")
print(f"Sesgo: {sesgo:.4f}")
print(f"Error final: {ecm:.6f}")

prob = predecir(X, pesos, sesgo)
respuesta = (prob >= 0.5).astype(int)
for i in range(10):
    print(f"Caso {i+1}: {X[i]} esperado={int(y[i,0])} "
          f"prob={prob[i,0]:.4f} salida={respuesta[i,0]}")
print("Correctas:", int(np.sum(respuesta == y)), "/ 10")

X_nuevos = np.array([[75, 30], [45, 34], [25, 22], [50, 25], [30, 40]], dtype=float)
p_nuevos = predecir(X_nuevos, pesos, sesgo)
print("\nCondiciones nuevas:")
for fila, p in zip(X_nuevos, p_nuevos):
    print(f"{fila} -> prob={p[0]:.4f} decisión={int(p[0] >= 0.5)}")

experimentos = [("Base", 10000, 0.5), ("Pocas épocas", 100, 0.5),
                ("Intermedia", 1000, 0.5), ("Más épocas", 20000, 0.5),
                ("Tasa pequeña", 10000, 0.01), ("Tasa moderada", 10000, 0.1),
                ("Tasa alta", 10000, 1.0), ("Tasa muy alta", 10000, 2.0)]
print("\nExperimentos:")

for nombre, ep, tasa in experimentos:
    w, b, e = entrenar(tasa, ep)
    correctas = int(np.sum((predecir(X, w, b) >= 0.5) == y))
    p45 = predecir(np.array([[45, 34]]), w, b)[0, 0]
    print(f"{nombre}: ECM={e:.6f} correctas={correctas}/10 P(45,34)={p45:.4f}")

print("\nUmbrales:")
todos = np.vstack([X, X_nuevos])
p_todos = predecir(todos, pesos, sesgo)
for fila, p in zip(todos, p_todos):
    print(fila, [int(p[0] >= u) for u in (0.4, 0.5, 0.6)])

