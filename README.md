# Neurona artificial para el riego de plantas

Neurona artificial (perceptrón con activación sigmoide) que decide si una planta necesita ser regada a partir de dos variables:

|Variable|Descripción|Escala usada para normalizar|
|-|-|-|
|`x1`|Humedad del suelo (%)|100|
|`x2`|Temperatura ambiental (°C)|50|
|`y`|1 = regar, 0 = no regar|—|

> Los datos son didácticos y no representan una recomendación agronómica para una especie real.

## Ejecución

Requiere [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Cristian335-creaty/neurona-riego-plantas.git
cd neurona-riego-plantas/src/neurona-riego-plantas
uv sync
uv run main.py
```

## Funcionamiento

1. **Datos:** `X` contiene los 10 casos (humedad, temperatura) y `y` las respuestas esperadas, ambos con `dtype=float`.
2. **Normalización:** `X\_normalizado = X / escala`, con `escala = np.array(\[100, 50])`. Los datos nuevos se dividen por la **misma** `escala`.
3. **Suma ponderada:** `z = X\_normalizado @ pesos + sesgo` (dos pesos y un sesgo).
4. **Activación:** `sigmoide(z) = 1 / (1 + e^(-z))` → probabilidad entre 0 y 1.
5. **Error:** error cuadrático medio (ECM) entre la probabilidad y `y`.
6. **Gradientes** (regla de la cadena):

   * `gradiente\_z = (2/n) · error · p · (1 − p)`
   * `gradiente\_pesos = X\_normalizado.T @ gradiente\_z`
   * `gradiente\_sesgo = Σ gradiente\_z`
7. **Actualización:** `pesos -= tasa · gradiente\_pesos`, `sesgo -= tasa · gradiente\_sesgo`.
8. **Salida binaria:** `respuesta = (probabilidad >= 0.5).astype(int)`.

Los pesos iniciales se generan con una semilla fija (`42`) para que los resultados sean reproducibles.

## Resultados del entrenamiento base (10 000 épocas, tasa 0.5)

|Parámetro|Valor|
|-|-|
|Peso humedad (w1)|**−19.5201**|
|Peso temperatura (w2)|**+2.1946**|
|Sesgo (b)|7.4780|
|Error final (ECM)|0.019540|

Evolución del error: 0.2534 (época 1) → 0.0670 (1 000) → 0.0305 (5 000) → 0.0195 (10 000).

|Caso|Humedad|Temperatura|Esperado|Probabilidad|Salida|
|-|-|-|-|-|-|
|1|80 %|18 °C|0|0.0006|0 |
|2|70 %|22 °C|0|0.0054|0 |
|3|65 %|28 °C|0|0.0183|0 |
|4|55 %|25 °C|0|0.1033|0 |
|5|50 %|32 °C|0|0.2937|0 |
|6|40 %|30 °C|1|0.7284|1 |
|7|35 %|25 °C|1|0.8511|1 |
|8|30 %|32 °C|1|0.9538|1 |
|9|20 %|35 °C|1|0.9940|1 |
|10|10 %|38 °C|1|0.9992|1 |

**Respuestas correctas: 10/10.**

### Significado del signo de los pesos

* **w1 (humedad) negativo:** cuanto más húmedo está el suelo, menor es `z` y menor la probabilidad de regar. Es la variable dominante (|w1| ≈ 19.5 frente a |w2| ≈ 2.2), lo cual es coherente con los datos: la frontera entre regar y no regar está entre 40 % y 50 % de humedad.
* **w2 (temperatura) positivo:** a mayor temperatura, mayor probabilidad de regar (más evaporación). Su efecto es menor y sirve para inclinar la decisión en casos dudosos.
* **Sesgo positivo:** desplaza la frontera de decisión; sin él la neurona no podría ubicar el punto de corte en ≈45 % de humedad.

## Pruebas con condiciones nuevas

Normalizadas con la misma `escala = \[100, 50]`.

|Humedad|Temperatura|Probabilidad|Decisión|
|-|-|-|-|
|75 %|30 °C|0.0029|0 – No regar|
|45 %|34 °C|0.5464|1 – Regar|
|25 %|22 °C|0.9724|1 – Regar|
|50 %|25 °C|0.2342|0 – No regar|
|30 %|40 °C|0.9670|1 – Regar|

El caso 45 % / 34 °C es el más dudoso: está justo en la frontera de humedad, y la temperatura alta lo inclina levemente hacia "regar".

## Experimentos con los parámetros

En cada prueba solo se cambió un parámetro respecto a la prueba base. Misma semilla en todas.

|Prueba|Épocas|Tasa|ECM final|Correctas|P(45 %, 34 °C)|Aprendizaje|
|-|-|-|-|-|-|-|
|Prueba base|10 000|0.5|0.019540|10/10|0.5464|Rápido|
|Pocas épocas|100|0.5|0.169906|10/10|0.5138|Insuficiente|
|Cantidad intermedia|1 000|0.5|0.066998|10/10|0.5739|Lento|
|Más épocas|20 000|0.5|0.011670|10/10|0.5246|Rápido|
|Tasa pequeña|10 000|0.01|0.131325|10/10|0.5318|Insuficiente|
|Tasa moderada|10 000|0.1|0.049286|10/10|0.5804|Lento|
|Tasa alta|10 000|1.0|0.011669|10/10|0.5246|Rápido|
|Tasa muy alta|10 000|2.0|0.006556|10/10|0.5070|Rápido|

Observaciones:

* Todas las configuraciones clasifican bien los 10 casos porque los datos son **linealmente separables** y casi solo dependen de la humedad. Lo que cambia es la **confianza**: con poco entrenamiento las probabilidades quedan cerca de 0.5 (ECM alto) y con más entrenamiento se acercan a 0 y 1.
* "Más épocas" (20 000, tasa 0.5) y "Tasa alta" (10 000, tasa 1.0) dan prácticamente el mismo ECM (0.01167): **duplicar la tasa equivale aproximadamente a duplicar las épocas**, porque el número total de pasos "efectivos" del descenso es el mismo.
* La tasa muy alta (2.0) no se volvió inestable con estos datos: como las entradas están normalizadas y la sigmoide con ECM tiene gradientes pequeños (la derivada `p(1−p)` es como máximo 0.25), los pasos no llegan a "saltarse" el mínimo. Con datos sin normalizar o con una tasa aún mayor sí aparecerían oscilaciones.

## Prueba adicional con el umbral

Sin volver a entrenar (mismos pesos y sesgo):

|Caso|Humedad|Temperatura|Probabilidad|u = 0.4|u = 0.5|u = 0.6|
|-|-|-|-|-|-|-|
|1–5|—|—|0.0006 – 0.2937|0|0|0|
|6–10|—|—|0.7284 – 0.9992|1|1|1|
|Nuevo|45 %|34 °C|0.5464|1|1|**0**|
|Otros nuevos|—|—|0.0029 / 0.2342 / 0.9670 / 0.9724|sin cambio|sin cambio|sin cambio|

* En los datos de entrenamiento **ningún caso cambia**: todas las probabilidades están por debajo de 0.30 o por encima de 0.72, lejos de los tres umbrales.
* Solo cambia el caso nuevo **45 % / 34 °C** (p = 0.5464): con umbral 0.6 pasa a "no regar".
* El umbral se aplica **después** de la sigmoide; no interviene en el cálculo de `z`, del error ni de los gradientes. Por eso cambiarlo no modifica los pesos aprendidos, solo el punto de corte para convertir la probabilidad en 0 o 1. Un umbral bajo (0.4) hace a la neurona más "propensa a regar"; uno alto (0.6) más conservadora.

## Análisis

**1. ¿Por qué fue necesario normalizar la humedad y la temperatura?**
Porque están en escalas diferentes (0–100 y 0–50). Sin normalizar, la variable con números más grandes produciría gradientes más grandes y dominaría la actualización de los pesos; además, valores grandes de `z` saturan la sigmoide (su derivada se vuelve casi 0) y el aprendizaje se detiene. Al dividir por 100 y 50, ambas quedan entre 0 y 1 y contribuyen de forma comparable.

**2. ¿En qué operaciones se utilizó `X\_normalizado` y para qué se conservó `X`?**
`X\_normalizado` se usó en la suma ponderada (`z = X\_normalizado @ pesos + sesgo`) y en el gradiente de los pesos (`X\_normalizado.T @ gradiente\_z`). `X` se conservó para mostrar los valores originales (% y °C) en las tablas, de modo que los resultados se puedan interpretar con unidades reales.

**3. ¿Qué ocurrió al utilizar solamente 100 épocas?**
El error quedó alto (ECM ≈ 0.17). Aunque acertó los 10 casos, las probabilidades quedaron muy cerca de 0.5, es decir, la neurona apenas empezaba a separar las clases y sus decisiones eran poco confiables. El aprendizaje fue insuficiente.

**4. ¿Más épocas siempre produjeron una mejora importante?**
No. De 100 a 1 000 épocas el ECM bajó de 0.170 a 0.067, y de 1 000 a 10 000 bajó a 0.020; pero de 10 000 a 20 000 solo bajó a 0.012. Las mejoras son cada vez menores (rendimientos decrecientes) y la exactitud ya era 10/10 desde el principio. Más épocas también cuestan más tiempo de cómputo.

**5. ¿Qué efecto tuvo una tasa de aprendizaje demasiado pequeña?**
Con tasa 0.01 los pasos fueron tan cortos que, después de 10 000 épocas, el ECM (0.131) seguía siendo peor que con 1 000 épocas y tasa 0.5. El aprendizaje fue muy lento: necesitaría muchísimas más épocas para llegar al mismo resultado.

**6. ¿Qué efecto tuvo una tasa de aprendizaje alta o muy alta?**
Con 1.0 y 2.0 el error bajó más rápido (0.0117 y 0.0066). En este problema no hubo inestabilidad porque las entradas están normalizadas y los gradientes de la sigmoide son pequeños. Sin embargo, en general una tasa demasiado alta puede hacer que los pesos oscilen o diverjan alrededor del mínimo; por eso conviene probarla y vigilar la curva del error.

**7. ¿Qué representa el signo del peso correspondiente a la humedad?**
Es negativo: la humedad tiene una relación **inversa** con la necesidad de riego. Un suelo más húmedo reduce `z` y la probabilidad de regar.

**8. ¿Qué representa el signo del peso correspondiente a la temperatura?**
Es positivo: la temperatura tiene una relación **directa** con la necesidad de riego. Más calor aumenta `z` y la probabilidad de regar, aunque con una influencia menor que la humedad.

**9. ¿Por qué una probabilidad debe convertirse en 0 o 1 mediante un umbral?**
Porque la sigmoide entrega un valor continuo (p. ej. 0.73), pero la decisión que se necesita es discreta: regar o no regar. El umbral define a partir de qué probabilidad se toma la acción; 0.5 es el punto neutro, y puede ajustarse según el costo de equivocarse (regar de más vs. dejar secar la planta).

**10. ¿Qué limitaciones tiene esta neurona para representar el riego de una planta real?**

* Solo usa dos variables; ignora el tipo de planta, tipo de suelo, luz solar, humedad del aire, viento, lluvia prevista, etapa de crecimiento y hora del día.
* Se entrenó con solo 10 ejemplos didácticos, sin datos reales ni validación con casos independientes.
* Una sola neurona solo traza una **frontera lineal**; no puede representar relaciones no lineales (p. ej., que tanto el exceso como la falta de agua sean dañinos).
* Solo decide "sí/no"; no indica **cuánta** agua aplicar ni con qué frecuencia.
* Fuera del rango de los datos (temperaturas bajo cero, humedad > 100 %) sus predicciones no tienen respaldo.

## Estructura del repositorio

```
neurona-riego-plantas/
├── src/neurona-riego-plantas/main.py           # Desarrollo de la neurona, experimentos y umbrales
├── pyproject.toml    # Configuración del proyecto (uv)
├── uv.lock           # Versiones exactas de las dependencias
├── .python-version
├── .gitignore        # Excluye .venv
└── README.md
```

