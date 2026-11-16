# TP3 - Regularización de un MLP para clasificar texto (20 Newsgroups)

Trabajo Práctico 3 de Redes Neuronales en Bioingeniería (ITBA). Sobre el clasificador MLP de 20 Newsgroups del TP2, se incluyen como hiperparámetros la regularización L1, L2 y L1+L2, el dropout (con dropout en la entrada y max-norm) y la BatchNormalization. Se analizan la magnitud de los pesos, los tiempos de entrenamiento, el learning rate y el ruido de las curvas de aprendizaje.

## Contenido

| Archivo | Descripción |
|---|---|
| `TP3_Regularizacion_DELIA.ipynb` | Notebook ejecutado, con todos los outputs y los análisis. |
| `requirements.txt` | Dependencias de Python. |
| `.gitignore` | Excluye los archivos que genera la ejecución (base de MLflow, checkpoints). |

El dataset (20 Newsgroups) se descarga automáticamente con `sklearn.datasets.fetch_20newsgroups`.

## Cómo reproducirlo

Se instalan las dependencias con `pip install -r requirements.txt` y se ejecuta el notebook completo. Con una GPU T4 la corrida tarda alrededor de una hora, y sin GPU puede llevar varias horas. Si se define la variable de entorno `TP3_MODO_RAPIDO=1`, el notebook corre una versión reducida en pocos minutos, que sirve para comprobar que el código funciona pero no reproduce los resultados del análisis.

## Metodología

- El preprocesamiento (stopwords, lematización, unigramas, 40.000 términos) y el split train/validación (80/20, estratificado, `random_state=42`) son los de los TP1 y TP2.
- Los experimentos controlados agregan una técnica por vez sobre una red sobredimensionada sin regularizar (2 capas de 512) y se repiten con 3 semillas (media ± desvío).
- L1 y L2 se aplican a los pesos de las capas, no a los sesgos. La L2 se compara en dos formas: dentro de la función de costo (acoplada con Adam) y como weight decay desacoplado (AdamW).
- El modelo final se elige con validación, volviendo a entrenar las mejores candidatas con 3 semillas para reducir el sesgo optimista de quedarse con el máximo de una búsqueda.
- El test se usa una sola vez, al final, con 3 semillas por modelo. Durante el entrenamiento final no se mide nada sobre test.

## Uso de herramientas de IA

Declarado en la sección 10 del notebook.
