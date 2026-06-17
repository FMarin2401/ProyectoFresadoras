# Diagnóstico de Salud en Fresadoras Industriales

## Definición
La industria manufacturera enfrenta pérdidas significativas por fallas imprevistas. Este proyecto aborda la predicción anticipada de fallas en equipos rotativos (fresadoras) para transicionar de un mantenimiento preventivo a uno predictivo. Se define como una tarea de Clasificación Binaria Supervisada.

## Base de Datos
Se utilizó el "AI4I 2020 Predictive Maintenance Dataset" del repositorio de UCI. Consta de 10,000 instancias y evalúa variables como:
- Temperatura del aire y del proceso.
- Velocidad de rotación y Torque.
- Desgaste de la herramienta.

## Modelos Evaluados
Se entrenaron y compararon tres algoritmos para lidiar con el desbalance de clases (las fallas representan solo el 3.4% de los datos):
1. **Random Forest** (Modelo Seleccionado)
2. Regresión Logística
3. Support Vector Machine (SVM)

## Resultados
El modelo ganador fue **Random Forest Optimizado**, demostrando un mejor equilibrio para una aplicación real
* **Recall:** 0.750 (Logra anticipar 3 de 4 averías antes de que ocurran).
* **Precisión:** 0.580
* **F1-Score:** 0.654

Se validó la robustez del modelo aplicando Bootstrapping con 1000 iteraciones, asegurando que las métricas no son producto del azar.

## Estructura del Repositorio
- `/data`: Contiene el dataset original `ai4i2020.csv`.
- `/notebooks`: Contiene el análisis exploratorio y entrenamiento en `ProyectoFresadoras.ipynb`.
- `/src`: Código fuente de los modelos en `Onlycode.py`.
- `/docs`: Documentación detallada del proyecto.