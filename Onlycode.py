# Librerías generales de datos y visualización
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# Librerías de Machine Learning (Scikit-Learn)
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
# Carga y exploración de datos
archivo = 'ai4i2020.csv' 
df = pd.read_csv(archivo)
print("Vista previa de los datos")
display(df.head())
print("--- INFORMACIÓN DEL DATASET ---")
print(f"Fuente: UCI Machine Learning Repository")
print(f"Tamaño (Filas, Columnas): {df.shape}")
print("\nTipos de Variables:")
print(df.dtypes)
print("\n--- DISTRIBUCIÓN DE FALLAS ---")
conteo = df['Machine failure'].value_counts()
print(f"Máquinas OK (0): {conteo[0]}")
print(f"Máquinas con Falla (1): {conteo[1]}")
# Eliminamos identificadores que no aportan valor
df_clean = df.drop(columns=['UDI', 'Product ID'])
# Convertimos variables categóricas en dummy
df_clean = pd.get_dummies(df_clean, columns=['Type'], drop_first=True)
# Separamos X (predictoras) e y (objetivo), evitando "Data Leakage"
X = df_clean.drop(columns=['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'])
y = df_clean['Machine failure']
# División en Train y Test (Estratificado por el desbalance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)
# Estandarización de los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# ENTRENAMIENTO Y EVALUACIÓN: Random Forest (Modelo Base)
rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')
rf_model.fit(X_train_scaled, y_train)
y_pred = rf_model.predict(X_test_scaled)
print("\n--- REPORTE DE CLASIFICACIÓN ---")
print(classification_report(y_test, y_pred))
# Matriz de Confusión del modelo optimizado
cm_opt = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm_opt, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicción OK', 'Predicción Falla'], 
            yticklabels=['Real OK', 'Real Falla'])
plt.title('Matriz de Confusión - Modelo RF Base')
plt.show()
df_importancias = pd.DataFrame({
    'Variable': X.columns,
    'Importancia': rf_model.feature_importances_
}).sort_values(by='Importancia', ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x='Importancia', y='Variable', data=df_importancias)
plt.title('Importancia de las Variables (Random Forest Base)', fontsize=14)
plt.xlabel('Importancia Relativa (0 a 1)', fontsize=12)
plt.ylabel('Variable', fontsize=12)
plt.tight_layout()
plt.show()
######### Mostrar la tabla exacta
print("\nTabla de Importancia de Variables:")
display(df_importancias)
# Eliminamos las variables 'Type' tras comprobar que su importancia es casi nula
columnas_a_eliminar = [
    'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF', 
    'Type_L', 'Type_M'
]
X_optimizada = df_clean.drop(columns=columnas_a_eliminar)
print(f"\nNueva forma de X Optimizada:", X_optimizada.shape)
# Nueva división y escalado
X_train_opt, X_test_opt, y_train, y_test = train_test_split(
    X_optimizada, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)
scaler_opt = StandardScaler()
X_train_opt_scaled = scaler_opt.fit_transform(X_train_opt)
X_test_opt_scaled = scaler_opt.transform(X_test_opt)
# Entrenamiento del modelo optimizado
rf_model_opt = RandomForestClassifier(random_state=42, class_weight='balanced')
rf_model_opt.fit(X_train_opt_scaled, y_train)
y_pred_opt = rf_model_opt.predict(X_test_opt_scaled)
print("\n--- REPORTE DE CLASIFICACIÓN (MODELO OPTIMIZADO) ---")
print(classification_report(y_test, y_pred_opt))
# Matriz de Confusión del modelo optimizado
cm_opt = confusion_matrix(y_test, y_pred_opt)
plt.figure(figsize=(6,4))
sns.heatmap(cm_opt, annot=True, fmt='d', cmap='Greens', 
            xticklabels=['Predicción OK', 'Predicción Falla'], 
            yticklabels=['Real OK', 'Real Falla'])
plt.title('Matriz de Confusión - RF Optimizado (Sin "Type")')
plt.show()
######### Definir la cuadrícula de parámetros a probar
# Vamos a probar diferentes cantidades de árboles y profundidades
param_grid = {
    'n_estimators': [50, 100, 200],         # Cantidad de árboles en el bosque
    'max_depth': [None, 10, 20],            # Profundidad máxima de cada árbol
    'min_samples_split': [2, 5, 10],        # Mínimo de muestras para dividir un nodo
    'class_weight': ['balanced']            # Siempre mantenemos esto para clases desbalanceadas
}
######### Inicializar el modelo base y el GridSearchCV
rf_base = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(
    estimator=rf_base, 
    param_grid=param_grid, 
    cv=5,                 # Validación cruzada de 5 pliegues
    scoring='recall',     # Metrica enfocada
    n_jobs=-1             # Nucleos del procesador para acelerar el proceso
)
######### Ejecutar la búsqueda con nuestros datos optimizados y escalados
grid_search.fit(X_train_opt_scaled, y_train)
######### Ver los resultados ganadores
print(f"Mejores parámetros encontrados:\n {grid_search.best_params_}")
######### Evaluar el modelo ganador
best_rf_model = grid_search.best_estimator_
y_pred_best = best_rf_model.predict(X_test_opt_scaled)
print("\n--- REPORTE DE CLASIFICACIÓN (Mejor Modelo) ---")
print(classification_report(y_test, y_pred_best))
# Matriz de confusión final
cm_best = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(6,4))
sns.heatmap(cm_best, annot=True, fmt='d', cmap='Oranges', 
            xticklabels=['Predicción OK', 'Predicción Falla'], 
            yticklabels=['Real OK', 'Real Falla'])
plt.title('Matriz de Confusión - RF (Mejores Hiperparámetros)')
plt.show()
# --- REGRESIÓN LOGÍSTICA (OPTIMIZADA CON GRID SEARCH) ---
# Definir la cuadrícula de parámetros
param_grid_lr = {
    'C': [0.01, 0.1, 1, 10, 100],        # C controla la regularización (inverso a la fuerza)
    'solver': ['liblinear', 'lbfgs']     # Algoritmos de optimización
}
# Inicializar el modelo base y GridSearchCV
lr_base = LogisticRegression(random_state=42, class_weight='balanced', max_iter=2000)
grid_search_lr = GridSearchCV(
    estimator=lr_base,
    param_grid=param_grid_lr,
    cv=5,                 # Validación cruzada de 5 pliegues
    scoring='recall',     # Optimizamos para Recall
    n_jobs=-1
)
# Realizar la búsqueda
grid_search_lr.fit(X_train_opt_scaled, y_train)
# Ver los resultados ganadores
print(f"Mejores parámetros encontrados (LR): {grid_search_lr.best_params_}")
# Evaluar el modelo ganador
best_lr_model = grid_search_lr.best_estimator_
y_pred_lr_best = best_lr_model.predict(X_test_opt_scaled)
print("\n--- REPORTE DE CLASIFICACIÓN (REGRESIÓN LOGÍSTICA OPTIMIZADA) ---")
print(classification_report(y_test, y_pred_lr_best))
# Matriz de confusión
cm_lr = confusion_matrix(y_test, y_pred_lr_best)
plt.figure(figsize=(6,4))
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Purples', 
            xticklabels=['Predicción OK', 'Predicción Falla'], 
            yticklabels=['Real OK', 'Real Falla'])
plt.title('Matriz de Confusión - LR (Mejores Hiperparámetros)')
plt.show()
# --- SUPPORT VECTOR MACHINE - SVM (OPTIMIZADO CON GRID SEARCH) ---
# Definir la cuadrícula de parámetros
param_grid_svm = {
    'C': [0.1, 1, 10],                 # Margen de error permitido
    'kernel': ['linear', 'rbf'],       # Tipo de línea de separación (recta o curva)
    'gamma': ['scale', 'auto']         # Influencia de las muestras individuales (solo para rbf)
}
# Inicializar el modelo base y GridSearchCV
svm_base = SVC(random_state=42, class_weight='balanced')
grid_search_svm = GridSearchCV(
    estimator=svm_base,
    param_grid=param_grid_svm,
    cv=5,
    scoring='recall',
    n_jobs=-1
)
# Realizar la búsqueda
grid_search_svm.fit(X_train_opt_scaled, y_train)
# Ver los resultados ganadores
print(f"Mejores parámetros encontrados (SVM): {grid_search_svm.best_params_}")
# Evaluar el modelo ganador
best_svm_model = grid_search_svm.best_estimator_
y_pred_svm_best = best_svm_model.predict(X_test_opt_scaled)
print("\n--- REPORTE DE CLASIFICACIÓN (SVM OPTIMIZADO) ---")
print(classification_report(y_test, y_pred_svm_best))
# Matriz de confusión
cm_svm = confusion_matrix(y_test, y_pred_svm_best)
plt.figure(figsize=(6,4))
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Reds', 
            xticklabels=['Predicción OK', 'Predicción Falla'], 
            yticklabels=['Real OK', 'Real Falla'])
plt.title('Matriz de Confusión - SVM (Mejores Hiperparámetros)')
plt.show()
# --- COMPARACIÓN FINAL DE MODELOS OPTIMIZADOS ---
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
resultados = [
    {
        'Modelo': 'Random Forest (Campeón)',
        'Recall': recall_score(y_test, y_pred_best),
        'Precisión': precision_score(y_test, y_pred_best),
        'F1-Score': f1_score(y_test, y_pred_best),
        'Accuracy': accuracy_score(y_test, y_pred_best)
    },
    {
        'Modelo': 'Regresión Logística',
        'Recall': recall_score(y_test, y_pred_lr_best),
        'Precisión': precision_score(y_test, y_pred_lr_best),
        'F1-Score': f1_score(y_test, y_pred_lr_best),
        'Accuracy': accuracy_score(y_test, y_pred_lr_best)
    },
    {
        'Modelo': 'SVM',
        'Recall': recall_score(y_test, y_pred_svm_best),
        'Precisión': precision_score(y_test, y_pred_svm_best),
        'F1-Score': f1_score(y_test, y_pred_svm_best),
        'Accuracy': accuracy_score(y_test, y_pred_svm_best)
    }
]
# Creamos el DataFrame y lo ordenamos por F1-Score
df_comparacion = pd.DataFrame(resultados).sort_values(by='F1-Score', ascending=False)
# Redondear los números a 3 decimales
df_comparacion = df_comparacion.round(3)
print("\n=== RESUMEN DE COMPARACIÓN DE MODELOS OPTIMIZADOS ===")
display(df_comparacion)
print("CÁLCULO DE INCERTIDUMBRE E INTERVALOS DE CONFIANZA (95%)")
# Configurar el Bootstrapping
n_bootstraps = 1000  # Vamos a simular 1000 conjuntos de prueba diferentes
rng_seed = 42
rng = np.random.RandomState(rng_seed)
# Listas para guardar los resultados de las 1000 simulaciones
bootstrapped_recalls = []
bootstrapped_precisions = []
bootstrapped_f1s = []
# Convertimos a arrays de numpy para que sea más fácil de procesar
y_test_np = y_test.values
y_pred_np = y_pred_best
# Ejecutar las 1000 simulaciones
for i in range(n_bootstraps):
    # Crear una muestra aleatoria con reemplazo del mismo tamaño que el test original
    indices = rng.randint(0, len(y_pred_np), len(y_pred_np))
    # Si por extrema casualidad la muestra no tiene fallas, la saltamos
    if len(np.unique(y_test_np[indices])) < 2:
        continue
    # Calculamos las métricas para esta muestra específica 
    recall = recall_score(y_test_np[indices], y_pred_np[indices], pos_label=1)
    precision = precision_score(y_test_np[indices], y_pred_np[indices], pos_label=1, zero_division=0)
    f1 = f1_score(y_test_np[indices], y_pred_np[indices], pos_label=1)
    bootstrapped_recalls.append(recall)
    bootstrapped_precisions.append(precision)
    bootstrapped_f1s.append(f1)
# Calcular los percentiles para obtener el 95% central
alpha = 0.95
p_lower = ((1.0 - alpha) / 2.0) * 100
p_upper = (alpha + ((1.0 - alpha) / 2.0)) * 100
print("\n--- RESULTADOS ESTADÍSTICOS PARA LA CLASE 1 (FALLAS) ---")
print(f"Recall:    {np.mean(bootstrapped_recalls):.3f} (Intervalo 95%: [{np.percentile(bootstrapped_recalls, p_lower):.3f}, {np.percentile(bootstrapped_recalls, p_upper):.3f}])")
print(f"Precision: {np.mean(bootstrapped_precisions):.3f} (Intervalo 95%: [{np.percentile(bootstrapped_precisions, p_lower):.3f}, {np.percentile(bootstrapped_precisions, p_upper):.3f}])")
print(f"F1-Score:  {np.mean(bootstrapped_f1s):.3f} (Intervalo 95%: [{np.percentile(bootstrapped_f1s, p_lower):.3f}, {np.percentile(bootstrapped_f1s, p_upper):.3f}])")