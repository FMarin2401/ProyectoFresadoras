# Librerías generales de datos y visualización
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Librerías de Machine Learning (Scikit-Learn)
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


######### 1. Definir la cuadrícula de parámetros a probar
# Vamos a probar diferentes cantidades de árboles y profundidades
param_grid = {
    'n_estimators': [50, 100, 200],         # Cantidad de árboles en el bosque
    'max_depth': [None, 10, 20],            # Profundidad máxima de cada árbol
    'min_samples_split': [2, 5, 10],        # Mínimo de muestras para dividir un nodo
    'class_weight': ['balanced']            # Siempre mantenemos esto para clases desbalanceadas
}

######### 2. Inicializar el modelo base y el GridSearchCV
# scoring='recall' le dice a la herramienta: "Encuentra la configuración que detecte MÁS fallas"
rf_base = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(
    estimator=rf_base, 
    param_grid=param_grid, 
    cv=5,                 # Validación cruzada de 5 pliegues (para asegurar que no sea suerte)
    scoring='recall',     # ¡Nuestra métrica estrella!
    n_jobs=-1             # Usa todos los núcleos de tu procesador para ir más rápido
)

######### 3. Ejecutar la búsqueda con nuestros datos optimizados y escalados
grid_search.fit(X_train_opt_scaled, y_train)

######### 4. Ver los resultados ganadores
print(f"Mejores parámetros encontrados:\n {grid_search.best_params_}")

######### 5. Evaluar el modelo ganador
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


# --- REGRESIÓN LOGÍSTICA ---
# Usamos max_iter=1000 por si el algoritmo necesita más tiempo para encontrar la línea perfecta
lr_model = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
lr_model.fit(X_train_opt_scaled, y_train)

y_pred_lr = lr_model.predict(X_test_opt_scaled)

print("\n--- REPORTE DE CLASIFICACIÓN (REGRESIÓN LOGÍSTICA) ---")
print(classification_report(y_test, y_pred_lr))

cm_lr = confusion_matrix(y_test, y_pred_lr)
plt.figure(figsize=(6,4))
# Usamos morado (Purples) para distinguir este modelo
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Purples', 
            xticklabels=['Predicción OK', 'Predicción Falla'], 
            yticklabels=['Real OK', 'Real Falla'])
plt.title('Matriz de Confusión - Regresión Logística')
plt.show()


# --- SUPPORT VECTOR MACHINE (SVM) ---
# SVM suele tomar un poquito más de tiempo en entrenar que los demás
svm_model = SVC(random_state=42, class_weight='balanced')
svm_model.fit(X_train_opt_scaled, y_train)

y_pred_svm = svm_model.predict(X_test_opt_scaled)

print("\n--- REPORTE DE CLASIFICACIÓN (SVM) ---")
print(classification_report(y_test, y_pred_svm))

cm_svm = confusion_matrix(y_test, y_pred_svm)
plt.figure(figsize=(6,4))
# Usamos rojo (Reds) para distinguir este modelo
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Reds', 
            xticklabels=['Predicción OK', 'Predicción Falla'], 
            yticklabels=['Real OK', 'Real Falla'])
plt.title('Matriz de Confusión - Support Vector Machine (SVM)')
plt.show()