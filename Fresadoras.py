import pandas as pd

# Primero subimos el archivo 
archivo = 'ai4i2020.csv' 
df = pd.read_csv(archivo)

# Y tomamos una muestra de los datos para verlos
print("Vista previa de los datos")
display(df.head())

# Imprimimos la ficha tecnica del dataset
print("\nInformación del dataset")
print(f"Fuente: UCI Machine Learning Repository")
print(f"Tamaño (Filas, Columnas): {df.shape}")
print("\nTipos de Variables:")
print(df.dtypes)

# Verificamos el balance de clases
conteo = df['Machine failure'].value_counts()
print("\nDistribución de fallas")
print(f"Máquinas OK (0): {conteo[0]}")
print(f"Máquinas con Falla (1): {conteo[1]}")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

######### 1.  Eliminar variables que no aportan (Identificadores)
df_clean = df.drop(columns=['UDI', 'Product ID'])

######### 2. Codificar variables categóricas (Type)
df_clean = pd.get_dummies(df_clean, columns=['Type'], drop_first=True)

######### 3. Separar variables independientes (X) y dependientes (y)
X = df_clean.drop(columns=['Machine failure'])
y = df_clean['Machine failure']

######### 4. División de datos (Estratificada por el desbalance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

######### 5. Estandarización (CRÍTICO para SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) # Fit SOLO en train
X_test_scaled = scaler.transform(X_test)       # Solo transform en test

######## 6. Implementar Soft SVM con Kernel RBF
# C=1.0: Es el parámetro de regularización (Margen vs Errores permitidos)
# class_weight='balanced': Obliga al modelo a darle más importancia a las máquinas 
# en el cual se presentan las fallas (1)

svm_model = SVC(kernel='rbf', C=1.0, class_weight='balanced', random_state=42)

####### 8. Entrenar el modelo
svm_model.fit(X_train_scaled, y_train)

######## 9. Realizar predicciones
y_pred = svm_model.predict(X_test_scaled)

######## 10. Evaluar el modelo (NO usar solo Accuracy)
print("REPORTE DE CLASIFICACIÓN")
print(classification_report(y_test, y_pred))

######## 11. Graficar la Matriz de Confusión para el documento
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['OK (0)', 'Falla (1)'], 
            yticklabels=['OK (0)', 'Falla (1)'])

plt.xlabel('Predicción del SVM')
plt.ylabel('Valor Real')
plt.title('Matriz de Confusión - Soft SVM (RBF)')
plt.show()