import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def cargar_y_preparar_datos(ruta_csv):
    """Carga el dataset, limpia variables irrelevantes, genera dummies y escala los datos."""
    df = pd.read_csv(ruta_csv)
    df_clean = df.drop(columns=['UDI', 'Product ID'])
    df_clean = pd.get_dummies(df_clean, columns=['Type'], drop_first=True)

    # Filtrar variables para evitar Data Leakage y quitar ruido ('Type')
    columnas_a_eliminar = ['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF', 'Type_L', 'Type_M']
    X_opt = df_clean.drop(columns=columnas_a_eliminar)
    y = df_clean['Machine failure']

    # División estratificada
    X_train, X_test, y_train, y_test = train_test_split(
        X_opt, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Escalado estándar
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test