# Importaciones locales de nuestros módulos personalizados
from data_prep import cargar_y_preparar_datos
from models import entrenar_pipeline_modelos
from evaluation import evaluar_y_graficar, calcular_intervalos_confianza, mostrar_comparacion_final

def main():
    # 1. Pipeline de Preparación de Datos
    print("Iniciando la preparación de datos...")
    X_train, X_test, y_train, y_test = cargar_y_preparar_datos('data/ai4i2020.csv')

    # 2. Pipeline de Optimización y Entrenamiento
    best_rf, best_lr, best_svm = entrenar_pipeline_modelos(X_train, y_train)

    # 3. Pipeline de Evaluación Individual
    y_pred_rf = best_rf.predict(X_test)
    evaluar_y_graficar(y_test, y_pred_rf, "Random Forest (Campeón)", cmap='Oranges')

    y_pred_lr = best_lr.predict(X_test)
    evaluar_y_graficar(y_test, y_pred_lr, "Regresión Logística", cmap='Purples')

    y_pred_svm = best_svm.predict(X_test)
    evaluar_y_graficar(y_test, y_pred_svm, "SVM", cmap='Reds')

    # 4. Análisis Estadístico y Comparativo Consolidado
    mostrar_comparacion_final(y_test, y_pred_rf, y_pred_lr, y_pred_svm)
    calcular_intervalos_confianza(y_test, y_pred_rf)

if __name__ == "__main__":
    main()