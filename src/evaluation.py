import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import recall_score, precision_score, f1_score, classification_report, confusion_matrix

def evaluar_y_graficar(y_test, y_pred, titulo, cmap='Blues'):
    """Muestra el reporte técnico y despliega la matriz de confusión visual."""
    print(f"\n--- REPORTE DE CLASIFICACIÓN: {titulo} ---")
    print(classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, 
                xticklabels=['Predicción OK', 'Predicción Falla'], 
                yticklabels=['Real OK', 'Real Falla'])
    plt.title(f'Matriz de Confusión - {titulo}')
    plt.show()

def mostrar_comparacion_final(y_test, y_pred_rf, y_pred_lr, y_pred_svm):
    """Genera e imprime un DataFrame comparativo consolidado."""
    resultados = [
        {'Modelo': 'Random Forest (Campeón)', 'Recall': recall_score(y_test, y_pred_rf), 'F1-Score': f1_score(y_test, y_pred_rf)},
        {'Modelo': 'Regresión Logística', 'Recall': recall_score(y_test, y_pred_lr), 'F1-Score': f1_score(y_test, y_pred_lr)},
        {'Modelo': 'SVM', 'Recall': recall_score(y_test, y_pred_svm), 'F1-Score': f1_score(y_test, y_pred_svm)}
    ]
    df_comparacion = pd.DataFrame(resultados).sort_values(by='F1-Score', ascending=False).round(3)
    print("\n=== RESUMEN DE COMPARACIÓN DE MODELOS OPTIMIZADOS ===")
    print(df_comparacion.to_string(index=False))

def calcular_intervalos_confianza(y_test, y_pred_best, n_bootstraps=1000):
    """Evalúa la robustez del modelo final mediante remuestreo Bootstrapping."""
    rng = np.random.RandomState(42)
    bootstrapped_recalls, bootstrapped_precisions, bootstrapped_f1s = [], [], []
    y_test_np = y_test.values

    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_pred_best), len(y_pred_best))
        if len(np.unique(y_test_np[indices])) < 2:
            continue
            
        bootstrapped_recalls.append(recall_score(y_test_np[indices], y_pred_best[indices], pos_label=1))
        bootstrapped_precisions.append(precision_score(y_test_np[indices], y_pred_best[indices], pos_label=1, zero_division=0))
        bootstrapped_f1s.append(f1_score(y_test_np[indices], y_pred_best[indices], pos_label=1))

    alpha = 0.95
    p_lower = ((1.0 - alpha) / 2.0) * 100
    p_upper = (alpha + ((1.0 - alpha) / 2.0)) * 100

    print("\n=== CÁLCULO DE INCERTIDUMBRE E INTERVALOS DE CONFIANZA (95%) ===")
    print(f"Recall:    {np.mean(bootstrapped_recalls):.3f} (Intervalo 95%: [{np.percentile(bootstrapped_recalls, p_lower):.3f}, {np.percentile(bootstrapped_recalls, p_upper):.3f}])")
    print(f"Precision: {np.mean(bootstrapped_precisions):.3f} (Intervalo 95%: [{np.percentile(bootstrapped_precisions, p_lower):.3f}, {np.percentile(bootstrapped_precisions, p_upper):.3f}])")
    print(f"F1-Score:  {np.mean(bootstrapped_f1s):.3f} (Intervalo 95%: [{np.percentile(bootstrapped_f1s, p_lower):.3f}, {np.percentile(bootstrapped_f1s, p_upper):.3f}])")