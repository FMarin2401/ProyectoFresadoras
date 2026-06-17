from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

def optimizar_modelo(estimator, param_grid, X_train, y_train):
    """Ejecuta GridSearchCV enfocado en maximizar el Recall."""
    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=5,
        scoring='recall',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    print(f"-> Mejores parámetros encontrados: {grid_search.best_params_}")
    return grid_search.best_estimator_

def entrenar_pipeline_modelos(X_train, y_train):
    """Orquesta la optimización secuencial de los tres modelos evaluados."""
    # 1. Random Forest
    print("\nOptimizando Random Forest...")
    param_grid_rf = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'class_weight': ['balanced']
    }
    best_rf = optimizar_modelo(RandomForestClassifier(random_state=42), param_grid_rf, X_train, y_train)
    
    # 2. Regresión Logística
    print("\nOptimizando Regresión Logística...")
    param_grid_lr = {'C': [0.01, 0.1, 1, 10, 100], 'solver': ['liblinear', 'lbfgs']}
    best_lr = optimizar_modelo(LogisticRegression(random_state=42, class_weight='balanced', max_iter=2000), param_grid_lr, X_train, y_train)
    
    # 3. SVM
    print("\nOptimizando SVM...")
    param_grid_svm = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto']}
    best_svm = optimizar_modelo(SVC(random_state=42, class_weight='balanced'), param_grid_svm, X_train, y_train)
    
    return best_rf, best_lr, best_svm