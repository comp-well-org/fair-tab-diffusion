from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

def default_sk_clf(choice: str, seed=42):
    max_iter = 1000
    n_neighbors = 10
    max_depth = 28
    classic_clfs = {
        'Logistic Regression': LogisticRegression(max_iter=max_iter),
        'Decision Tree': DecisionTreeClassifier(max_depth=max_depth, random_state=seed),
        'Random Forest': RandomForestClassifier(max_depth=max_depth, random_state=seed),
        'AdaBoost': AdaBoostClassifier(),
        'XGBoost': XGBClassifier(random_state=seed),
        'CatBoost': CatBoostClassifier(allow_writing_files=False, random_seed=seed, silent=True),
        'SVM': SVC(random_state=seed),
        'KNN': KNeighborsClassifier(n_neighbors=n_neighbors),
        'MLP': MLPClassifier(alpha=1, max_iter=max_iter, random_state=seed),
    }
    return classic_clfs[choice]
