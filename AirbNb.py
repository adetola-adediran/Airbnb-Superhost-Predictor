import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc
from sklearn.feature_selection import SelectKBest, f_classif

def load_and_split_data(filepath):
    """Loads preprocessed Airbnb data and splits into training/testing sets."""
    df = pd.read_csv(filepath, header=0)
    
    # Isolate the target variable for the binary classification problem
    y = df['host_is_superhost']
    X = df.drop(columns='host_is_superhost', axis=1)
    
    # Retain a 10% holdout set for final model evaluation
    return train_test_split(X, y, test_size=0.10, random_state=1234)

def optimize_and_train_model(X_train, y_train):
    """Utilizes GridSearchCV to find the optimal regularization hyperparameter."""
    print("Running GridSearchCV for hyperparameter tuning...")
    base_model = LogisticRegression(max_iter=1000)
    
    # Define a logarithmic parameter grid for regularization strength (C)
    param_grid = {'C': [10**i for i in range(-5, 5)]}
    
    grid = GridSearchCV(base_model, param_grid, cv=5)
    grid_search = grid.fit(X_train, y_train)
    best_c = grid_search.best_params_['C']
    
    print(f"Optimal C parameter identified: {best_c}")
    
    # Train and return the final model using the optimized hyperparameter
    final_model = LogisticRegression(C=best_c, max_iter=1000)
    final_model.fit(X_train, y_train)
    return final_model

def perform_feature_selection(X, y):
    """Identifies the top 5 predictive features to analyze dimensionality impact."""
    selector = SelectKBest(f_classif, k=5)
    selector.fit(X, y)
    top_5_features = X.columns[selector.get_support()]
    print(f"\nTop 5 Predictive Features: {list(top_5_features)}")
    return top_5_features

def main():
    # 1. Data Pipeline
    filepath = os.path.join(os.getcwd(), "data_LR", "airbnbData_train.csv")
    X_train, X_test, y_train, y_test = load_and_split_data(filepath)
    
    # 2. Model Training & Optimization
    best_model = optimize_and_train_model(X_train, y_train)
    
    # 3. Model Evaluation (Extracting probabilities for AUC/ROC)
    proba_predictions = [prob[1] for prob in best_model.predict_proba(X_test)]
    fpr, tpr, thresholds = roc_curve(y_test, proba_predictions)
    model_auc = auc(fpr, tpr)
    print(f"Optimized Model AUC: {model_auc:.4f}")
    
    # 4. Feature Selection Analysis
    perform_feature_selection(pd.concat([X_train, X_test]), pd.concat([y_train, y_test]))
    
    # 5. Model Serialization
    pkl_filename = "LR_Model.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(best_model, file)
    print(f"\nModel successfully serialized and saved as {pkl_filename}")

if __name__ == "__main__":
    main()