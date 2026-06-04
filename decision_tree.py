import pandas as pd
import numpy as np
import os
import sys
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from joblib import dump


def fetch_data():
    path = "/data/mlproject22" if os.path.exists("/data/mlproject22") else "."
    train_data = pd.read_csv(os.path.join(path, "transactions.csv.zip"))
    X_train = train_data.drop(columns = "Class")
    y_train = train_data["Class"]
    return X_train, y_train

def over_sample(X_train, y_train, ratio):
    np.random.seed(100)
    
    X_tr = X_train.values
    y_tr = y_train.values

    # 2) find indices of each class
    maj_idx = np.where(y_tr == 0)[0]
    min_idx = np.where(y_tr == 1)[0]

    # 3) how many to upsample? e.g. match majority
    n_maj = len(maj_idx)
    n_samples = int(n_maj * ratio)

    # 4) sample with replacement from the minority
    upsampled_min_idx = np.random.choice(min_idx, size=n_samples, replace=True)

    # 5) combine majority + upsampled minority
    new_idx = np.concatenate([maj_idx, upsampled_min_idx])
    np.random.shuffle(new_idx)

    X_arr = X_tr[new_idx]
    y_arr = y_tr[new_idx]
    
    X_bal = pd.DataFrame(X_arr, columns=X_train.columns, index=None)
    y_bal = pd.Series(y_arr, name=y_train.name)
    return X_bal, y_bal


def plot_results(y_test, y_pred):
    # 1) Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # 2) Prepare display matrix and labels
    display_matrix = np.array([[tn, fp],
                           [fn, tp]])
    cell_labels = np.array([['TN', 'FP'],
                        ['FN', 'TP']])

    # 3) Plot
    fig, ax = plt.subplots()
    im = ax.imshow(display_matrix, interpolation='nearest', cmap='Blues')

    # Colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Count', rotation=-90, va="bottom")

    # Tick labels
    classes = ['Non-Fraud (0)', 'Fraud (1)']
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=classes, yticklabels=classes,
           ylabel='True label',
           xlabel='Predicted label',
           title='Confusion Matrix Heatmap')

    # Loop over data dimensions and create text annotations.
    thresh = display_matrix.max() / 2.
    for i in range(display_matrix.shape[0]):
        for j in range(display_matrix.shape[1]):
            ax.text(j, i,
                    f"{cell_labels[i, j]}\n{display_matrix[i, j]}",
                    ha="center", va="center",
                    color="white" if display_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.show()
    
def train_model_dt():

    # fetch data from table
    X, y = fetch_data()

    #split data into test and validation set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=100
    )
    
    # define pipe where scaler and model is defined
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("dt", DecisionTreeClassifier(
            random_state=100,
            min_samples_leaf=10,
            min_samples_split=2,
            criterion='gini'
            ))
    ])
    
    oversample_factor = 0.01
    
    X_oversampled, y_oversampled = over_sample(X_train, y_train, oversample_factor)
    
    # train the model on the test set
    pipe.fit(X_oversampled, y_oversampled)
    
    # print report to console
    y_pred = pipe.predict(X_test)
    print(classification_report(y_test, y_pred, digits=4))
    plot_results(y_test, y_pred)
    
    y_probs = pipe.predict_proba(X_test)[:,1]
    
    print(roc_auc_score(y_true=y_test, y_score=y_probs))
    
    # save the model
    dump(pipe, "dt_pipe.joblib")
    
def find_best_hyperparameters():
    X, y = fetch_data()
    
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("dt", DecisionTreeClassifier(
            random_state=100
            ))
    ])
    
    oversample_factor = 0.01
    
    X_oversampled, y_oversampled = over_sample(X, y, oversample_factor)
    
    param_grid = {
        # "dt__max_depth":        [None, 3, 5, 10, 20],
        "dt__criterion":['gini','entropy','log_loss'],
        "dt__min_samples_split":[2, 3, 4],
        "dt__min_samples_leaf": [7, 10, 15],
        # "dt__min_samples_split":[2, 5, 10],
        # "dt__min_samples_leaf": [1, 5, 10],
        # "dt__max_features":     [None, "sqrt", "log2"],
    }
    
    grid = GridSearchCV(
        pipe,
        param_grid,
        scoring="recall",
        cv=5,
        n_jobs=-1,
        verbose=3,
        
    )
    grid.fit(X_oversampled, y_oversampled)
    print("Best params:", grid.best_params_)
    print("Best Recall:",    grid.best_score_)


train_model_dt()
# find_best_hyperparameters()
