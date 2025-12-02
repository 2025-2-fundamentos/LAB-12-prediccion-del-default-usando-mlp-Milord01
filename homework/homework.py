# flake8: noqa: E501
#

# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando componentes principales.
#   El pca usa todas las componentes.
# - Escala la matriz de entrada al intervalo [0, 1].
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una red neuronal tipo MLP.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

import json
import gzip
import pickle
import os
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    precision_score,
    balanced_accuracy_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def load_compressed_datasets():
    train_dataset = pd.read_csv(
        "./files/input/train_data.csv.zip",
        index_col=False,
        compression="zip",
    )
    test_dataset = pd.read_csv(
        "./files/input/test_data.csv.zip",
        index_col=False,
        compression="zip",
    )
    return train_dataset, test_dataset


def preprocess_dataframe(dataframe):
    df = dataframe.copy()
    df = df.rename(columns={"default payment next month": "default"})
    df = df.drop(columns=["ID"])
    df = df.loc[df["MARRIAGE"] != 0]
    df = df.loc[df["EDUCATION"] != 0]
    df["EDUCATION"] = df["EDUCATION"].apply(lambda x: 4 if x >= 4 else x)
    df = df.dropna()
    return df


def separate_features_labels(dataframe):
    features = dataframe.drop(columns=["default"])
    labels = dataframe["default"]
    return features, labels


def build_neural_pipeline(feature_matrix):
    cat_cols = ["SEX", "EDUCATION", "MARRIAGE"]
    num_cols = [col for col in feature_matrix.columns if col not in cat_cols]
    
    transformer = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(), cat_cols),
            ("scaler", StandardScaler(), num_cols),
        ]
    )
    
    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", transformer),
            ("feature_selection", SelectKBest(score_func=f_classif)),
            ("pca", PCA()),
            ("classifier", MLPClassifier(max_iter=500, random_state=21)),
        ]
    )
    
    return model_pipeline


def train_with_grid_search(pipeline):
    hyperparameter_grid = {
        "pca__n_components": [None],
        "feature_selection__k": [20],
        "classifier__hidden_layer_sizes": [(50, 30, 40, 60)],
        "classifier__alpha": [0.26],
        "classifier__learning_rate_init": [0.001],
    }
    
    grid_cv = GridSearchCV(
        estimator=pipeline,
        param_grid=hyperparameter_grid,
        cv=5,
        scoring="balanced_accuracy",
        n_jobs=-1,
        refit=True,
    )
    
    return grid_cv


def save_compressed_model(trained_model, output_path):
    output_dir = os.path.dirname(output_path)
    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, file))
    else:
        os.makedirs(output_dir)
    
    with gzip.open(output_path, "wb") as file:
        pickle.dump(trained_model, file)


def compute_classification_metrics(split_name, true_labels, predicted_labels):
    return {
        "type": "metrics",
        "dataset": split_name,
        "precision": precision_score(true_labels, predicted_labels, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(true_labels, predicted_labels),
        "recall": recall_score(true_labels, predicted_labels, zero_division=0),
        "f1_score": f1_score(true_labels, predicted_labels, zero_division=0),
    }


def compute_confusion_matrix_dict(split_name, true_labels, predicted_labels):
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    return {
        "type": "cm_matrix",
        "dataset": split_name,
        "true_0": {"predicted_0": int(conf_matrix[0][0]), "predicted_1": int(conf_matrix[0][1])},
        "true_1": {"predicted_0": int(conf_matrix[1][0]), "predicted_1": int(conf_matrix[1][1])},
    }


def save_all_metrics(metrics_list, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as file:
        for metric in metrics_list:
            file.write(json.dumps(metric) + "\n")


def main():
    train_df, test_df = load_compressed_datasets()
    
    train_df = preprocess_dataframe(train_df)
    test_df = preprocess_dataframe(test_df)
    
    X_train, y_train = separate_features_labels(train_df)
    X_test, y_test = separate_features_labels(test_df)
    
    neural_pipeline = build_neural_pipeline(X_train)
    
    trained_model = train_with_grid_search(neural_pipeline)
    trained_model.fit(X_train, y_train)
    
    save_compressed_model(trained_model, "files/models/model.pkl.gz")
    
    y_train_pred = trained_model.predict(X_train)
    y_test_pred = trained_model.predict(X_test)
    
    train_metrics = compute_classification_metrics("train", y_train, y_train_pred)
    test_metrics = compute_classification_metrics("test", y_test, y_test_pred)
    
    train_cm = compute_confusion_matrix_dict("train", y_train, y_train_pred)
    test_cm = compute_confusion_matrix_dict("test", y_test, y_test_pred)
    
    all_metrics = [train_metrics, test_metrics, train_cm, test_cm]
    save_all_metrics(all_metrics, "files/output/metrics.json")


if __name__ == "__main__":
    main()

