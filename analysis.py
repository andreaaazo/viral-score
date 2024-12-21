import os
import re
import joblib
import optuna
import shap
import warnings
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler

from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import lightgbm as lgb

from functools import partial

warnings.filterwarnings("always")


def safe_division(numerator, denominator):
    return np.where(denominator > 0, numerator / denominator, 0)


def load_dataset(path):
    data = pd.read_json(path, orient="records", convert_dates=False)
    data.fillna(0, inplace=True)
    return data


def add_features(df):
    today_unix = pd.Timestamp.now().timestamp()
    df["days_since_post"] = (today_unix - df["posted_time"]) / (60 * 60 * 24)
    df["days_since_post"] = np.maximum(df["days_since_post"], 1)

    df["engagement_normalized"] = df["engagement_rate"] / (
        1 + np.log1p(df["days_since_post"])
    )
    df["likes_per_second"] = safe_division(df["likes"], df["video_duration"])
    df["comments_per_second"] = safe_division(df["comments"], df["video_duration"])

    dim_df = pd.json_normalize(df["dimensions"])
    dim_df.fillna(1, inplace=True)
    if "width" not in dim_df.columns:
        dim_df["width"] = 1
    if "height" not in dim_df.columns:
        dim_df["height"] = 1

    df["aspect_ratio"] = dim_df["width"] / dim_df["height"]
    df["views_to_followers"] = safe_division(df["views"], df["followers"])
    df["comment_to_like"] = safe_division(df["comments"], df["likes"])
    df["like_to_view"] = safe_division(df["likes"], df["views"])
    df["time_penalty"] = 1 / np.log1p(df["days_since_post"])
    df["share_of_engagement"] = safe_division(
        df["likes"] + df["comments"], df["views"] + df["followers"]
    )
    df["view_retention_rate"] = safe_division(df["views"], df["video_duration"])
    df["average_engagement_time"] = (
        df["likes_per_second"] + df["comments_per_second"]
    ) / 2

    return df


def get_feature_columns():
    return [
        "views_to_followers",
        "comment_to_like",
        "like_to_view",
        "time_penalty",
        "video_duration",
        "likes_per_second",
        "comments_per_second",
        "aspect_ratio",
        "share_of_engagement",
        "view_retention_rate",
        "average_engagement_time",
    ]


def get_base_estimators():
    return [
        ("gb", GradientBoostingRegressor(random_state=42)),
        (
            "xgb",
            XGBRegressor(
                objective="reg:squarederror",
                random_state=42,
                tree_method="hist",
                n_jobs=8,
            ),
        ),
        ("cat", CatBoostRegressor(random_state=42, verbose=0, thread_count=8)),
        (
            "lgb",
            lgb.LGBMRegressor(
                boosting_type="gbdt",
                objective="regression",
                metric="r2",
                random_state=42,
                force_col_wise=True,
                verbose=-1,
                n_jobs=8,
            ),
        ),
    ]


def create_model(model_type, trial):
    if model_type == "gradient_boosting":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "random_state": 42,
        }
        return GradientBoostingRegressor(**params)

    if model_type == "xgboost":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "tree_method": "hist",
            "random_state": 42,
            "n_jobs": 8,
        }
        return XGBRegressor(**params, objective="reg:squarederror")

    if model_type == "catboost":
        params = {
            "iterations": trial.suggest_int("iterations", 100, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "depth": trial.suggest_int("depth", 3, 10),
            "random_seed": 42,
            "verbose": 0,
            "thread_count": 8,
        }
        return CatBoostRegressor(**params)

    if model_type == "lightgbm":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "random_state": 42,
            "force_col_wise": True,
            "verbose": -1,
            "n_jobs": 8,
        }
        return lgb.LGBMRegressor(**params)

    # Stacking model
    base_estimators = get_base_estimators()
    final_params = {
        "n_estimators": trial.suggest_int("final_n_estimators", 100, 500),
        "learning_rate": trial.suggest_float("final_learning_rate", 0.01, 0.2),
        "max_depth": trial.suggest_int("final_max_depth", 3, 10),
        "random_state": 42,
    }
    final_estimator = GradientBoostingRegressor(**final_params)
    return StackingRegressor(
        estimators=base_estimators, final_estimator=final_estimator, n_jobs=8
    )


def reconstruct_best_model(best_params, best_model_type):
    if best_model_type == "gradient_boosting":
        return GradientBoostingRegressor(**best_params)

    if best_model_type == "xgboost":
        return XGBRegressor(**best_params, objective="reg:squarederror")

    if best_model_type == "catboost":
        return CatBoostRegressor(**best_params)

    if best_model_type == "lightgbm":
        return lgb.LGBMRegressor(**best_params)

    # Stacking model
    base_estimators = get_base_estimators()
    final_params = {
        k.replace("final_", ""): v
        for k, v in best_params.items()
        if k.startswith("final_")
    }
    final_estimator = GradientBoostingRegressor(**final_params)
    return StackingRegressor(
        estimators=base_estimators, final_estimator=final_estimator, n_jobs=8
    )


def interpret_and_save(model, X_data, df, feature_cols):
    df["viral_score"] = model.predict(X_data)
    explainer = shap.TreeExplainer(model)
    sample_size = min(500, X_data.shape[0])
    X_sample = X_data[:sample_size]
    shap_values = explainer.shap_values(X_sample)
    shap.summary_plot(shap_values, X_sample, feature_names=feature_cols)

    top_reels = df.sort_values(by="viral_score", ascending=False)
    top_reels.to_json("top_reels.json", orient="records", indent=4)
    joblib.dump(model, "best_model.pkl")


def create_optuna_study(
    studies_dir="studies", base_name="optuna_study", direction="maximize"
):
    os.makedirs(studies_dir, exist_ok=True)
    next_number = get_next_study_number(studies_dir, base_name)
    db_name = f"{base_name}_{next_number}.db"
    db_path = os.path.join(studies_dir, db_name)
    storage_uri = f"sqlite:///{os.path.abspath(db_path)}"
    study = optuna.create_study(
        direction=direction,
        study_name=f"{base_name}_{next_number}",
        storage=storage_uri,
        load_if_exists=False,
    )
    return study


def get_next_study_number(studies_dir, base_name):
    existing_numbers = []
    pattern = re.compile(rf"{re.escape(base_name)}_(\d+)\.db$")

    if not os.path.exists(studies_dir):
        return 1

    for file in os.listdir(studies_dir):
        match = pattern.match(file)
        if match:
            existing_numbers.append(int(match.group(1)))

    if existing_numbers:
        return max(existing_numbers) + 1
    else:
        return 1


def run_optimization(
    X_data, y_data, n_trials=100, existing_study_name=None, studies_dir="studies"
):
    def objective(trial, X_data, y_data):
        model_type = trial.suggest_categorical(
            "model",
            ["gradient_boosting", "xgboost", "catboost", "lightgbm", "stacking"],
        )
        model = create_model(model_type, trial)
        score = cross_val_score(
            model, X_data, y_data, cv=5, scoring="r2", n_jobs=8
        ).mean()
        return score

    if existing_study_name is not None:
        # Carico lo study esistente
        db_path = os.path.join(studies_dir, f"{existing_study_name}.db")
        if not os.path.exists(db_path):
            raise FileNotFoundError(
                f"Lo study {existing_study_name} non esiste nella cartella {studies_dir}."
            )
        storage_uri = f"sqlite:///{os.path.abspath(db_path)}"
        study = optuna.load_study(study_name=existing_study_name, storage=storage_uri)
    else:
        # Creo un nuovo study
        study = create_optuna_study()

    # Esegue l'ottimizzazione solo se non stiamo usando uno studio già ottimizzato
    if existing_study_name is None:
        study.optimize(
            partial(objective, X_data=X_data, y_data=y_data),
            n_trials=n_trials,
            n_jobs=2,
        )

    best_params = study.best_params
    best_model_type = best_params.pop("model")
    return best_params, best_model_type


def main():
    # Da qui, si può scegliere se specificare il nome di uno studio esistente o meno
    # Esempio: existing_study_name = "optuna_study_1" per caricare quello studio
    existing_study_name = (
        None  # Impostare il nome dello studio esistente, es. "optuna_study_2"
    )

    print(">> Caricamento e preparazione dei dati...")
    dataset = load_dataset("data/cars_profiles.json")
    dataset = add_features(dataset)
    feature_columns = get_feature_columns()

    X = dataset[feature_columns]
    y = dataset["engagement_normalized"]

    print(">> Normalizzazione delle feature...")
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X)

    if existing_study_name is not None:
        print(f">> Caricamento dello study Optuna esistente: {existing_study_name}")
    else:
        print(">> Inizio ottimizzazione iperparametri con Optuna (incluso Stacking)...")

    best_params, best_model_type = run_optimization(
        X_normalized, y, n_trials=100, existing_study_name=existing_study_name
    )

    print(">> Migliori iperparametri trovati:", best_params)
    print(">> Ricostruzione del modello migliore in assoluto...")
    best_model = reconstruct_best_model(best_params, best_model_type)

    print(">> Addestramento del modello migliore sull'intero dataset...")
    best_model.fit(X_normalized, y)

    print(">> Interpretazione con SHAP sul modello migliore...")
    interpret_and_save(best_model, X_normalized, dataset, feature_columns)

    print(
        ">> Processo completato con successo! Il miglior modello è di tipo:",
        best_model_type,
    )


if __name__ == "__main__":
    main()
