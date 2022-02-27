import click
import mlflow
import numpy as np
import optuna
import sklearn.datasets
import sklearn.ensemble
import sklearn.metrics
import sklearn.model_selection
import sklearn.svm
from optuna.integration.mlflow import MLflowCallback
from sklearn.base import BaseEstimator, ClassifierMixin


class Classifier(BaseEstimator, ClassifierMixin):
    def __init__(self, params):
        self.params = params
        if params["classifier_name"] == "SVC":
            self.classifier = sklearn.svm.SVC(**params["classifier_params"])
        else:
            self.classifier = sklearn.ensemble.RandomForestClassifier(
                **params["classifier_params"]
            )

    def fit(self, x, y):
        return self.classifier.fit(x, y)

    def predict(self, x):
        return self.classifier.predict(x)


class Utils:
    @staticmethod
    def generate_params(trial):
        params = {}
        classifier_params = {}

        params["classifier_name"] = trial.suggest_categorical(
            "classifier", ["SVC", "RandomForest"]
        )

        if params["classifier_name"] == "SVC":
            classifier_params["C"] = trial.suggest_float(
                "svc_c", 1e-10, 1e10, log=True
            )
            classifier_params["gamma"] = "auto"

        else:
            classifier_params["max_depth"] = trial.suggest_int(
                "rf_max_depth", 2, 32, log=True
            )
            classifier_params["n_estimators"] = trial.suggest_int(
                "rf_n_estimators", 2, 100, log=False
            )

        params["classifier_params"] = classifier_params
        return params


class Objective:
    def __init__(self, x, y, cv_folds=5):
        self.x = x
        self.y = y
        self.cv_folds = cv_folds

    def __call__(self, trial):

        x = self.x
        y = self.y
        params = Utils.generate_params(trial)
        classifier = Classifier(params)
        scores = sklearn.model_selection.cross_validate(
            classifier,
            x,
            y,
            cv=self.cv_folds,
            n_jobs=-1,
            scoring=("accuracy"),
        )
        accuracy = np.mean(scores["test_score"])

        return accuracy


def objective_wrapper(x, y, **kwargs):
    def objective(trial):
        mlflow_params = {}
        mlflow.start_run(run_name=kwargs["mlflow_run_name"])

        classifier_name = trial.suggest_categorical(
            "classifier", ["SVC", "RandomForest"]
        )
        if classifier_name == "SVC":
            svc_c = trial.suggest_float("svc_c", 1e-10, 1e10, log=True)
            classifier_obj = sklearn.svm.SVC(C=svc_c, gamma="auto")
            mlflow_params["svc_c"] = svc_c
        else:
            rf_max_depth = trial.suggest_int("rf_max_depth", 2, 32, log=True)
            rf_n_estimators = trial.suggest_int(
                "rf_n_estimators", 2, 100, log=False
            )
            classifier_obj = sklearn.ensemble.RandomForestClassifier(
                max_depth=rf_max_depth,
                n_estimators=rf_n_estimators,
            )
            mlflow_params["rf_max_depth"] = rf_max_depth
            mlflow_params["rf_n_estimators"] = rf_n_estimators

        scores = sklearn.model_selection.cross_validate(
            classifier_obj,
            x,
            y,
            cv=kwargs["cv_folds"],
            n_jobs=-1,
            scoring=("accuracy"),
        )
        accuracy = np.mean(scores["test_score"])

        mlflow_params["classifier"] = classifier_name
        mlflow_params["count.train_x"] = x.shape[0]
        mlflow_params["count.train_y"] = y.shape[0]
        mlflow_params["cv_folds"] = kwargs["cv_folds"]
        mlflow.log_metric("avg_accuracy", accuracy)
        mlflow.log_params(mlflow_params)
        mlflow.end_run()
        return accuracy

    return objective


mlflc = MLflowCallback(mlflow_kwargs=dict(experiment_id=0))


@click.command()
@click.option("--mlflow-run-name", type=str, default="mlproject_template")
@click.option("--cv-folds", type=int, default=5)
@click.option("--n-trials", type=int, default=50)
@click.option("--seed-train-test-split", type=int, default=1234)
def main(**kwargs):
    iris = sklearn.datasets.load_iris()
    x, y = iris.data, iris.target
    (
        train_x,
        test_x,
        train_y,
        test_y,
    ) = sklearn.model_selection.train_test_split(
        x,
        y,
        test_size=0.30,
        random_state=kwargs["seed_train_test_split"],
        shuffle=True,
        stratify=y,
    )

    objective = Objective(train_x, train_y, cv_folds=kwargs["cv_folds"])
    study = optuna.create_study(direction="maximize")
    study.optimize(
        objective,
        n_trials=kwargs["n_trials"],
        callbacks=[mlflc],
    )
    print(study.best_trial)
    best_params = Utils.generate_params(study.best_trial)
    best_model = Classifier(best_params)
    best_model.fit(train_x, train_y)
    pred_y = best_model.predict(test_x)
    score = sklearn.metrics.accuracy_score(test_y, pred_y)
    print(score)


if __name__ == "__main__":
    main()
