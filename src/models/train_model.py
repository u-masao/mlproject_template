import click
import mlflow
import numpy as np
import optuna
import sklearn.datasets
import sklearn.ensemble
import sklearn.model_selection
import sklearn.svm


def objective_wrapper(x, y, **kwargs):
    def objective(trial):
        mlflow.start_run(run_name=kwargs["mlflow_run_name"])

        classifier_name = trial.suggest_categorical(
            "classifier", ["SVC", "RandomForest"]
        )
        mlflow_params = {}
        mlflow_params["classifier"] = classifier_name

        if classifier_name == "SVC":
            svc_c = trial.suggest_float("svc_c", 1e-10, 1e10, log=True)
            classifier_obj = sklearn.svm.SVC(C=svc_c, gamma="auto")
            mlflow_params["svc_c"] = svc_c
        else:
            rf_max_depth = trial.suggest_int("rf_max_depth", 2, 32, log=True)
            classifier_obj = sklearn.ensemble.RandomForestClassifier(
                max_depth=rf_max_depth, n_estimators=10
            )
            mlflow_params["rf_max_depth"] = rf_max_depth

        scores = sklearn.model_selection.cross_validate(
            classifier_obj,
            x,
            y,
            cv=3,
            n_jobs=-1,
            scoring=("accuracy"),
        )
        accuracy = np.mean(scores["test_score"])

        mlflow.log_metric("avg_accuracy", accuracy)
        mlflow.log_params(mlflow_params)
        mlflow.end_run()
        return accuracy

    return objective


@click.command()
@click.option("--mlflow-run-name", type=str, default="mlproject_template")
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

    study = optuna.create_study(direction="maximize")
    study.optimize(objective_wrapper(train_x, train_y, **kwargs), n_trials=100)
    print(study.best_trial)


if __name__ == "__main__":
    main()
