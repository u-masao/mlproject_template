import click
import mlflow
import optuna
import sklearn.datasets
import sklearn.ensemble
import sklearn.model_selection
import sklearn.svm


def objective(trial):
    mlflow.start_run()
    iris = sklearn.datasets.load_iris()
    x, y = iris.data, iris.target

    classifier_name = trial.suggest_categorical(
        "classifier", ["SVC", "RandomForest"]
    )
    mlflow.log_param("classifier", classifier_name)
    if classifier_name == "SVC":
        svc_c = trial.suggest_float("svc_c", 1e-10, 1e10, log=True)
        classifier_obj = sklearn.svm.SVC(C=svc_c, gamma="auto")
        mlflow.log_param("svc_c", svc_c)
    else:
        rf_max_depth = trial.suggest_int("rf_max_depth", 2, 32, log=True)
        classifier_obj = sklearn.ensemble.RandomForestClassifier(
            max_depth=rf_max_depth, n_estimators=10
        )
        mlflow.log_param("rf_max_depth", rf_max_depth)

    score = sklearn.model_selection.cross_val_score(
        classifier_obj, x, y, n_jobs=-1, cv=3
    )
    accuracy = score.mean()
    mlflow.log_metric("avg_accuracy", accuracy)

    mlflow.end_run()
    return accuracy


@click.command()
@click.option("--mlflow-run-name", type=str, default="mlproject_template")
def main(**kwargs):
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)
    print(study.best_trial)


if __name__ == "__main__":
    main()
