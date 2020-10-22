import argparse
from pathlib import Path

import IPython
import joblib
import numpy as np
from joblib import Parallel, delayed, parallel_backend
from sklearn import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, make_scorer,
                             mean_squared_error, roc_auc_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     permutation_test_score, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR
from sklearn.utils import check_random_state, safe_indexing

from data import read_static_data

pipeline = {}
params = {}


def p_value_permute(estimator,
                    best_score,
                    scorer,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    n_permutations=1000,
                    random_state=0):
    """ https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/model_selection/_validation.py """
    random_state = check_random_state(random_state)
    y_train_size = y_train.shape[0]
    y_cat = np.concatenate((y_train, y_test))
    y_train_perms = []
    y_test_perms = []
    for _ in range(n_permutations):
        y_shuffled = _shuffle(y_cat, random_state)
        y_train_perms.append(y_shuffled[:y_train_size])
        y_test_perms.append(y_shuffled[y_train_size:])
    permutation_scores = Parallel()(delayed(fit_and_test)(clone(
        estimator), scorer, X_train, y_train_perms[i], X_test, y_test_perms[i])
                                    for i in range(n_permutations))
    permutation_scores = np.array(permutation_scores)
    pvalue = (np.sum(permutation_scores >= best_score) +
              1.0) / (n_permutations + 1)
    return pvalue, permutation_scores


def fit_and_test(estimator, scorer, X_train, y_train, X_test, y_test):
    estimator.fit(X_train, y_train)
    return scorer(estimator, X_test, y_test)


def _shuffle(y, random_state):
    indices = random_state.permutation(len(y))
    return safe_indexing(y, indices)


def main():
    print(args)

    pipeline = Pipeline([
        ("std", None),
        ("dim_red", None),
        ("clf", None),
    ])

    n_components_gs = [3, 5, 15]

    if args.model == "SVC":
        SVM = SVC
        params = [
            {
                "std": [StandardScaler()] if args.pca else
                [MinMaxScaler(), StandardScaler(), None],
                "dim_red": [PCA()] if args.pca else [None],
                "clf": [SVM()],
                "clf__kernel": ["rbf", "poly"],
                "clf__C": [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],
                "clf__gamma": ["scale"],
            },
        ]
        if args.pca:
            params[0]["dim_red__n_components"] = n_components_gs
    elif args.model == "LSVC":
        LinearSVM = LinearSVC
        params = [
            {
                "std": [MinMaxScaler(), StandardScaler(), None],
                "dim_red": [PCA()] if args.pca else [None],
                "clf": [LinearSVM()],
                "clf__C": [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],
                "clf__max_iter": [100000],
            },
        ]
        if args.pca:
            params[0]["dim_red__n_components"] = n_components_gs
    elif args.model == "LR":
        LR = LogisticRegression
        params = [
            {
                "std": [MinMaxScaler(), StandardScaler(), None],
                "dim_red": [PCA()] if args.pca else [None],
                "clf": [LR()],
            },
        ]
        if args.pca:
            params[0]["dim_red__n_components"] = n_components_gs
    elif args.model == 'LASSO':
        lasso = SGDClassifier
        loss = ''
        params = [
            {
                'std': [MinMaxScaler(), StandardScaler(), None],
                'dim_red': [PCA()] if args.pca else [None],
                'clf': [lasso()],
                'clf__loss': ['squared_loss'],
                'clf__penalty': ['l1'],
                'clf__alpha': [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],
            },
        ]
        if args.pca:
            params[0]['dim_red__n_components'] = n_components_gs
    elif args.model == "RF":
        RF = RandomForestClassifier
        params = [
            {
                "std": [MinMaxScaler(), StandardScaler(), None],
                "dim_red": [PCA()] if args.pca else [None],
                "clf": [RF()],
                "clf__n_estimators": [
                    500,
                ],
                "clf__max_depth": [None, 2, 8, 16],
                "clf__min_samples_split": [2, 0.1, 0.5],
                "clf__max_features": ["sqrt", "log2"],
            },
        ]
        if args.pca:
            params[0]["dim_red__n_components"] = n_components_gs
    elif args.model == "GB":
        GB = GradientBoostingClassifier
        loss_list = ["deviance", "exponential"]
        params = [
            {
                "std": [MinMaxScaler(), StandardScaler(), None],
                "dim_red": [PCA()] if args.pca else [None],
                "clf": [GB()],
                "clf__loss": loss_list,
                "clf__learning_rate": [0.1, 0.01, 0.001],
                "clf__n_estimators": [32, 100, 500],
                "clf__max_depth": [2, 8, 16],
                "clf__min_samples_split": [2, 0.1, 0.5],
            },
        ]
        if args.pca:
            params[0]["dim_red__n_components"] = n_components_gs

    X, y = read_static_data(args.data,
                            args.labels,
                            args.include,
                            skip_control=not args.with_control)
    print(f"X shape: {X.shape} y shape: {y.shape}")
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.5,
                                                        random_state=42,
                                                        stratify=y)
    print(f'trainset size: {X_train.shape[0]}')

    results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    included = set(args.include)
    included = ("all" if included == {"FC", "REHO", "ALFF", "fALFF"} else
                "_".join(sorted(included)))
    filename_infix = f"{included}{'_pca' if args.pca else ''}_{args.model}"
    model_path = results_dir / f"model_{filename_infix}.joblib"
    results_path = results_dir / f"results_{filename_infix}.txt"
    importances_path = results_dir / f"importances_{filename_infix}.csv"
    roc_fpr_path = results_dir / f"roc_fpr_{filename_infix}.csv"
    roc_tpr_path = results_dir / f"roc_tpr_{filename_infix}.csv"
    roc_thr_path = results_dir / f"roc_thr_{filename_infix}.csv"

    with parallel_backend("loky", n_jobs=args.jobs), open(results_path,
                                                          "w") as results_file:
        folding = StratifiedKFold(5)
        scoring = make_scorer(roc_auc_score, needs_threshold=True)
        if not model_path.exists():
            gs = GridSearchCV(pipeline,
                              params,
                              scoring=scoring,
                              cv=folding,
                              verbose=1)
            gs.fit(X_train, y_train)
            # save model
            joblib.dump(gs, model_path.resolve())
        else:
            gs = joblib.load(model_path.resolve())
        print(f"Best params: {gs.best_params_}", file=results_file)
        # print variance explained for PCA
        if args.pca:
            pca = gs.best_estimator_["dim_red"]
            print(
                f"PCA variance explained: \n {pca.explained_variance_ratio_.tolist()}",
                file=results_file,
            )
        # print feature importance on Random Forest
        if args.model == "RF":
            rf = gs.best_estimator_["clf"]
            importances = rf.feature_importances_
            if args.pca:
                importances = pca.inverse_transform(importances)
                importances = np.absolute(importances)
                importances /= np.sum(importances)
            np.savetxt(importances_path, [importances], delimiter=',')
        elif args.model == "LR" or args.model == "LASSO":
            model = gs.best_estimator_["clf"]
            importances = model.coef_
            if args.pca:
                importances = pca.inverse_transform(importances)
            importances = np.absolute(importances)
            importances /= np.sum(importances)
            np.savetxt(importances_path, importances, delimiter=',')
        # validate
        y_test_pred = gs.predict(X_test)
        if hasattr(gs, 'decision_function'):
            y_scores = gs.decision_function(X_test)
        elif hasattr(gs, 'predict_proba'):
            y_scores = gs.predict_proba(X_test)
            if y_scores.ndim > 1:
                y_scores = y_scores[:, 1]
        else:
            y_scores = None
        acc = accuracy_score(y_test, y_test_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
        specificity = tn / (tn + fp)
        precision = tp / (tp + fp)
        print(f"Accuracy: {acc}", file=results_file)
        print(f"Specificity: {specificity}", file=results_file)
        print(f"Precision: {precision}", file=results_file)
        if y_scores is not None:
            auroc = roc_auc_score(y_test, y_scores)
            print(f"AUROC: {auroc}", file=results_file)
            fprs, tprs, thrhlds = roc_curve(y_test, y_scores)
            np.savetxt(roc_fpr_path, fprs, delimiter=',')
            np.savetxt(roc_tpr_path, tprs, delimiter=',')
            np.savetxt(roc_thr_path, thrhlds, delimiter=',')
            # validation score
            print(f"Validation AUROC: {gs.best_score_}", file=results_file)
        # test on trainset
        y_train_pred = gs.predict(X_train)
        if hasattr(gs, 'decision_function'):
            y_train_scores = gs.decision_function(X_train)
        elif hasattr(gs, 'predict_proba'):
            y_train_scores = gs.predict_proba(X_train)
            if y_train_scores.ndim > 1:
                y_train_scores = y_train_scores[:, 1]
        else:
            y_train_scores = None
        train_acc = accuracy_score(y_train, y_train_pred)
        print(f"Train accuracy: {train_acc}", file=results_file)
        if y_train_scores is not None:
            train_auroc = roc_auc_score(y_train, y_train_scores)
            print(f"Train AUROC: {train_auroc}", file=results_file)
        # calculate p-values
        best_estimator = gs.best_estimator_
        pvalue, permutation_scores = p_value_permute(best_estimator, auroc,
                                                     scoring, X_train, y_train,
                                                     X_test, y_test)
        print(f"Test p-value: {pvalue}", file=results_file)
        # print(f"Test permutation scores: {permutation_scores}", file=results_file)

    if args.shell:
        IPython.embed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path, help="directory with data")
    parser.add_argument("labels", type=Path, help="csv file with labels")
    parser.add_argument("results_dir",
                        type=Path,
                        help="path to results directory")
    parser.add_argument(
        "--include",
        type=str,
        nargs="+",
        help="data to include: 'FC', 'REHO', 'ALFF', 'fALFF'",
        default=["FC", "REHO", "ALFF", "fALFF"],
    )
    parser.add_argument("--with_control",
                        action="store_true",
                        help="include the control group")
    parser.add_argument("--model",
                        default="SVC",
                        help="classifier type (e.g. SVM, RF)")
    parser.add_argument("--pca", action="store_true", help="apply PCA")
    parser.add_argument("--jobs",
                        type=int,
                        default=1,
                        help="number of processes")
    parser.add_argument("--shell",
                        action="store_true",
                        help="run IPython shell after completion")
    args = parser.parse_args()
    main()
