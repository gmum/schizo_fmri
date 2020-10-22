import argparse
from functools import partial
from pathlib import Path

import IPython
import joblib
import numpy as np
import scipy
from joblib import Parallel, delayed, parallel_backend
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, make_scorer, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC, LinearSVC

from common import stratified_group_k_fold
from data import read_dynamic_data
from sklearn.utils import check_random_state, safe_indexing
from sklearn import clone

SEED = 666


def p_value_permute(estimator,
                    best_score,
                    scorer,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    n_permutations=1000,
                    random_state=0):
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
    X_train_sliced, y_train_sliced, train_groups_sliced = slice_data(
        X_train, y_train)
    X_test_sliced, y_test_sliced, test_groups_sliced = slice_data(
        X_test, y_test)
    estimator.fit(X_train_sliced, y_train_sliced)
    aggr_test_acc, aggr_test_auroc, aggr_test_specificity, aggr_test_precision = aggregate_test(
        estimator,
        X_test_sliced,
        test_groups_sliced,
        y_test,
        aggr_type='count')
    return aggr_test_auroc


def _shuffle(y, random_state):
    indices = random_state.permutation(len(y))
    return safe_indexing(y, indices)


def slice_data(X, y):
    samples = X.shape[0]
    time_size = X.shape[1]
    features_size = X.shape[2]
    X_sliced = X.reshape(-1, features_size)
    y_sliced = np.repeat(y, time_size)
    groups_sliced = np.repeat(np.arange(samples, dtype=np.int32), time_size)
    return X_sliced, y_sliced, groups_sliced


def sliced_test(model_sliced, X_sliced, y_sliced, auroc=True):
    y_sliced_pred = model_sliced.predict(X_sliced)
    if hasattr(model_sliced, 'decision_function'):
        y_sliced_scores = model_sliced.decision_function(X_sliced)
    elif hasattr(model_sliced, 'predict_proba'):
        y_sliced_scores = model_sliced.predict_proba(X_sliced)
        if y_sliced_scores.ndim > 1:
            y_sliced_scores = y_sliced_scores[:, 1]
    else:
        y_sliced_scores = None
    acc = accuracy_score(y_sliced, y_sliced_pred)
    if auroc:
        if y_sliced_scores is not None:
            auroc = roc_auc_score(y_sliced, y_sliced_scores)
            return acc, auroc
        else:
            return acc, -0.0
    else:
        return acc


def aggregate_predictions(y_pred_sliced, series_len, aggr_type='average'):
    preds = y_pred_sliced.reshape((-1, series_len))
    if aggr_type == 'average':
        averaged_preds = np.average(preds, axis=1)
        return averaged_preds
    elif aggr_type == 'skew':
        aggr_preds = scipy.stats.skew(preds, axis=1)
        aggr_preds = np.array((aggr_preds <= 0), dtype=np.double)
        return aggr_preds
    elif aggr_type == 'count':
        aggr_preds = np.average(preds.round(), axis=1)
        return aggr_preds


def aggregate_test(model_sliced,
                   X_sliced,
                   groups_sliced,
                   y,
                   aggr_type='average'):
    series_len = np.unique(groups_sliced, return_counts=True)[1][0]
    y_pred_sliced = model_sliced.predict(X_sliced)
    if hasattr(model_sliced, 'decision_function'):
        y_scores_sliced = model_sliced.decision_function(X_sliced)
    elif hasattr(model_sliced, 'predict_proba'):
        y_scores_sliced = model_sliced.predict_proba(X_sliced)
        if y_scores_sliced.ndim > 1:
            y_scores_sliced = y_scores_sliced[:, 1]
    else:
        y_scores_sliced = None
    y_pred = aggregate_predictions(y_pred_sliced,
                                   series_len,
                                   aggr_type=aggr_type).round()
    acc = accuracy_score(y, y_pred)
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp)
    if y_scores_sliced is not None:
        y_scores = aggregate_predictions(y_scores_sliced,
                                         series_len,
                                         aggr_type=aggr_type)
        auroc = roc_auc_score(y, y_scores)
        return acc, auroc, specificity, precision
    else:
        return acc, -0.0, specificity, precision


def main():
    print(args)

    if 'FC' in args.include:
        print(f'Using IncrementalPCA instead of PCA')
        from sklearn.decomposition import IncrementalPCA
        PCA = partial(IncrementalPCA, batch_size=100)
    else:
        from sklearn.decomposition import PCA

    n_components_gs = [5, 20, 100]

    pipeline = Pipeline([
        ('std', None),
        ('dim_red', None),
        ('clf', None),
    ])

    if args.model == 'SVC':
        params = [
            {
                'std': [StandardScaler()] if args.pca else
                [MinMaxScaler(), StandardScaler(), None],
                'dim_red': [PCA()] if args.pca else [None],
                'clf': [SVC()],
                'clf__kernel': ['rbf', 'poly'],
                'clf__C': [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],
                'clf__gamma': ['scale'],
            },
        ]
        if args.pca:
            params[0]['dim_red__n_components'] = n_components_gs
    elif args.model == 'LSVC':
        params = [
            {
                'std': [MinMaxScaler(), StandardScaler(), None],
                'dim_red': [PCA()] if args.pca else [None],
                'clf': [LinearSVC()],
                'clf__C': [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],
                'clf__max_iter': [100000],
            },
        ]
        if args.pca:
            params[0]['dim_red__n_components'] = n_components_gs
    elif args.model == 'LR':
        params = [
            {
                'std': [MinMaxScaler(), StandardScaler(), None],
                'dim_red': [PCA()] if args.pca else [None],
                'clf': [LogisticRegression()],
            },
        ]
        if args.pca:
            params[0]['dim_red__n_components'] = n_components_gs
    elif args.model == 'LASSO':
        lasso = SGDClassifier
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
    elif args.model == 'RF':
        params = [
            {
                'std': [MinMaxScaler(), StandardScaler(), None],
                'dim_red': [PCA()] if args.pca else [None],
                'clf': [RandomForestClassifier()],
                'clf__n_estimators': [
                    500,
                ],
                'clf__max_depth': [None, 2, 8, 16],
                'clf__min_samples_split': [2, 0.1, 0.5],
                'clf__max_features': ['sqrt', 'log2'],
            },
        ]
        if args.pca:
            params[0]['dim_red__n_components'] = n_components_gs
    elif args.model == 'GB':
        params = [
            {
                'std': [MinMaxScaler(), StandardScaler(), None],
                'dim_red': [PCA()] if args.pca else [None],
                'clf': [GradientBoostingClassifier()],
                'clf__loss': ['deviance', 'exponential'],
                'clf__learning_rate': [0.1, 0.01, 0.001],
                'clf__n_estimators': [32, 100, 500],
                'clf__max_depth': [2, 8, 16],
                'clf__min_samples_split': [2, 0.1, 0.5],
            },
        ]
        if args.pca:
            params[0]['dim_red__n_components'] = n_components_gs

    X, y = read_dynamic_data(args.data, args.labels, args.include)
    print(f'X shape: {X.shape} y shape: {y.shape}')
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.5,
                                                        random_state=42,
                                                        stratify=y)

    # slice dataset into windows
    X_train_sliced, y_train_sliced, train_groups_sliced = slice_data(
        X_train, y_train)
    X_test_sliced, y_test_sliced, test_groups_sliced = slice_data(
        X_test, y_test)
    print(
        f'X_train_sliced.shape: {X_train_sliced.shape} y_train_sliced.shape: {y_train_sliced.shape}'
        f' train_groups_sliced.shape: {train_groups_sliced.shape}')

    # create files to save the results to
    results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    included = set(args.include)
    included = 'all' if included == {'FC', 'REHO', 'ALFF', 'fALFF'}\
        else '_'.join(sorted(included))
    filename_infix = f"{included}{'_pca' if args.pca else ''}_{args.model}"
    model_path = results_dir / f'model_{filename_infix}.joblib'
    results_path = results_dir / f'dynamic_results_{filename_infix}.txt'
    importances_path = results_dir / f"importances_{filename_infix}.csv"

    with parallel_backend('loky', n_jobs=args.jobs), open(results_path,
                                                          'w') as results_file:
        scoring = make_scorer(roc_auc_score, needs_threshold=True)
        if not model_path.exists():
            gs = GridSearchCV(pipeline,
                              params,
                              scoring=scoring,
                              cv=list(
                                  stratified_group_k_fold(X_train_sliced,
                                                          y_train_sliced,
                                                          train_groups_sliced,
                                                          5,
                                                          seed=SEED)),
                              verbose=1)
            gs.fit(X_train_sliced, y_train_sliced, groups=train_groups_sliced)
            # save model
            joblib.dump(gs, model_path.resolve())
        else:
            gs = joblib.load(model_path.resolve())
        print(f'Best params: {gs.best_params_}', file=results_file)
        if args.pca:
            pca = gs.best_estimator_['dim_red']
            print(
                f'PCA variance explained: \n {pca.explained_variance_ratio_.tolist()}',
                file=results_file)
        # print feature importance on Random Forest
        if args.model == "RF":
            rf = gs.best_estimator_["clf"]
            importances = rf.feature_importances_
            if args.pca:
                importances = pca.inverse_transform(importances)
                importances = np.absolute(importances)
                importances /= np.sum(importances)
            np.savetxt(importances_path, [importances], delimiter=',')
        elif args.model == "LR":
            lr = gs.best_estimator_["clf"]
            importances = lr.coef_
            if args.pca:
                importances = pca.inverse_transform(importances)
            importances = np.absolute(importances)
            importances /= np.sum(importances)
            np.savetxt(importances_path, importances, delimiter=',')
        # test on trainset
        train_acc, train_auroc = sliced_test(gs, X_train_sliced,
                                             y_train_sliced)
        print(f'Sliced train AUROC: {train_auroc}', file=results_file)
        print(f'Sliced train accuracy: {train_acc}', file=results_file)
        # validation score
        print(f'Sliced validation AUROC: {gs.best_score_}', file=results_file)
        # test on testset - sliced
        test_acc, test_auroc = sliced_test(gs, X_test_sliced, y_test_sliced)
        print(f'Sliced test AUROC: {test_auroc}', file=results_file)
        print(f'Sliced test accuracy: {test_acc}', file=results_file)
        # test on testset - aggregated
        for aggr_type in ['average', 'skew', 'count']:
            aggr_test_acc, aggr_test_auroc, aggr_test_specificity, aggr_test_precision = aggregate_test(
                gs,
                X_test_sliced,
                test_groups_sliced,
                y_test,
                aggr_type=aggr_type)
            if aggr_type == 'count':
                auroc = aggr_test_auroc
            print(f'Aggregate with {aggr_type} test AUROC: {aggr_test_auroc}',
                  file=results_file)
            print(f'Aggregate with {aggr_type} test accuracy: {aggr_test_acc}',
                  file=results_file)
            print(
                f'Aggregate with {aggr_type} test specificity: {aggr_test_specificity}',
                file=results_file)
            print(
                f'Aggregate with {aggr_type} test precision: {aggr_test_precision}',
                file=results_file)
        # test each patient separately
        for i in range(X_test.shape[0]):
            subject_X = X_test[i:i + 1]
            subject_y = y_test[i:i + 1]
            sliced_subject_X, sliced_subject_y, _ = slice_data(
                subject_X, subject_y)
            subject_acc = sliced_test(gs,
                                      sliced_subject_X,
                                      sliced_subject_y,
                                      auroc=False)
            print(f'Subject {i} sliced test accuracy: {subject_acc}',
                  file=results_file)
        # calculate p-values
        best_estimator = gs.best_estimator_
        pvalue, permutation_scores = p_value_permute(best_estimator, auroc,
                                                     scoring, X_train, y_train,
                                                     X_test, y_test)
        print(f"Test p-value: {pvalue}", file=results_file)

    if args.shell:
        IPython.embed()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=Path, help='directory with data')
    parser.add_argument('labels', type=Path, help='csv file with labels')
    parser.add_argument("results_dir",
                        type=Path,
                        help="path to results directory")
    parser.add_argument('--include',
                        type=str,
                        nargs='+',
                        help="data to include: 'FC', 'REHO', 'ALFF', 'fALFF'",
                        default=['FC', 'REHO', 'ALFF', 'fALFF'])
    parser.add_argument('--model',
                        default='SVC',
                        help='classifier type (e.g. SVM, RF)')
    parser.add_argument('--pca', action='store_true', help='apply PCA')
    parser.add_argument('--jobs',
                        type=int,
                        default=1,
                        help='number of processes')
    parser.add_argument('--shell',
                        action='store_true',
                        help='run IPython shell after completion')
    args = parser.parse_args()
    main()
