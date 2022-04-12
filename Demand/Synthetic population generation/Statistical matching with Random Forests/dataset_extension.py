#!/usr/bin/python
import itertools
import numpy as np
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier


def statistical_matching(x_mat1, y_mat1, x_mat2, M=20, verbose=0):
    K = x_mat1.shape[1]
    y_mat2 = np.zeros((x_mat2.shape[0], y_mat1.shape[1]), np.int32)
    for i in range(x_mat2.shape[0]):
        if verbose and ((i % 100) == 0):
            print(f'Processed up to example {i} of {x_mat2.shape[0]}')
        k = K
        while k > 0:
            candidate_attrs = list(itertools.combinations(range(K), k))
            exit_while = False
            for subset in candidate_attrs:
                s = list(subset)
                idx = np.all(x_mat2[None, i, s] == x_mat1[:, s], axis=1)
                n_matchings = idx.sum()
                if n_matchings > M:
                    choice = np.random.randint(0, n_matchings)
                    y_mat2[i, :] = y_mat1[idx, :][choice, :]
                    exit_while = True
                    break
            if exit_while:
                break
            else:
                k = k - 1
    return y_mat2


def statistical_matching_cv_test(x_mat, y_mat, n_folds=5):
    kf = KFold(n_splits=n_folds, shuffle=False)
    kf.get_n_splits(y_mat)
    for fold, (train_idx, test_idx) in enumerate(kf.split(y_mat)):
        y_pred = statistical_matching(x_mat[train_idx, :], y_mat[train_idx, :], x_mat[test_idx, :], M=20, verbose=1)
        f1_score, precision, recall = _get_precision_recall_f1(y_mat[test_idx, :], y_pred)


def _get_precision_recall_f1(y_mat, y_pred):
    precision = np.zeros((y_mat.shape[1], ))
    recall = np.zeros((y_mat.shape[1], ))
    f1_score = np.zeros((y_mat.shape[1], ))
    for output in range(y_mat.shape[1]):
        if isinstance(y_pred, list):
            precision[output], recall[output], f1_score[output], _ = metrics.precision_recall_fscore_support(
                y_mat[:, output], np.argmax(y_pred[output], axis=1), average='binary')
        else:
            precision[output], recall[output], f1_score[output], _ = metrics.precision_recall_fscore_support(
                y_mat[:, output], y_pred[:, output], average='binary')
        # f1_score = 2 * precision * recall / (precision + recall)
        print(f'Output {output} --> F1-score: {f1_score[output]}, Precision: {precision[output]}, '
              f'Recall: {recall[output]}')
    return f1_score, precision, recall


def to_onehot(x_mat1, x_mat2=None):
    if x_mat2 is None:
        x_mat = x_mat1.copy()
    else:
        x_mat = np.concatenate((x_mat1, x_mat2), axis=0)
    x_oh = np.zeros((x_mat.shape[0], 0), dtype=np.bool_)
    for k in range(x_mat.shape[1]):
        categories = np.sort(np.unique(x_mat[:, k]))[None, :]
        x_oh = np.concatenate((x_oh, x_mat[:, k, None] == categories), axis=1)
    if x_mat2 is None:
        return x_oh
    else:
        return x_oh[:x_mat1.shape[0], :], x_oh[x_mat1.shape[0]:, :]


def extend_with_rf(x_mat1, y_mat1, x_mat2, n_estimators=1000, n_splits=None):
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=n_splits, criterion='entropy')
    rf.fit(x_mat1, y_mat1)
    return rf.predict_proba(x_mat2)


def rf_cv_test(x_mat, y_mat, n_folds=5, n_estimators=1000, n_splits=None):
    kf = KFold(n_splits=n_folds, shuffle=False)
    kf.get_n_splits(y_mat)
    for fold, (train_idx, test_idx) in enumerate(kf.split(y_mat)):
        y_pred = extend_with_rf(x_mat[train_idx, :], y_mat[train_idx, :], x_mat[test_idx, :], n_estimators=n_estimators,
                                n_splits=n_splits)
        f1_score, precision, recall = _get_precision_recall_f1(y_mat[test_idx, :], y_pred)


def sample_from_prob(y_prob):
    y_sampled = np.zeros((y_prob[0].shape[0], len(y_prob)), dtype=np.bool_)
    for k in range(y_sampled.shape[1]):
        y_sampled[:, k] = np.random.rand(y_prob[k].shape[0]) > y_prob[k][:, 0]
    return y_sampled

