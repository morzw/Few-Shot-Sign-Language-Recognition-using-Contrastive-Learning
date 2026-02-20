import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform


def knn_support_query_accuracy(
    support_embeddings: np.ndarray,
    support_labels: np.ndarray,
    query_embeddings: np.ndarray,
    query_labels: np.ndarray,
    k: int = 5,
):
    distances = cdist(query_embeddings, support_embeddings, metric="euclidean")
    y_pred = []
    for row in distances:
        nn_idx = np.argsort(row)[:k]
        nn_labels = support_labels[nn_idx]
        y_pred.append(np.bincount(nn_labels).argmax())
    y_pred = np.asarray(y_pred)
    acc = float(np.mean(y_pred == query_labels))
    return y_pred, acc


def knn_leave_one_out_accuracy(embeddings: np.ndarray, labels: np.ndarray, k: int = 3):
    distances = squareform(pdist(embeddings, metric="euclidean"))
    y_pred = []
    for i, row in enumerate(distances):
        nn_idx = np.argpartition(row, k + 1)[: k + 1]
        nn_idx = nn_idx[nn_idx != i][:k]
        nn_labels = labels[nn_idx]
        y_pred.append(np.bincount(nn_labels).argmax())
    y_pred = np.asarray(y_pred)
    acc = float(np.mean(y_pred == labels))
    return y_pred, acc

