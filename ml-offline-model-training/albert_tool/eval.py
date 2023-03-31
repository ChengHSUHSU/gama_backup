import numpy as np


def ranking_precision_score(y_true, y_score, k=10):
    """Precision at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    Returns
    -------
    precision @k : float
    """
    # y_true = [int(id) for id in y_true]
    # y_score = [int(id) for id in y_score]
    unique_y = np.unique(y_true)

    if len(unique_y) > 2:
        raise ValueError("Only supported for two relevance levels.")

    pos_label = unique_y[1]
    n_pos = np.sum(y_true == pos_label)

    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    n_relevant = np.sum(y_true == pos_label)

    # Divide by min(n_pos, k) such that the best achievable score is always 1.0.
    return float(n_relevant) / min(n_pos, k)


def average_precision_score(y_true, y_score, k=10):
    """Average precision at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    Returns
    -------
    average precision @k : float
    """
    unique_y = np.unique(y_true)

    if len(unique_y) > 2:
        raise ValueError("Only supported for two relevance levels.")

    pos_label = unique_y[1]
    n_pos = np.sum(y_true == pos_label)

    order = np.argsort(y_score)[::-1][:min(n_pos, k)]
    y_true = np.asarray(y_true)[order]

    score = 0
    for i in xrange(len(y_true)):
        if y_true[i] == pos_label:
            # Compute precision up to document i
            # i.e, percentage of relevant documents up to document i.
            prec = 0
            for j in xrange(0, i + 1):
                if y_true[j] == pos_label:
                    prec += 1.0
            prec /= (i + 1.0)
            score += prec

    if n_pos == 0:
        return 0

    return score / n_pos


def dcg_score(y_true, y_score, k=10, gains="exponential"):
    """Discounted cumulative gain (DCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    gains : str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    DCG @k : float
    """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])

    if gains == "exponential":
        gains = 2 ** y_true - 1
    elif gains == "linear":
        gains = y_true
    else:
        raise ValueError("Invalid gains option.")

    # highest rank is 1 so +2 instead of +1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10, gains="exponential"):
    """Normalized discounted cumulative gain (NDCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    gains : str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    NDCG @k : float
    """
    # y_true = [int(id) for id in y_true]
    # y_score = [int(id) for id in y_score]
    best = dcg_score(y_true, y_true, k, gains)
    actual = dcg_score(y_true, y_score, k, gains)
    return actual / best

def demo_ndcg_score():
    '''
    background:
    given total 10 contents, we calculate NDCG@K (k=5) 
    y_score : its score, probability
    y_true : ground truth (1 or 0)
    example, A, B model, and A is better than B
    '''
    # setting
    k = 5
    y_score_A = [0.9, 0.1, 0.2, 0.5, 0.2, 0.8, 0.7, 0.7, 0.1, 0.1]
    y_score_B = [0.2, 0.1, 0.2, 0.5, 0.2, 0.8, 0.7, 0.7, 0.1, 0.1]
    y_true = [1,0,0,1,0,1,1,1,0,0]

    # demo
    ndcg_A = ndcg_score(y_true=y_true, y_score=y_score_A, k=k)
    ndcg_B = ndcg_score(y_true=y_true, y_score=y_score_B, k=k)

    print('ndcg_A : ', ndcg_A)
    print('ndcg_B : ', ndcg_B)




if __name__ == '__main__':
    demo_ndcg_score()

    


