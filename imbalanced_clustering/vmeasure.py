def homogeneity_completeness_v_measure(labels_true, labels_pred, *, beta=1.0):
    """Compute the homogeneity and completeness and V-Measure scores at once.
    Those metrics are based on normalized conditional entropy measures of
    the clustering labeling to evaluate given the knowledge of a Ground
    Truth class labels of the same samples.
    A clustering result satisfies homogeneity if all of its clusters
    contain only data points which are members of a single class.
    A clustering result satisfies completeness if all the data points
    that are members of a given class are elements of the same cluster.
    Both scores have positive values between 0.0 and 1.0, larger values
    being desirable.
    Those 3 metrics are independent of the absolute values of the labels:
    a permutation of the class or cluster label values won't change the
    score values in any way.
    V-Measure is furthermore symmetric: swapping ``labels_true`` and
    ``label_pred`` will give the same score. This does not hold for
    homogeneity and completeness. V-Measure is identical to
    :func:`normalized_mutual_info_score` with the arithmetic averaging
    method.
    Read more in the :ref:`User Guide <homogeneity_completeness>`.
    Parameters
    ----------
    labels_true : int array, shape = [n_samples]
        ground truth class labels to be used as a reference
    labels_pred : array-like of shape (n_samples,)
        cluster labels to evaluate
    beta : float, default=1.0
        Ratio of weight attributed to ``homogeneity`` vs ``completeness``.
        If ``beta`` is greater than 1, ``completeness`` is weighted more
        strongly in the calculation. If ``beta`` is less than 1,
        ``homogeneity`` is weighted more strongly.
    Returns
    -------
    homogeneity : float
       score between 0.0 and 1.0. 1.0 stands for perfectly homogeneous labeling
    completeness : float
       score between 0.0 and 1.0. 1.0 stands for perfectly complete labeling
    v_measure : float
        harmonic mean of the first two
    See Also
    --------
    homogeneity_score
    completeness_score
    v_measure_score
    """
    labels_true, labels_pred = check_clusterings(labels_true, labels_pred)

    if len(labels_true) == 0:
        return 1.0, 1.0, 1.0

    entropy_C = entropy(labels_true)
    entropy_K = entropy(labels_pred)

    contingency = contingency_matrix(labels_true, labels_pred, sparse=True)
    MI = mutual_info_score(None, None, contingency=contingency)

    homogeneity = MI / (entropy_C) if entropy_C else 1.0
    completeness = MI / (entropy_K) if entropy_K else 1.0

    if homogeneity + completeness == 0.0:
        v_measure_score = 0.0
    else:
        v_measure_score = (
            (1 + beta)
            * homogeneity
            * completeness
            / (beta * homogeneity + completeness)
        )

    return homogeneity, completeness, v_measure_score