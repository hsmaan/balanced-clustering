# imbalanced-clustering
## Assessing clustering performance in imbalanced data contexts  

Class imbalance is prevalent across real-world datasets, including images, natural language, and biological data. In unsupervised learning, clustering performance is often assessed with respect to a ground-truth set of labels using metrics such as the Adjusted Rand Index (ARI). Akin to the issue in classification when using overall accuracy, clustering metrics fail to capture information about class imbalance. imbalanced-clustering presents *balanced* clustering metrics, that take into account class imbalance and reweigh the results accordingly. Combined with vanilla clustering metrics (https://scikit-learn.org/stable/modules/clustering.html), imbalanced-clustering offers a more complete perspective on clustering and related tasks.


