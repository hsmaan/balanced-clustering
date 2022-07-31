# balanced-clustering
## Assessing clustering performance in imbalanced data contexts

Class imbalance is prevalent across real-world datasets, including images, natural language, and biological data. In unsupervised learning, clustering performance is often assessed with respect to a ground-truth set of labels using metrics such as the Adjusted Rand Index (ARI). Akin to the issue in classification when using overall accuracy, clustering metrics fail to capture information about class imbalance. imbalanced-clustering presents *balanced* clustering metrics, that take into account class imbalance and reweigh the results accordingly. Combined with vanilla clustering metrics (https://scikit-learn.org/stable/modules/clustering.html), imbalanced-clustering offers a more complete perspective on clustering and related tasks.

## Installation - **TBD**

```
# Installation for development
git clone https://github.com/hsmaan/imbalanced-clustering
cd imbalanced-clustering
poetry install 
```

## Usage 

Currently, there are five balanced metrics that can be used - **Balanced Adjusted Rand Index, Balanced Adjusted Mutual Information, Balanced Homogeneity, Balanced Completeness, Balanced V-measure**.

```
import numpy as np
from imbalanced_clustering import balanced_adjusted_rand_index

labels_sim = np.random.choice(["A","B","C"], 1000, replace=True)
cluster_sim = np.random.choice([1,2,3,4,5], 1000, replace=True)
balanced_adjusted_rand_index(labels_sim, cluster_sim)
```

## Notebooks 

For more details on the implementation of the balanced clustering metrics, mathematical formalism, and specific use-cases, please see [Walkthrough notebook #1](notebooks/01_imbalanced_metric_demo.ipynb). Extended experiments on interpolation behavior between the balanced and vanilla imbalanced metrics can be found in [Walkthrough notebook #2](notebooks/02_imbalanced_metric_interpolation_tests.ipynb).

## Issues/bugs

If any issues occur in either installation or usage, please open them and include a reproducible example. 
