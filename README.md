# balanced-clustering <!-- omit in toc -->
## Assessing clustering performance in imbalanced data contexts <!-- omit in toc -->

Class imbalance is prevalent across real-world datasets, including images, natural language, and biological data. In unsupervised learning, clustering performance is often assessed with respect to a ground-truth set of labels using metrics such as the Adjusted Rand Index (ARI). Akin to the issue in classification when using overall accuracy, clustering metrics fail to capture information about class imbalance. imbalanced-clustering presents *balanced* clustering metrics, that take into account class imbalance and reweigh the results accordingly. Combined with vanilla clustering metrics (https://scikit-learn.org/stable/modules/clustering.html), imbalanced-clustering offers a more complete perspective on clustering and related tasks.

## Table of contents  <!-- omit in toc -->
- [Installation via pip](#installation-via-pip)
- [Usage](#usage)
- [Detailed example](#detailed-example)
- [Notebooks](#notebooks)
- [Issues/bugs](#issuesbugs)
- [Citation information](#citation-information)

## Installation via pip

```
pip install balanced-clustering
```

## Usage 

Currently, there are five balanced metrics that can be used - **Balanced Adjusted Rand Index, Balanced Adjusted Mutual Information, Balanced Homogeneity, Balanced Completeness, Balanced V-measure**.

```
import numpy as np
from balanced_clustering import balanced_adjusted_rand_index

labels_sim = np.random.choice(["A","B","C"], 1000, replace=True)
cluster_sim = np.random.choice([1,2,3,4,5], 1000, replace=True)
balanced_adjusted_rand_index(labels_sim, cluster_sim)
```

## Detailed example 

Consider the following example where we have 3 classes simulated from isotropic Gaussian distributions, and the clustering algorithm mis-clusters the smallest class:

```
import numpy as np
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, \
    homogeneity_score, completeness_score, v_measure_score
from sklearn.cluster import KMeans

from balanced_clustering import balanced_adjusted_rand_index, \
    balanced_adjusted_mutual_info, balanced_completeness, \
    balanced_homogeneity, balanced_v_measure, return_metrics

# Set a seed for reproducibility and create generator
np.random.seed(42)

# Sample three classes from separated gaussian distributions with varying
# standard deviations and class size 
c_1 = np.random.default_rng(seed = 0).normal(loc = 0, scale = 0.5, size = (500, 2))
c_2 = np.random.default_rng(seed = 1).normal(loc = -2, scale = 0.1, size = (20, 2))
c_3 = np.random.default_rng(seed = 2).normal(loc = 3, scale = 1, size = (500, 2))
classes = np.concatenate(
    [np.repeat("A", len(c_1)), np.repeat("B", len(c_2)), np.repeat("C", len(c_3))]
)

# Perform k-means clustering with k = 2 - this misclusters the smallest class 
cluster_arr = np.concatenate([c_1, c_2, c_3])
kmeans_res = KMeans(n_clusters = 2, random_state = 42).fit_predict(X = cluster_arr)

# Return and print balanced and imbalanced comparisons 
return_metrics(
    class_arr = classes, cluster_arr = kmeans_res,
)
```

```
ARI imbalanced: 0.915 ARI balanced: 0.5434
AMI imbalanced: 0.8671 AMI balanced: 0.686
Homogeneity imbalanced: 0.8204 Homogeneity balanced: 0.5402
Completeness imbalanced: 0.9198 Completeness balanced : 0.941
V-measure imbalanced: 0.8673 V-measure balanced: 0.6864
```

## Notebooks 

For more details on the implementation of the balanced clustering metrics, mathematical formalism, and specific use-cases, please see [Walkthrough notebook #1](notebooks/01_imbalanced_metric_demo.ipynb). Extended experiments on interpolation behavior between the balanced and vanilla imbalanced metrics can be found in [Walkthrough notebook #2](notebooks/02_imbalanced_metric_interpolation_tests.ipynb).

**Please note that *Extended Documentation* is currently a work in progress**

## Issues/bugs

If any issues occur in either installation or usage, please open them and include a reproducible example. 

## Citation information

If you use the balanced clustering metrics in your research, please reference the following publication:

> The differential impacts of dataset imbalance in single-cell data integration
>
> Hassaan Maan, Lin Zhang, Chengxin Yu, Michael Geuenich, Kieran R. Campbell, Bo Wang
>
> bioRxiv December 19, 2022; doi: https://doi.org/10.1101/2022.10.06.511156  