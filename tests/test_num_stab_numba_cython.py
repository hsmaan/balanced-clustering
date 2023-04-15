import pytest
import random

import numpy as np
import pandas as pd
from sklearn import cluster

from balanced_clustering import (
    balanced_adjusted_rand_index,
    balanced_adjusted_mutual_info,
    balanced_homogeneity,
    balanced_v_measure,
    balanced_completeness,
)

from balanced_clustering_cython import (
    balanced_adjusted_rand_index as balanced_adjusted_rand_index_cython,
    balanced_adjusted_mutual_info as balanced_adjusted_mutual_info_cython,
    balanced_homogeneity as balanced_homogeneity_cython,
    balanced_v_measure as balanced_v_measure_cython,
    balanced_completeness as balanced_completeness_cython
)

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Fixture for loading gaussian blobs of 3 classes with 1 minority class
@pytest.fixture
def three_classes_one_small():
    # Generate gaussian blobs simulating an imbalanced clustering setting
    c_1 = np.random.normal(loc=0, scale=0.5, size=(500, 2))
    c_2 = np.random.normal(loc=-2, scale=0.1, size=(20, 2))
    c_3 = np.random.normal(loc=3, scale=1, size=(500, 2))

    # Create dataframe of all data and return
    class_df = pd.DataFrame(
        {
            "x": np.concatenate((c_1[:, 0], c_2[:, 0], c_3[:, 0])),
            "y": np.concatenate((c_1[:, 1], c_2[:, 1], c_3[:, 1])),
            "cluster": np.concatenate(
                (
                    np.repeat("A", len(c_1)),
                    np.repeat("B", len(c_2)),
                    np.repeat("C", len(c_3)),
                )
            ),
        }
    )
    return class_df


# Fixture for loading two well separated clusters of the same size
@pytest.fixture
def two_classes_balanced():
    # Generate gaussian blobs simulating an imbalanced clustering setting
    c_1 = np.random.normal(loc=0, scale=0.5, size=(500, 2))
    c_2 = np.random.normal(loc=3, scale=1, size=(500, 2))

    # Create dataframe of all data and return
    class_df = pd.DataFrame(
        {
            "x": np.concatenate((c_1[:, 0], c_2[:, 0])),
            "y": np.concatenate((c_1[:, 1], c_2[:, 1])),
            "cluster": np.concatenate(
                (np.repeat("A", len(c_1)), np.repeat("B", len(c_2)))
            ),
        }
    )
    return class_df

# Fixture for three clusters, two with smaller sizes, overlapping in the middle
@pytest.fixture
def three_classes_mixed_imbalanced():
    # Generate gaussian blobs simulating an imbalanced clustering setting
    c_1 = np.random.normal(loc=0, scale=0.5, size=(1500, 2))
    c_2 = np.random.normal(loc=1.5, scale=1, size=(100, 2))
    c_3 = np.random.normal(loc=-1.5, scale=1, size=(500, 2))

    # Create dataframe of all data and return
    class_df = pd.DataFrame(
        {
            "x": np.concatenate((c_1[:, 0], c_2[:, 0], c_3[:, 0])),
            "y": np.concatenate((c_1[:, 1], c_2[:, 1], c_3[:, 1])),
            "cluster": np.concatenate(
                (
                    np.repeat("A", len(c_1)),
                    np.repeat("B", len(c_2)),
                    np.repeat("C", len(c_3)),
                )
            ),
        }
    )
    return class_df
    
# Function for generating k-means clusters from a given dataset
def k_means_df(class_df, n_clusters=2):
    # Perform k-means clustering and append result to dataframe
    cluster_arr = np.array(class_df.iloc[:, 0:2])
    kmeans_res = cluster.KMeans(n_clusters=n_clusters, random_state=0).fit_predict(
        X=cluster_arr
    )

    # Append results to dataframe and return
    class_df["kmeans"] = kmeans_res
    return class_df


def test_bal_ari_3_class_1_small(three_classes_one_small):
    # Perform k-means clustering on three classes with one minority class
    class_cluster_df = k_means_df(three_classes_one_small, n_clusters=2)

    # Calculate both versions of the balanced ARIs
    bal_ari = balanced_adjusted_rand_index(
        class_cluster_df["cluster"], class_cluster_df["kmeans"], reweigh=True
    )
    bal_ari_cython = balanced_adjusted_rand_index_cython(
        class_cluster_df["cluster"], class_cluster_df["kmeans"], reweigh=True
    )
    
    # Ensure that the two values are approximately equal
    assert bal_ari == pytest.approx(bal_ari_cython)
    
def test_bal_ari_2_class_balanced(two_classes_balanced):
    # Perform k-means clustering on two classes with the same size
    class_cluster_df = k_means_df(two_classes_balanced, n_clusters=2)

    # Calculate both versions of the balanced ARIs
    bal_ari = balanced_adjusted_rand_index(
        class_cluster_df["cluster"], class_cluster_df["kmeans"], reweigh=True
    )
    bal_ari_cython = balanced_adjusted_rand_index_cython(
        class_cluster_df["cluster"], class_cluster_df["kmeans"], reweigh=True
    )
    
    # Ensure that the two values are approximately equal
    assert bal_ari == pytest.approx(bal_ari_cython)


def test_bal_ari_3_class_mixed_imbalanced(three_classes_mixed_imbalanced):
    # Perform k-means clustering on three classes with mixed sizes and
    # imbalanced/overlapping
    class_cluster_df = k_means_df(three_classes_mixed_imbalanced, n_clusters=3)

    # Calculate both versions of the balanced ARIs
    bal_ari = balanced_adjusted_rand_index(
        class_cluster_df["cluster"], class_cluster_df["kmeans"], reweigh=True
    )
    bal_ari_cython = balanced_adjusted_rand_index_cython(
        class_cluster_df["cluster"], class_cluster_df["kmeans"], reweigh=True
    )
    
    # Ensure that the two values are approximately equal
    assert bal_ari == pytest.approx(bal_ari_cython)
    
    
def test_bal_aris_3_class_1_small(three_classes_one_small):
    # Perform k-means clustering on three classes with one minority class
    class_cluster_df = k_means_df(three_classes_one_small, n_clusters=2)

    # Calculate both versions of the balanced ARIs
    bal_ari = balanced_adjusted_rand_index(
        class_cluster_df["cluster"], class_cluster_df["kmeans"], reweigh=True
    )
    bal_ari_cython = balanced_adjusted_rand_index_cython(
        class_cluster_df["cluster"], class_cluster_df["kmeans"], reweigh=True
    )
    
    # Ensure that the two values are approximately equal
    assert bal_ari == pytest.approx(bal_ari_cython)

def test_bal_ami_2_class_balanced(two_classes_balanced):
    # Perform k-means clustering on two classes with the same size
    class_cluster_df = k_means_df(two_classes_balanced, n_clusters=2)

    # Calculated both versions of the balanced AMI
    bal_ami = balanced_adjusted_mutual_info(
        class_cluster_df["cluster"], class_cluster_df["kmeans"], reweigh=True
    )
    bal_ami_cython = balanced_adjusted_mutual_info_cython(
        class_cluster_df["cluster"], class_cluster_df["kmeans"], reweigh=True
    )
    
    # Ensure that the two values are approximately equal
    assert bal_ami == pytest.approx(bal_ami_cython)

def test_bal_ami_3_class_mixed_imbalanced(three_classes_mixed_imbalanced):
    # Perform k-means clustering on three classes with mixed sizes and
    # imbalanced/overlapping
    class_cluster_df = k_means_df(three_classes_mixed_imbalanced, n_clusters=3)

    # Calculated both versions of the balanced AMI
    bal_ami = balanced_adjusted_mutual_info(
        class_cluster_df["cluster"], class_cluster_df["kmeans"], reweigh=True
    )
    bal_ami_cython = balanced_adjusted_mutual_info(
        class_cluster_df["cluster"], class_cluster_df["kmeans"], reweigh=True
    )
    
    # Ensure that the two values are approximately equal
    assert bal_ami == pytest.approx(bal_ami_cython)

def test_bal_homogeneity_3_class_1_small(three_classes_one_small):
    # Perform k-means clustering on three classes with one minority class
    class_cluster_df = k_means_df(three_classes_one_small, n_clusters=2)

    # Calculate both versions of the balanced homogeneity
    bal_homog = balanced_homogeneity(
        class_cluster_df["cluster"], class_cluster_df["kmeans"], reweigh=True
    )
    bal_homog_cython = balanced_homogeneity_cython(
        class_cluster_df["cluster"], class_cluster_df["kmeans"], reweigh=True
    )
    
    # Ensure that the two values are approximately equal
    assert bal_homog == pytest.approx(bal_homog_cython)

def test_bal_v_measure_3_class_1_small(three_classes_one_small):
    # Perform k-means clustering on three classes with one minority class
    class_cluster_df = k_means_df(three_classes_one_small, n_clusters=2)

    # Calculated both versions of the balanced v-measure
    bal_v = balanced_v_measure(
        class_cluster_df["cluster"], class_cluster_df["kmeans"], reweigh=True
    )
    bal_v_cython = balanced_v_measure(
        class_cluster_df["cluster"], class_cluster_df["kmeans"], reweigh=True
    )
    
    # Ensure that the two values are approximately equal
    assert bal_v == pytest.approx(bal_v_cython)

def test_bal_complete_3_class_1_small(three_classes_one_small):
    # Perform k-means clustering on three classes with one minority class
    class_cluster_df = k_means_df(three_classes_one_small, n_clusters=2)
    
    # Calculate both versions of the balanced completeness
    bal_complete = balanced_completeness(
        class_cluster_df["cluster"], class_cluster_df["kmeans"], reweigh=True
    )
    bal_complete_cython = balanced_completeness_cython(
        class_cluster_df["cluster"], class_cluster_df["kmeans"], reweigh=True
    )
    
    # Ensure that the two values are approximately equal
    assert bal_complete == pytest.approx(bal_complete_cython)