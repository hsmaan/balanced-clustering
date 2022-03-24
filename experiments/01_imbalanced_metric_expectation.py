from dataclasses import replace
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, \
    homogeneity_score, completeness_score, v_measure_score
from sklearn.cluster import KMeans

from imbalanced_clustering import balanced_adjusted_rand_index, \
    balanced_adjusted_mutual_info, balanced_completeness, \
    balanced_homogeneity, balanced_v_measure 

# Define function to return all the relevant metrics - both balanced
# and imbalanced
def return_metrics(class_arr, cluster_arr):
    # Determine the imbalanced (base) metric scores 
    ari_imbalanced = adjusted_rand_score(class_arr, cluster_arr)
    ami_imbalanced = adjusted_mutual_info_score(class_arr, cluster_arr)
    homog_imbalanced = homogeneity_score(class_arr, cluster_arr)
    complete_imbalanced = completeness_score(class_arr, cluster_arr)
    v_measure_imbalanced = v_measure_score(class_arr, cluster_arr)

    # Determine the balanced metrics from `imbalanced-clustering`
    ari_balanced = balanced_adjusted_rand_index(class_arr, cluster_arr)
    ami_balanced = balanced_adjusted_mutual_info(class_arr, cluster_arr)
    homog_balanced = balanced_homogeneity(class_arr, cluster_arr)
    complete_balanced = balanced_completeness(class_arr, cluster_arr)
    v_measure_balanced = balanced_v_measure(class_arr, cluster_arr)
    
    # Return paired balanced imbalance scores
    return (ari_imbalanced, ari_balanced), (ami_imbalanced, ami_balanced), \
        (homog_imbalanced, homog_balanced), (complete_imbalanced, complete_balanced), \
        (v_measure_imbalanced, v_measure_balanced)
        
# Define function for generating completely random data from uniform
# distributions, clustering and scoring results 
def random_data(num_classes, num_clusters, min_class_size, max_class_size):
    # Ensure number of classes and clusters is greater than 1
    assert num_classes > 1 and num_clusters > 1
    
    # Sample class sizes from uniform distribution
    class_size_samples = [
        np.random.randint(min_class_size, max_class_size) for i in range(num_classes)
    ]
    
    # Iterate over each class and generate random data
    x_1_all = []
    x_2_all = []
    class_vals = []
    for i in range(num_classes):
        # Sample class values from uniform distribution
        x = np.random.uniform(0, 100, size=(class_size_samples[i], 2))
        # Append values to list 
        x_1_all.append(x[:, 0])
        x_2_all.append(x[:, 1])        
        class_vals.append(i * np.ones(class_size_samples[i], dtype = "int"))
    
    # Concatenate all class values into single dataframe 
    class_df = pd.DataFrame({
        "x": np.concatenate(x_1_all),
        "y": np.concatenate(x_2_all),
        "class": np.concatenate(class_vals)
    })
    
    # Perform clustering with given number of clusters
    class_arr = np.array(class_df.iloc[:, 0:2])
    kmeans_res = KMeans(n_clusters = num_clusters).fit_predict(X = class_arr)
    class_df["kmeans"] = kmeans_res
    
    # Get all metrics and return 
    aris, amis, homogs, completes, v_measures = return_metrics(
        class_arr = class_df["class"].__array__(), 
        cluster_arr = class_df["kmeans"].__array__()
    )
    return aris, amis, homogs, completes, v_measures
    
# Define function for generating perfectly separated isotropic Gaussian
# distributions 
def separated_gaussians(num_classes, num_clusters, min_class_size, max_class_size):
    # Ensure number of classes and clusters is greater than 1
    assert num_classes > 1 and num_clusters > 1
    
    # Sample class sizes from uniform distribution
    class_size_samples = [
        np.random.randint(min_class_size, max_class_size) for size in range(num_classes)
    ]
    
    # Sample centers for Gaussian distributions
    gaussian_center_interval = np.linspace(0, 1000, 100)
    gaussian_centers = [
        np.random.choice(gaussian_center_interval, num_classes, replace = False) \
            for center in range(num_classes)
    ]
    
    # Sample values from 0.1 stdev Gauassian distributions
    x_1_all = []
    x_2_all = []
    class_vals = []
    for i in range(num_classes):
        # Sample class values from Gaussian distribution
        x = np.random.normal(
            loc = gaussian_centers[i], 
            scale = 0.1,
            size=(class_size_samples[i], 2)
        )
        # Append values to list 
        x_1_all.append(x[:, 0])
        x_2_all.append(x[:, 1])        
        class_vals.append(i * np.ones(class_size_samples[i], dtype = "int"))
    
    # Concatenate all class values into single dataframe 
    class_df = pd.DataFrame({
        "x": np.concatenate(x_1_all),
        "y": np.concatenate(x_2_all),
        "class": np.concatenate(class_vals)
    })
    
    # Perform clustering with given number of clusters
    class_arr = np.array(class_df.iloc[:, 0:2])
    kmeans_res = KMeans(n_clusters = num_clusters).fit_predict(X = class_arr)
    class_df["kmeans"] = kmeans_res
    
    # Get all metrics and return 
    aris, amis, homogs, completes, v_measures = return_metrics(
        class_arr = class_df["class"], cluster_arr = class_df["kmeans"]
    )
    return aris, amis, homogs, completes, v_measures

# Define main expectation function
def main(num_trials = 1000):
    # Define necessary values of importance 
    ari_random = []
    ami_random = []
    homog_random = []
    complete_random = []
    vmeasure_random = []
    
    bal_ari_random = []
    bal_ami_random = []
    bal_homog_random = []
    bal_complete_random = []
    bal_vmeasure_random = []
    
    ari_sep = []
    ami_sep = []
    homog_sep = []
    complete_sep = []
    vmeasure_sep = []
    
    bal_ari_sep = []
    bal_ami_sep = []
    bal_homog_sep = []
    bal_complete_sep = []
    bal_vmeasure_sep = []
    
    # Define ranges for key values
    class_size_range = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    cluster_size_range = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    # Iterate over given trials and generate random data
    for i in range(num_trials):
        # Sample class and cluster numbers
        num_classes = np.random.choice(class_size_range)
        num_clusters = np.random.choice(cluster_size_range)
        
        # Get scores from generating random data 
        aris, amis, homogs, completes, v_measures = random_data(
            num_classes = num_classes,
            num_clusters = num_clusters,
            min_class_size = 50,
            max_class_size = 2000
        )
        
        # Append scores for random data
        ari_random.append(aris[0])
        ami_random.append(amis[0])
        homog_random.append(homogs[0])
        complete_random.append(completes[0])
        vmeasure_random.append(v_measures[0])
        
        bal_ari_random.append(aris[1])
        bal_ami_random.append(amis[1])
        bal_homog_random.append(homogs[1])
        bal_complete_random.append(completes[1])
        bal_vmeasure_random.append(v_measures[1])
        
    # Iterate over given trials and generate separated data
    for i in range(num_trials):
        # Sample class and cluster numbers
        num_classes = np.random.choice(class_size_range)
        num_clusters = np.random.choice(cluster_size_range)
        
        # Get scores from generating random data 
        aris, amis, homogs, completes, v_measures = separated_gaussians(
            num_classes = num_classes,
            num_clusters = num_clusters,
            min_class_size = 50,
            max_class_size = 2000
        )
        
        # Append scores for random data
        ari_sep.append(aris[0])
        ami_sep.append(amis[0])
        homog_sep.append(homogs[0])
        complete_sep.append(completes[0])
        vmeasure_sep.append(v_measures[0])
        
        bal_ari_sep.append(aris[1])
        bal_ami_sep.append(amis[1])
        bal_homog_sep.append(homogs[1])
        bal_complete_sep.append(completes[1])
        bal_vmeasure_sep.append(v_measures[1])

    # Create dataframe of results and save
    res_df = pd.DataFrame({
        "ari_random": ari_random,
        "bal_ari_random": bal_ari_random,
        "ami_random": ami_random,
        "bal_ami_random": bal_ami_random,
        "homog_random": homog_random,
        "bal_homog_random": bal_homog_random,
        "complete_random": complete_random,
        "bal_complete_random": bal_complete_random,
        "vmeasure_random": vmeasure_random,
        "bal_vmeasure_random": bal_vmeasure_random,
        "ari_sep": ari_sep,
        "bal_ari_sep": bal_ari_sep,
        "ami_sep": ami_sep,
        "bal_ami_sep": bal_ami_sep,
        "homog_sep": homog_sep,
        "bal_homog_sep": bal_homog_sep,
        "complete_sep": complete_sep,
        "bal_complete_sep": bal_complete_sep,
        "vmeasure_sep": vmeasure_sep,
        "bal_vmeasure_sep": bal_vmeasure_sep,
        "trial_num": np.arange(num_trials)
    })
    res_df.to_csv("../outs/01_expectation_results_full.tsv", sep = "\t")
    
    # Get the mean and standard deviation of each metric
    ari_random_mean = np.mean(ari_random)
    ari_random_stdev = np.std(ari_random)
    
    ami_random_mean = np.mean(ami_random)
    ami_random_stdev = np.std(ami_random)
    
    homog_random_mean = np.mean(homog_random)
    homog_random_stdev = np.std(homog_random)
    
    complete_random_mean = np.mean(complete_random)
    complete_random_stdev = np.std(complete_random)
    
    vmeasure_random_mean = np.mean(vmeasure_random)
    vmeasure_random_stdev = np.std(vmeasure_random)
    
    bal_ari_random_mean = np.mean(bal_ari_random)
    bal_ari_random_stdev = np.std(bal_ari_random)
    
    bal_ami_random_mean = np.mean(bal_ami_random)
    bal_ami_random_stdev = np.std(bal_ami_random)
    
    bal_homog_random_mean = np.mean(bal_homog_random)
    bal_homog_random_stdev = np.std(bal_homog_random)
    
    bal_complete_random_mean = np.mean(bal_complete_random)
    bal_complete_random_stdev = np.std(bal_complete_random)
    
    bal_vmeasure_random_mean = np.mean(bal_vmeasure_random)
    bal_vmeasure_random_stdev = np.std(bal_vmeasure_random)
    
    ari_sep_mean = np.mean(ari_sep)
    ari_sep_stdev = np.std(ari_sep)
    
    ami_sep_mean = np.mean(ami_sep)
    ami_sep_stdev = np.std(ami_sep)
    
    homog_sep_mean = np.mean(homog_sep)
    homog_sep_stdev = np.std(homog_sep)
    
    complete_sep_mean = np.mean(complete_sep)
    complete_sep_stdev = np.std(complete_sep)
    
    vmeasure_sep_mean = np.mean(vmeasure_sep)
    vmeasure_sep_stdev = np.std(vmeasure_sep)
    
    bal_ari_sep_mean = np.mean(bal_ari_sep)
    bal_ari_sep_stdev = np.std(bal_ari_sep)
    
    bal_ami_sep_mean = np.mean(bal_ami_sep)
    bal_ami_sep_stdev = np.std(bal_ami_sep)
    
    bal_homog_sep_mean = np.mean(bal_homog_sep)
    bal_homog_sep_stdev = np.std(bal_homog_sep)
    
    bal_complete_sep_mean = np.mean(bal_complete_sep)
    bal_complete_sep_stdev = np.std(bal_complete_sep)
    
    bal_vmeasure_sep_mean = np.mean(bal_vmeasure_sep)
    bal_vmeasure_sep_stdev = np.std(bal_vmeasure_sep)
    
    # Assert that all means are close to expected value 
    assert np.isclose(ari_random_mean, 0, atol = 0.01)
    assert np.isclose(ami_random_mean, 0, atol = 0.01)
    assert np.isclose(homog_random_mean, 0, atol = 0.01)
    assert np.isclose(complete_random_mean, 0, atol = 0.01)
    assert np.isclose(vmeasure_random_mean, 0, atol = 0.01)
    
    assert np.isclose(bal_ari_random_mean, 0, atol = 0.01)
    assert np.isclose(bal_ami_random_mean, 0, atol = 0.01)
    assert np.isclose(bal_homog_random_mean, 0, atol = 0.01)
    assert np.isclose(bal_complete_random_mean, 0, atol = 0.01)
    assert np.isclose(bal_vmeasure_random_mean, 0, atol = 0.01)
    
    print("Random clustering asserts passed")
    
    assert np.isclose(ari_sep_mean, 1, atol = 0.01)
    assert np.isclose(ami_sep_mean, 1, atol = 0.01)
    assert np.isclose(homog_sep_mean, 1, atol = 0.01)
    assert np.isclose(complete_sep_mean, 1, atol = 0.01)
    assert np.isclose(vmeasure_sep_mean, 1, atol = 0.01)
    
    assert np.isclose(bal_ari_sep_mean, 1, atol = 0.01)
    assert np.isclose(bal_ami_sep_mean, 1, atol = 0.01)
    assert np.isclose(bal_homog_sep_mean, 1, atol = 0.01)
    assert np.isclose(bal_complete_sep_mean, 1, atol = 0.01)
    assert np.isclose(bal_vmeasure_sep_mean, 1, atol = 0.01)
    
    print("Separated clustering asserts passed")
    
    # Create dataframe of mean and stdev results for each metric
    # and save 
    res_df_mean_stdev = pd.DataFrame(
        {
            "ari_random_mean": ari_random_mean,
            "ari_random_stdev": ari_random_stdev,
            "ami_random_mean": ami_random_mean,
            "ami_random_stdev": ami_random_stdev,
            "homog_random_mean": homog_random_mean,
            "homog_random_stdev": homog_random_stdev,
            "complete_random_mean": complete_random_mean,
            "complete_random_stdev": complete_random_stdev,
            "vmeasure_random_mean": vmeasure_random_mean,
            "vmeasure_random_stdev": vmeasure_random_stdev,
            "bal_ari_random_mean": bal_ari_random_mean,
            "bal_ari_random_stdev": bal_ari_random_stdev,
            "bal_ami_random_mean": bal_ami_random_mean,
            "bal_ami_random_stdev": bal_ami_random_stdev,
            "bal_homog_random_mean": bal_homog_random_mean,
            "bal_homog_random_stdev": bal_homog_random_stdev,
            "bal_complete_random_mean": bal_complete_random_mean,
            "bal_complete_random_stdev": bal_complete_random_stdev,
            "bal_vmeasure_random_mean": bal_vmeasure_random_mean,
            "bal_vmeasure_random_stdev": bal_vmeasure_random_stdev,
            "ari_sep_mean": ari_sep_mean,
            "ari_sep_stdev": ari_sep_stdev,
            "ami_sep_mean": ami_sep_mean,
            "ami_sep_stdev": ami_sep_stdev,
            "homog_sep_mean": homog_sep_mean,
            "homog_sep_stdev": homog_sep_stdev,
            "complete_sep_mean": complete_sep_mean,
            "complete_sep_stdev": complete_sep_stdev,
            "vmeasure_sep_mean": vmeasure_sep_mean,
            "vmeasure_sep_stdev": vmeasure_sep_stdev,
            "bal_ari_sep_mean": bal_ari_sep_mean,
            "bal_ari_sep_stdev": bal_ari_sep_stdev,
            "bal_ami_sep_mean": bal_ami_sep_mean,
            "bal_ami_sep_stdev": bal_ami_sep_stdev,
            "bal_homog_sep_mean": bal_homog_sep_mean,
            "bal_homog_sep_stdev": bal_homog_sep_stdev,
            "bal_complete_sep_mean": bal_complete_sep_mean,
            "bal_complete_sep_stdev": bal_complete_sep_stdev,
            "bal_vmeasure_sep_mean": bal_vmeasure_sep_mean,
            "bal_vmeasure_sep_stdev": bal_vmeasure_sep_stdev
        },
        index = [0]
    )
    res_df_mean_stdev.to_csv("../outs/01_expectation_results_mean_std.tsv", sep = "\t")
    
if __name__ == "__main__":
    # Run main script 
    main(num_trials=1000)