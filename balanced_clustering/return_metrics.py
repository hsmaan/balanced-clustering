from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, \
    homogeneity_score, completeness_score, v_measure_score

from .ari import balanced_adjusted_rand_index
from .ami import balanced_adjusted_mutual_info
from .vmeasure import balanced_homogeneity, balanced_completeness, balanced_v_measure

def return_metrics(class_arr, cluster_arr, print_metrics = True):
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
    
    # If print is True, print the scores
    if print_metrics:
        print(
            "ARI imbalanced: " + str(round(ari_imbalanced, 4)) + " " + 
            "ARI balanced: " + str(round(ari_balanced, 4))
        )
        print(
            "AMI imbalanced: " + str(round(ami_imbalanced, 4)) + " " +
            "AMI balanced: " + str(round(ami_balanced, 4))
        )
        print(
            "Homogeneity imbalanced: " + str(round(homog_imbalanced, 4)) + " " +
            "Homogeneity balanced: " + str(round(homog_balanced, 4))
        )
        print(
            "Completeness imbalanced: " + str(round(complete_imbalanced, 4)) + " " +
            "Completeness balanced : " + str(round(complete_balanced, 4))
        )
        print(
            "V-measure imbalanced: " + str(round(v_measure_imbalanced, 4)) + " " +
            "V-measure balanced: " + str(round(v_measure_balanced, 4))
        )
    
    # Return paired balanced imbalanced scores
    return (ari_imbalanced, ari_balanced), (ami_imbalanced, ami_balanced), \
        (homog_imbalanced, homog_balanced), (complete_imbalanced, complete_balanced), \
        (v_measure_imbalanced, v_measure_balanced)