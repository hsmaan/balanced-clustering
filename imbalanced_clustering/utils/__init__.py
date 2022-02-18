from .contingency import pair_confusion_matrix, contingency_matrix
from .checks import check_clusterings
from .mi import mutual_info_score, entropy, expected_mutual_information
from .avg import generalized_average
from ._emi_cython import expected_mutual_information 

check_clusterings, contingency_matrix, entropy, mutual_info_score, \
    expected_mutual_information, generalized_average