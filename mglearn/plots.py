from .plot_2d_separator import plot_2d_separator
from .plot_2d_separator import plot_2d_classification

from .plot_helpers import cm2, cm3

from .plot_knn_classification import plot_knn_classification
from .plot_knn_regression import plot_knn_regression

from .plot_linear_regression import plot_linear_regression_wave

from .plot_ridge import plot_ridge_n_samples

from .plot_linear_svc_regularization import plot_linear_svc_regularization

from .plot_animal_tree import plot_animal_tree

from .plot_tree_nonmonotonous import plot_tree_not_monotone

from .plot_interactive_tree import plot_tree_partition

from .plot_rbf_svm_parameters import plot_svm

from .plot_nn_graphs import (
    plot_logistic_regression_graph,
    plot_single_hidden_layer_graph,
    plot_two_hidden_layer_graph,
)

from .plot_scaling import plot_scaling

from .plot_pca import (
    plot_pca_illustration,
    plot_pca_whitening,
    plot_pca_faces,
)

from .plot_nmf import (
    plot_nmf_illustration,
    plot_nmf_faces
)

from .plot_kmeans import (
    plot_kmeans_algorithm,
    plot_kmeans_boundaries,
    plot_kmeans_faces
)

from .plot_agglomerative import (
    plot_agglomerative_algorithm,
    plot_agglomerative,
)

from .plot_dbscan import plot_dbscan

from .plot_cross_validation import (
    plot_cross_validation,
    plot_stratified_cross_validation,
    plot_shuffle_split,
    plot_threefold_split,
    plot_label_kfold,
)

from .plot_grid_search import (
    plot_cross_val_selection,
    plot_grid_search_overview,
)

from .plot_metrics import (
    plot_confusion_matrix_illustration,
    plot_binary_confusion_matrix,
    plot_decision_threshold,
)
