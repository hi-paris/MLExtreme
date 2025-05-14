from .dataset_generation import (
    gen_rv_dirimix,
    gen_classif_data_diriClasses,
    gen_target_CovariateRV,
    gen_rv_functional_data,
    gen_rv_functional_data_gaussianNoise,
    gen_subfaces,
    normalize_param_dirimix,
    gen_dirichlet,
    pdf_dirichlet,
    pdf_dirimix,
    gen_dirimix,
    plot_pdf_dirimix_2D,
    plot_pdf_dirimix_3D,
    # gen_PositiveStable,
    gen_multilog,
    transform_target_lin,
    inv_transform_target_lin,
    transform_target_nonlin,
    inv_transform_target_nonlin
 
)

from .EVT_basics import (
    round_signif,
    hill_estimator,
    rank_transform,
    rank_transform_test,
    test_indep_radius_rest,
    plot_indep_radius_rest
)
