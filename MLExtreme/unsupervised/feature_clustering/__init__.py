from .damex import (
    damex_0,
    damex,
    damex_estim_subfaces_mass,
    damex_select_epsilon_AIC
)

from .clef import (
    clef,
    find_maximal_faces,
    clef_estim_subfaces_mass,
    clef_select_kappa_AIC
    )

from .ftclust_analysis import (
    # setDistance_subface_to_matrix,
    # setDistance_subface_to_list,
    setDistance_error,
    setDistance_subfaces_data,
    list_to_dict_size
    )


from .utilities import (
    binary_large_features,
    subfaces_list_to_matrix,
    AIC_clustering
    )
