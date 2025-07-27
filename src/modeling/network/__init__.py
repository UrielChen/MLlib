from .alexnet import alexnet_1d
from .mlp import mlp_model
from .phil_nn_1 import phil_nn_1
from .phil_nn_2 import (
    phil_nn_2, phil_rp_shallow_network_1, phil_rp_shallow_network_2, phil_rp_shallow_network_3,
    phil_rp_model_a_shallow_wide, phil_rp_model_b_moderate_depth, phil_rp_model_c_deep_narrow,
    phil_rp_model_d_encoder_decoder, phil_rp_model_f_wide_deep,
    phil_rp_model_g_wide_shallow, phil_rp_model_h_deep_bottleneck, phil_rp_model_i_skip_projection,
    phil_rp_model_j_deep_narrow_but_gradual, phil_rp_model_k_stack_residual_and_dense,
    phil_rp_model_l_symmetric_autoencoder
)
from .tf_asset_pricing import phil_rp_model_e_wide_then_sparse, phil_rp_model_e_moderate_wide_then_sparse, phil_rp_model_e_huge_wide_then_sparse
from .renet18_1d import resnet18_1d
from .new_models import new_model_a_bottleneck_deep_wide, new_model_b_bottleneck_shallow, new_model_c_bottleneck_deep_narrow
