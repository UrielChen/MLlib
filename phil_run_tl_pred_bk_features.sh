#!/bin/bash

# Run all commands sequentially
python run_ts_fine_tuning.py -c config/FactorModelMainBKFeatures/tl_pred/tl_pred_network_phil_rp_shallow_network_1_data_49_1M_excess_return_n_5.yaml &
python run_ts_fine_tuning.py -c config/FactorModelMainBKFeatures/pure_dl/pure_dl_network_phil_rp_shallow_network_1_data_49_1M_excess_return_n_5.yaml &


