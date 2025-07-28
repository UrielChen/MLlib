#!/bin/bash

# Run all commands sequentially
python run_ts_fine_tuning.py -c config/RPBKFeatures/deep_learning/network_model_e_q2.yaml &
python run_ts_fine_tuning.py -c config/RPBKFeatures/transfer_learning/network_model_e_q2.yaml &

