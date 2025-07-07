#! /usr/bin/env python
from src.tools.common import parse_args, default_setup
from src.config.config import get_cfg
from src.engine.ts_fine_tuning_runner import TSFineTuningRunner


def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # cfg.freeze()
    default_setup(args, cfg)
    return cfg


def main():
    args = parse_args()
    cfg = setup(args)
    runner = TSFineTuningRunner(cfg)
    if args.test_only:
        runner.run(test_only=True)
    if args.train_only:
        return runner.run(train_only=True)
    runner.run()


if __name__ == "__main__":
    import sys
    sys.argv = ["run.py", "-c", "config/phil_rp/tl_pred_network_phil_nn_2_monthly_k_5.yaml"]
    main()
