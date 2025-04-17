import os
from datetime import datetime
from src.data.build import build_dataloader
from src.modeling.network.build import build_model
from src.modeling.loss.build import build_loss_func
from src.modeling.solver.build import build_solver, build_scheduler
from src.engine.build import build_trainer
from src.evaluate.summary import TrainSummary
from src.checkpoint.save import save_model, resume_training, load_model
from src.tools.logger import make_logger


class Constructor:
    """
    construct certain experiment by the following template
    step 1: prepare dataloader
    step 2: prepare model and loss function
    step 3: select optimizer for gradient descent algorithm
    step 4: prepare trainer for typical training in pytorch manner
    """
    def __init__(self, cfg):
        """ run exp from config file

        arg:
            cfg_file: config file of an experiment
        """
        self.cfg = cfg
        self.data_loader = None
        self.model = None
        self.loss_f = None
        self.optimizer = None
        self.scheduler = None
        self.collector = None
        self.trainer = None

    def build(self):
        self.model = self.build_model()
        self.data_loader = self.build_dataloader()
        self.loss_f = self.build_loss_function()
        self.optimizer = self.build_solver()
        self.scheduler = self.build_scheduler()
        self.collector = TrainSummary()
        self.trainer = self.build_trainer()

    def build_dataloader(self):
        return build_dataloader(self.cfg)

    def build_model(self):
        return build_model(self.cfg)

    def build_loss_function(self):
        return build_loss_func(self.cfg)

    def build_solver(self):
        return build_solver(self.cfg, self.model)

    def build_scheduler(self):
        return build_scheduler(self.cfg, self.optimizer)

    def build_trainer(self):
        return build_trainer(self.cfg, self.collector, self.logger)


class ExpRunner(Constructor):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.logger, self.log_dir = make_logger(cfg.TRAIN.OUTPUT_DIR)
        self.build()
        self.logger.info(cfg)

    def before_train(self, cfg):
        # cfg = self.cfg.TRAIN
        if cfg.RESUME:
            self.model, self.optimizer, epoch = resume_training(cfg.RESUME, self.model, self.optimizer)
            if cfg.RESUME_EPOCH == "resume":
                cfg.START_EPOCH = epoch
            elif cfg.RESUME_EPOCH == "restart":  # this is for pretraining and fine-tuning schema
                cfg.START_EPOCH = 0
            else:
                raise ValueError(cfg.RESUME_EPOCH)  # phil_todo

            self.logger.info(f"resume training from {cfg.RESUME}")

        if self.cfg.SOLVER.RESET_LR:
            self.logger.info("change learning rate form [{}] to [{}]".format(
                self.optimizer.param_groups[0]["lr"],
                self.cfg.SOLVER.LR_INIT,
            ))
            self.optimizer.param_groups[0]["lr"] = self.cfg.SOLVER.LR_INIT

    def train_epochs(self, cfg):
        for epoch in range(cfg.START_EPOCH, cfg.MAX_EPOCH):
            self.trainer.train(self.data_loader["train"], self.model, self.loss_f, self.optimizer, epoch)

            if cfg.IS_VALIDATE and epoch % cfg.VALID_INTERVAL == 0:
                self.trainer.valid(self.data_loader["valid"], self.model, self.loss_f, epoch)

            self.scheduler.step()

            if self.collector.model_save and epoch % cfg.VALID_INTERVAL == 0:
                save_model(epoch, self.collector.best_valid_acc, self.model, self.optimizer, self.log_dir, cfg)
                self.collector.update_best_epoch(epoch)
            if self.cfg.TRAIN.SAVE_LAST and epoch == (cfg.MAX_EPOCH - 1):
                save_model(epoch, self.collector.best_valid_acc, self.model, self.optimizer, self.log_dir, cfg)

    def after_train(self, cfg):
        # cfg = self.cfg.TRAIN
        # self.collector.draw_epo_info(log_dir=self.log_dir)
        self.logger.info(
            "{} done, best acc: {} in :{}".format(
                datetime.strftime(datetime.now(), '%m-%d_%H-%M'),
                self.collector.best_valid_acc,
                self.collector.best_epoch,
            )
        )

    def train(self):
        cfg = self.cfg.TRAIN
        self.before_train(cfg)
        self.train_epochs(cfg)
        self.after_train(cfg)

    def test(self, weight=None):
        self.logger.info("Test only mode")
        cfg = self.cfg.TEST
        # cfg.WEIGHT = weight if weight else cfg.WEIGHT

        if cfg.WEIGHT:
            self.model = load_model(self.model, cfg.WEIGHT)
        else:
            try:
                weights = [
                    file for file in os.listdir(self.log_dir)
                    if file.endswith(".pkl") and ("last" not in file)
                ]
                weights = sorted(weights, key=lambda x: int(x[11:-4]))
                weight_file = os.path.join(self.log_dir, weights[-1])
            except IndexError:
                weight_file = os.path.join(self.log_dir, "checkpoint_last.pkl")
            self.logger.info(f"test with model {weight_file}")
            self.model = load_model(self.model, weight_file)

        acc_avg, uar = self.trainer.test(
            self.data_loader["test"], self.model
        )
        self.logger.info("acc: {} ; uar: {}".format(acc_avg, uar))
        return

    def run(self):
        self.train()
        self.test()
