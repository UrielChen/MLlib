from .core import Base


class NewRunner(Base):
    """base trainer for bi-modal input"""
    def __init__(self, cfg):
        self.cfg = cfg
        self._hooks = []

    def register_hook(self, hook_obj):
        self._hooks.append(hook_obj)

    def before_train(self, *args, **kwargs):
        for hook in self._hooks:
            hook.before_train(*args, **kwargs)

        # self.model.train()

    def after_train(self, *args, **kwargs):
        for hook in self._hooks:
            hook.after_train(*args, **kwargs)

    def before_epoch(self, *args, **kwargs):
        for hook in self._hooks:
            hook.before_epoch(*args, **kwargs)

        # lr = self.optimizer.param_groups[0]['lr']
        # self.logger.info(f"Training: learning rate:{lr}")

    def after_epoch(self, epoch, *args, **kwargs):
        for hook in self._hooks:
            hook.after_epoch(*args, **kwargs)

        # self.scheduler.step()
        #
        # if self.collector.model_save and epoch % self.cfg.VALID_INTERVAL == 0:
        #     save_model(epoch, self.collector.best_valid_acc, self.model, self.optimizer, self.log_dir, self.cfg)
        #     self.collector.update_best_epoch(epoch)
        # if self.cfg.TRAIN.SAVE_LAST and epoch == (self.cfg.MAX_EPOCH - 1):
        #     save_model(epoch, self.collector.best_valid_acc, self.model, self.optimizer, self.log_dir, self.cfg)

    def before_batch(self, data, *args, **kwargs):
        for hook in self._hooks:
            hook.before_batch(*args, **kwargs)

        inputs, label = self.data_fmt(data)
        return inputs, label

    def after_batch(self, data, *args, **kwargs):
        for hook in self._hooks:
            hook.after_batch(*args, **kwargs)

    def run_epoch(self):
        for i, data in enumerate(self.train_data_loader):
            self.before_batch()
            self.run_batch(data)
            self.after_batch()

    def run_batch(self, data):
        inputs, labels = data
        outputs = self.model(*inputs)
        loss = self.loss_func(outputs.cpu(), labels.cpu())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def run(self):
        self.before_train()
        for epoch in range(self.cfg.START_EPOCH, self.cfg.MAX_EPOCH):
            self.before_epoch(epoch)
            self.run_epoch(epoch)
            self.after_epoch(epoch)
        self.after_train()
