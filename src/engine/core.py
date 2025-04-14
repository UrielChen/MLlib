

class Base:
    def __init__(self):
        self.model = None

    def before_train(self, *args, **kwargs):
        pass

    def after_train(self, *args, **kwargs):
        pass

    def before_epoch(self, *args, **kwargs):
        pass

    def after_epoch(self, *args, **kwargs):
        pass

    def before_batch(self, *args, **kwargs):
        pass

    def after_batch(self, *args, **kwargs):
        pass


class Hooks:

    def before_train(self, ):
        pass

    def after_train(self):
        pass

    def before_epoch(self):
        pass

    def after_epoch(self):
        pass

    def before_batch(self):
        pass

    def after_batch(self):
        pass
