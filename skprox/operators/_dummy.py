class Dummy:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, x):
        return 0

    def prox(self, x, tau):
        return x
