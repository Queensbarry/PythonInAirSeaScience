from .error import ModuleNotRunError


def after_fit(func):
    def wrapper(self, *args, **kwargs):
        if not self._is_fit:
            raise ModuleNotRunError()
        else:
            return func(self, *args, **kwargs)
    return wrapper
