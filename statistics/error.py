class ModuleNotRunError(Exception):
    def __init__(self):
        super(ModuleNotRunError, self).__init__('You must execute `fit()` method first.')
