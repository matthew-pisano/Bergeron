import datetime

_logger_registry = {}


def get_logger(logger_name: str):
    if logger_name in _logger_registry:
        return _logger_registry[logger_name]

    raise ValueError(f"Could not find a registered logger with the name '{logger_name}'")


class Logger:
    """A custom logger class for easy utilization and verbosity setting"""

    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50
    QUIET = 60

    def __init__(self, name: str, level=INFO, auto_register=True):
        self.name = name
        self.level = level

        if auto_register:
            self.register()

    def register(self):
        if self.name in _logger_registry:
            raise ValueError(f"A logger with the name '{self.name}' already exists!  Choose a different name for the logger.")

        _logger_registry[self.name] = self

    def set_level(self, level: int):
        self.level = level

    def debug(self, *msg):
        if self.level <= self.DEBUG:
            print(f"[DEBUG @ {datetime.datetime.now().isoformat()}]", *msg)

    def info(self, *msg):
        if self.level <= self.INFO:
            print(f"[INFO @ {datetime.datetime.now().isoformat()}]", *msg)

    def warning(self, *msg):
        if self.level <= self.WARNING:
            print(f"[WARNING @ {datetime.datetime.now().isoformat()}]", *msg)

    def error(self, *msg):
        if self.level <= self.ERROR:
            print(f"[ERROR @ {datetime.datetime.now().isoformat()}]", *msg)

    def critical(self, *msg):
        if self.level <= self.CRITICAL:
            print(f"[CRITICAL @ {datetime.datetime.now().isoformat()}]", *msg)


Logger('root')
root_logger: Logger = _logger_registry['root']
