import logging
import sys

logger = logging.getLogger("Bergeron")
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(logging.Formatter('[%(levelname)s %(name)s @ %(asctime)s] %(message)s', "%H:%M:%S"))
logger.addHandler(console_handler)
