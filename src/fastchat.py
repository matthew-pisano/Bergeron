import subprocess
import time
from subprocess import Popen

from src.logger import root_logger


class FastChatController:
    """Manages fastchat servers and workers for quicker model inference"""

    _workers = {}
    controller_process = None
    _enabled = True

    port_generator = (i for i in range(8000, 8005))
    """Generates unique port numbers if multiple models are being used at once"""

    @classmethod
    def is_available(cls):
        """Check if the fastchat module is available and installed

        Returns:
            Whether the fastchat module is available"""

        p = Popen(['python3', '-c', 'import fastchat'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        p.communicate()
        return p.returncode == 0

    @classmethod
    def is_enabled(cls):
        """Checks if fastchat has been manually disabled within this program. It is enabled by default

        Returns:
            Whether fastchat is enabled"""

        return cls._enabled

    @classmethod
    def enable(cls):
        """Enables fastchat manually if it has been disabled"""

        cls._enabled = True

    @classmethod
    def disable(cls):
        """Manually disables fastchat"""

        cls._enabled = False

    @classmethod
    def get_worker(cls, model_path: str):
        """Gets a particular worker for a model

        Args:
            model_path: The model to get the worker for
        Returns:
            The worker associated with the given model"""

        return cls._workers[model_path]

    @classmethod
    def open(cls, model_path: str, port: int = None):
        """Initiates the fastchat controller, server, and worker for a particular model

        Args:
            model_path: The model to use fastchat for
            port: The port to run the server on"""

        if not cls.is_available():
            raise ValueError('fastChat not available, please install fastchat to use it')
        if not cls.is_enabled():
            raise ValueError('fastChat has been disabled, please enable it to use it')

        # If there is already a fastchat worker for this model
        if model_path in cls._workers:
            return

        # Generate a port if none is provided
        if port is None:
            port = next(cls.port_generator)

        if cls.controller_process is None:
            root_logger.info("Initializing fastchat controller...")
            cls.controller_process = Popen(['python3', '-m', 'fastchat.serve.controller'], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            time.sleep(3)

        if model_path not in cls._workers:
            root_logger.info(f"Initializing {model_path} worker...")
            worker_process = Popen(['python3', '-m', 'fastchat.serve.model_worker', '--model-path', model_path], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            time.sleep(10)

            root_logger.info("Starting fastchat openai server...")
            server_process = Popen(['python3', '-m', 'fastchat.serve.openai_api_server', '--host', 'localhost', '--port', str(port)], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            time.sleep(15)

            cls._workers[model_path] = {"port": port, "worker_process": worker_process, "server_process": server_process}

        root_logger.info("Started!")

    @classmethod
    def close(cls, model_path: str = None):
        """Closes the fastchat processes for one or all models

        Args:
            model_path: The path of the model to close.  Closes all models if this is None"""

        if model_path is not None:
            root_logger.info(f"Closing {model_path} worker...")
            cls._workers[model_path]["worker_process"].terminate()
            cls._workers[model_path]["server_process"].terminate()
            cls._workers.pop(model_path)
        else:
            for path in list(cls._workers.keys()):
                cls.close(path)

            if cls.controller_process is not None:
                cls.controller_process.terminate()
                cls.controller_process = None
