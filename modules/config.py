"""
Module which is responsible for loading, and parsing the config file
so different parts of the program can access the same global variables.
"""
# tomllib is new in Python 3.11
# so just ensure the user is running
# on a valid version
try:
    import tomllib
except ModuleNotFoundError:
    print("Python v3.11 or higher is required to run this program")
    exit()
    
from os.path    import exists
from os         import mkdir

import logging

class ConfigParser:
    """
    Loads config file and verifies it's structure
    """
    def __init__(self, filename="config.toml", override_logfile=False) -> None:
        
        with open(filename, "rb") as file:
            try:
                self.data: dict = tomllib.load(file)
            except tomllib.TOMLDecodeError as error:
                self._error(f"(ConfigParser) Could not load config from file: {str(error)}")

        # Create formatter object using value(s) from config file if valid, else use a default value
        # Formatter defines how each line will look in the config file
        formatter = logging.Formatter(self.getstr("logging.format") or "%(levelname)s:%(name)s:%(message)s", datefmt=self.getstr("logging.datefmt") or "%H:%M:%S")
        
        if not exists("logs"):
            mkdir("logs")

        # Handler defines which file to write to and how to write to it
        handler = logging.FileHandler("logs/" + (self.getstr("logging.destination") or "runtime.log"), mode="a")
        handler.setFormatter(formatter)
        
        # Get logging level from config file if valid, else use value 0 (all messages)
        log_level = self.getint(f"logging.levels.{self.getstr('logging.level') or 'ALL'}") or 0
        
        # Create and setup logging object
        self._logger = logging.getLogger("config")
        self._logger.setLevel(log_level)
        self._logger.addHandler(handler)
        
        # This is a workaround to allow multiple programs to
        # log to the same file, since this program is the first
        # to start logging, it is responsible for clearing the
        # old log file
        if override_logfile:
            logfile_path = "logs/" + self.getstr("logging.destination")
            if logfile_path is not None:
                open(logfile_path, "w").close()
                self._logger.log(self.getint("logging.levels.INFO"), "Successfully cleared log file")
            else:
                self._logger.log(self.getint("logging.levels.WARNING"), "Could not clear logfile, continuing anyway")
        
        self._logger.log(self.getint("logging.levels.INFO"), f"{type(self).__name__}: Successfully read from file {repr(filename)}")
        
        self._verify_structure()

        self._logger.log(self.getint("logging.levels.INFO"), f"{type(self).__name__}: Verified config structure")
        
        self._logger.log(self.getint("logging.levels.INFO"), f"{type(self).__name__}: Successfully parsed config")
        
    def _error(self, message=None):
        # Set default error message
        if message is None:
            message = "(ConfigParser) An unexpected error occured while trying to parse config file"
        
        # If self._error() was called before self.data was loaded,
        # we want to use hard-coded values instead
        if hasattr(self, "data"):
            # The 'or' statements ensure a valid value is assigned,
            # even if the get functions return None
            should_raise_error = self.getbool("dev.raise_error_stack")
            if should_raise_error is None:
                should_raise_error = False
            log_level = self.getint("logging.levels.INFO") or 0
        else:
            should_raise_error = False
            log_level = 0
        
        # Only attempt to log of self.logger exists
        if hasattr(self, "logger"):
            self._logger.log(msg=f"{type(self).__name__}: Successfully constructed error message", level=log_level)
            self._logger.log(msg=f"{type(self).__name__}: Program terminating with a success state", level=log_level)

        # Print error message
        if should_raise_error:
            raise Exception(message)
        else:
            print(message)
            exit()

    def _verify_structure(self):
        """
        Hard-coded checks to verify config file follows a specified structure
        """
        checks = (
            (self.getbool, "behaviour.logging_enabled"),
            (self.getstr,  "behaviour.read_mode"),
            
            (self.getstr,  "logging.level"),
            (self.getstr,  "logging.destination"),
            (self.getstr,  "logging.format"),
            (self.getstr,  "logging.datefmt"),
            
            (self.getint,  "logging.levels.CRITICAL"),
            (self.getint,  "logging.levels.INFO"),
            (self.getint,  "logging.levels.DEBUG"),
            (self.getint,  "logging.levels.VERBOSE"),
            (self.getint,  "logging.levels.HIGHLY_VERBOSE"),
            (self.getint,  "logging.levels.EAT_STACK"),
            (self.getint,  "logging.levels.ALL"),
            
            (self.getstr,  "dev.default_filename"),
            (self.getbool, "dev.raise_error_stack"),
            (self.getbool, "dev.strict_semicolons"),
        )
        
        for get_function, path in checks:
            result = get_function(path)
            if result is None:
                self._logger.log(msg=f"{type(self).__name__}: Config file structure invalid, invalid option {repr(path)}", level=self.getint("logging.levels.CRITICAL") or 50)
                self._error(f"(ConfigParser) Required cofiguration option {repr(path)} does not exist/has an invalid type")

    def get(self, path: str | list, data: dict = None):
        """
        Takes in a path (format: "category.value") and returns the object
        at that location, if path does not exist, it returns None
        """
        # If no data was provided, use the default file
        if data is None:
            data = self.data
        
        # Convert path into expected format
        if isinstance(path, str):
            split_path = path.split(".")
        elif isinstance(path, list):
            split_path = path
        else:
            self._logger.log(self.getint("logging.levels.WARNING"), f"(ConfigParser) get(): got unexpected type {repr(type(path))} for 'path' ")
            return None
        
        # If the path consists of only a single word, then we have reached the end
        if len(split_path) == 0:
            return data
        
        # Get the first value in the the path from the provided data
        value = data.get(split_path[0])
        
        # If it's another dictionary, recursively search down that
        if isinstance(value, dict):
            result = self.get(split_path[1:], data=value)
        # Else we just want to return the current value since we can't search down it
        else:
            result = value
            
        return result

    def getstr(self, path: str):
        """
        Takes in a path (format: "category.value") and returns the object at that
        location if that object exists and is a str type, else, it returns None
        """
        result = self.get(path)
        return (result if isinstance(result, str) else None)

    def getbool(self, path: str):
        """
        Takes in a path (format: "category.value") and returns the object at that
        location if that object exists and is a bool type, else, it returns None
        """
        result = self.get(path)
        return (result if isinstance(result, bool) else None)

    def getint(self, path: str):
        """
        Takes in a path (format: "category.value") and returns the object at that
        location if that object exists and is a int type, else, it returns None
        """
        result = self.get(path)
        return (result if isinstance(result, int) else None)


if __name__ == '__main__':
    config = ConfigParser("config.toml")
    print(config.get("logging.levels"))