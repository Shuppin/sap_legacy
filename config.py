# TODO: Document all of this (lol)

try:
    import tomllib
except ModuleNotFoundError:
    print("Python v3.11 or higher is required to run this program")
    exit()
    
import logging

class ConfigParser:
    def __init__(self, filename="config.toml", override_logfile=False) -> None:
        
        with open(filename, "rb") as file:
            try:
                self.data: dict = tomllib.load(file)
            except tomllib.TOMLDecodeError as error:
                self._error(f"(ConfigParser) Could not load config from file: {str(error)}")
                
        formatter = logging.Formatter(self.getstr("logging.format") or "%(levelname)s:%(name)s:%(message)s", datefmt=self.getstr("logging.datefmt") or "%H:%M:%S")
        
        handler = logging.FileHandler(self.getstr("logging.destination") or "logs/runtime.log", mode="a")
        handler.setFormatter(formatter)
        
        log_level = self.getint(f"logging.levels.{self.getstr('logging.level') or 'ALL'}") or 0
        
        self._logger = logging.getLogger("config")
        self._logger.setLevel(log_level)
        self._logger.addHandler(handler)
        
        if override_logfile:
            logfile_path = self.getstr("logging.destination")
            if logfile_path is not None:
                open(logfile_path, "w").close()
                self._logger.info("Successfully cleared log file")
            else:
                self._logger.warning("Could not clear logfile, continuing anyway")
        
        self._logger.info(f"{type(self).__name__}: Successfully loaded read from file {repr(filename)}")
        
        self._verify_structure()
        self._logger.info(f"{type(self).__name__}: Verified config structure")
        
        self._logger.info(f"{type(self).__name__}: Successfully parsed config")
        
    def _error(self, message=None):
        if message is None:
            message = "(ConfigParser) An unexpected error occured while trying to parse config file"
        
        if hasattr(self, "data"):
            should_raise_error = self.getbool("dev.raise_error_stack") or False
            log_level = self.getint("logging.levels.INFO") or 0
        else:
            should_raise_error = False
            log_level = 0
        
        if hasattr(self, "logger"):
            self._logger.log(msg=f"{type(self).__name__}: Successfully constructed error message", level=log_level)
            self._logger.log(msg=f"{type(self).__name__}: Program terminating with a success state", level=log_level)

        if should_raise_error:
            raise Exception(message)
        else:
            print(message)
            exit()
        
                
    def _verify_structure(self):
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
        
        for check, path in checks:
            result = check(path)
            if result is None:
                self._logger.log(msg=f"{type(self).__name__}: Config file structure invalid, invalid option {repr(path)}", level=self.getint("logging.levels.CRITICAL") or 50)
                self._error(f"(ConfigParser) Required cofiguration option {repr(path)} does not exist/has an invalid type")

    def get(self, path: str | list, data: dict = None):
        if data is None:
            data = self.data
        
        # path should be str, however it can be a list too.
        if isinstance(path, str):
            split_path = path.split(".")
        elif isinstance(path, list):
            split_path = path
        else:
            self._logger.warning(f"(ConfigParser) get(): got unexpected type {repr(type(path))} for 'path' ")
            return None
            
        if len(split_path) == 0:
            return data
        
        value = data.get(split_path[0])
        
        if isinstance(value, dict):
            result = self.get(split_path[1:], data=value)
        else:
            result = value
            
        return result

    def getstr(self, path: str):
        result = self.get(path)
        return (result if isinstance(result, str) else None)

    def getbool(self, path: str):
        result = self.get(path)
        return (result if isinstance(result, bool) else None)

    def getint(self, path: str):
        result = self.get(path)
        return (result if isinstance(result, int) else None)
    
if __name__ == '__main__':
    config = ConfigParser("config.toml")
    print(config.get("logging.levels"))