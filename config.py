import tomllib

with open("config.toml", "rb") as file:
    try:
        DATA = tomllib.load(file)
    except tomllib.TOMLDecodeError as error:
        print("Could not load config file:", error)
        exit()

def get(path, data: dict = DATA):
    if isinstance(path, str):
        split_path = path.split(".")
    else:
        split_path = path
    if len(split_path) == 0:
        return data
    value = data.get(split_path[0])
    if isinstance(value, dict):
        result = get(split_path[1:], data=value)
    else:
        result = value
    return result

def getstr(path):
    ...

def getbool(path):
    ...

def getint(path):
    ...

print(get("logging.levels.DEBUG"))