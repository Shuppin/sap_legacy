# Work-In Progress
# Eventually will be used to parse toml config files for the main program

import tomllib

config_schema = {
    'behaviour': {
        'read_mode': str,
        'logging_enabled': bool
    },
    'logging': {
        'level': str,
        'destination': str,
        'format': str,
        'datefmt': str,
        # Only a dict type since we don't care about contents
        'levels': dict
    },
    'dev': {
        'default_filename': str,
        'raise_error_stack': bool,
        'strict_semicolons': bool
    }
}

with open("config.toml", "rb") as file:
    try:
        data = tomllib.load(file)
    except tomllib.TOMLDecodeError as error:
        print("Could not load config file:", error)
        exit()

expected_categories = ['behaviour', 'logging', 'dev']

assert all(item in list(data) for item in expected_categories)

for extra in [item for item in list(data) if item not in expected_categories]:
    print("Unexpected category", extra)
