"""
main sap package
"""

__all__ = [
    "errors",
    "interpreter",
    "lexer",
    "nodes",
    "nodevstr",
    "parser",
    "symbols",
    "symtable",
    "tokens",
    "typing",
    "MODE",
    "FILENAME",
    "PRINT_TOKENS",
    "PRINT_EAT_STACK",
    "PRINT_TREE",
    "PRINT_SCOPE",
    "_DEV_RAISE_ERROR_STACK",
    "current_filename"
]
__version__ = "0.0.1"

# The valid modes are: "file" or "cmdline"
MODE = "file"

# Default file to execute, changed py main.py
FILENAME = "syntax_showcase.sap"

# Output booleans
PRINT_TOKENS = False
PRINT_EAT_STACK = True
PRINT_TREE = False
PRINT_SCOPE = False

_DEV_RAISE_ERROR_STACK = False

# Basic implementation of file name tracking
current_filename = "<cmdline>"