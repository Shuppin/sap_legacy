"""
Module containing all the built-in functions

Don't import individual functions import `mapping` once and create the symbol table from there
"""
import sys

if not __name__.startswith("modules."):
    from builtin_types import *
else:
    from modules.builtin_types import *

def _console_out(*args: Type) -> None:
    message = ""
    for arg in args:
        message += arg.to(String).value.replace("\\n", "\n") + " "
    
    sys.stdout.write(message)
    
def _console_out_line(*args: Type) -> None:
    message = ""
    for arg in args:
        message += arg.to(String).value.replace("\\n", "\n") + " "
    
    sys.stdout.write(message + "\n")
        
mapping = {
    "print": {
        "callable": _console_out,
        "arg_spec": []
    },
    "println": {
        "callable": _console_out_line,
        "arg_spec": []
    }
}