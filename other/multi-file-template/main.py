import os
import sys

from sap.typing import SemanticAnalyser
from sap.interpreter import Interpreter
from sap.parser import Parser

from sap import FILENAME, PRINT_TREE

def get_filename():

    if len(sys.argv) == 1:
        # Execute in command-line mode
        pass

    elif len(sys.argv) == 2:
        path = sys.argv[1]
        if os.path.isfile(path):
            return path
        else:
            return None

    else:
        return None

def process_code(source_code: str):
    parser = Parser(source_code)
    semantic_analyser = SemanticAnalyser(source_code)
    interpreter = Interpreter()

    tree = parser.parse()

    if PRINT_TREE:
        print(tree)

    semantic_analyser.analyse(tree)
    interpreter.interpret(tree)

    print()
    print("Global vars (doesn't account for functions):")
    print(interpreter.global_scope)
    print()

def main(filename: str):
    file = open(filename, "r")
    soruce_code = file.read()

    if not soruce_code:
        return
    
    process_code(soruce_code)

if __name__ == "__main__":
    filename = get_filename()
    if filename is not None:
        FILENAME = filename
    print(FILENAME)
    main(FILENAME)