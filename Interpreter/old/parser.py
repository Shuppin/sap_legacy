from lexer import Lexer, Type, Token

def num(value):
    return {
        "type": "num",
        "value": value
    }
    
def str(value):
    return {
        "type": "str",
        "value": value
    }

def bool(value):
    return {
        "type": "bool",
        "value": value
    }

def call(func, *args):
    return {
        "type": "call",
        "func": func,
        "args": args
    }

def assign(nonterminal, terminal):
    return {
        "type": "assign",
        "op": "=",
        "left": nonterminal,
        "right": terminal
    }

def bin(op, expr1, expr2):
    return {
        "type": "binary_op",
        "op": op,
        "left": expr1,
        "right": expr2
    }

def identifier(id, value):
    return {
        "type": "variable",
        "id": id,
        "value": value
    }

def var(id, value):
    return {
        "type": "variable",
        "id": id,
        "value": value
    }

input_string = """
myvar1 = 5
myvar2 = 120 + 54 + myvar1
"""

lexer = Lexer(input_string)
tokens = lexer.lex()

syntaxTree = {}

class Parser():

    def __init__(self, tokens):
        self.tokens: list[Token] = tokens

    def expr(self):
        num = self.eat()
    

