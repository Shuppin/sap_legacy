from __future__ import annotations

import sys
import os

from inspect import currentframe, getframeinfo
from collections import defaultdict
from typing import Any
from enum import Enum

# Constants

# The valid modes are: "file" or "cmdline"
MODE = "file"

DEFAULT_FILENAME = "compounds.sap"

PRINT_TOKENS = False
PRINT_EAT_STACK = False
PRINT_TREE = False
PRINT_SCOPE = False

_DEV_RAISE_ERROR_STACK = False

# Basic implementation of file name tracking
current_filename = "<cmdline>"


###########################################
#                                         #
#   Enums                                 #
#                                         #
###########################################

class TokenType(Enum):
    """
    Enum class to hold all the token types for SAP language
    """
    # symbols
    MULT            = '*'
    INTEGER_DIV     = '//'  # Currently not in use, may be removed in future
    FLOAT_DIV       = '/'
    PLUS            = '+'
    MINUS           = '-'
    RETURNS_OP      = '->'
    LPAREN          = '('
    RPAREN          = ')'
    ASSIGN          = '='
    SEMI            = ';'
    COLON           = ':'
    COMMA           = ','
    BEGIN           = '{'
    END             = '}'
    # reserved keywords
    INTEGER         = 'int'
    FLOAT           = 'float'
    DEFINITION      = 'def'
    # dynamic token types
    INTEGER_CONST   = 'INTEGER_CONST'
    FLOAT_CONST     = 'FLOAT_CONST'
    IDENTIFIER      = 'IDENTIFIER'
    EOF             = 'EOF'


class ErrorCode(Enum):
    SYNTAX_ERROR        = "SyntaxError"
    NAME_ERROR          = "NameError"
    TYPE_ERROR          = "TypeError"


###########################################
#                                         #
#   Data classes                          #
#                                         #
###########################################

# Temporary
class GlobalScope(dict):
    def __init_subclass__(cls):
        return super().__init_subclass__()

    def __str__(self):
        text = []

        for key, value in sorted(self.items(), key=lambda x: x[1][0], reverse=True):
            text.append("  <" + str(value[0]) + "> " + str(key) + " = " + str(value[1]))

        return "\n".join(text)


class Token:
    """
    Token data class

    Simple data class to hold information about a token
    """
    def __init__(self, datatype: TokenType, id: Any, lineno: int, linecol: int, startcol: int | None = None):
        self.type: TokenType = datatype
        self.id: Any = id
        self.lineno: int = lineno
        self.linecol: int = linecol
        self.startcol: int | None = startcol

    def __str__(self) -> str:
        return f"Token[type = {self.type}, id = '{self.id}', position = <{self.lineno}:{self.linecol}>]"

    def __repr__(self) -> str:
        return repr(self.__str__())


###########################################
#                                         #
#  Error handler                          #
#                                         #
###########################################

class Error(Exception):
    def __init__(self, error_code: ErrorCode, message: str, token:Token=None, position:list[int]=None, surrounding_lines:list[str]=None):
        self.error_code: ErrorCode = error_code
        self.message: str = f'({self.__class__.__name__[:-5]}) {self.error_code.value}: {message}'
        self.token: Token | None = token
        self.surrounding_lines: list[str] | None = surrounding_lines
        self.lineno: int
        self.linecol: int
        if token is not None:
            self.lineno = token.lineno
            self.linecol = token.linecol
        elif position is not None:
            self.lineno = position[0]
            self.linecol = position[1]
        else:
            raise ValueError("Not enough information parsed into Error()")

    def trigger(self):
        if self.surrounding_lines is not None and self.token is not None and self.token.startcol is not None:
            error_message = [
                (f'File "{current_filename}", position <{self.lineno}:{self.linecol}>\n'),
                (f" │ {' '*(len(str(self.lineno+2)) - len(str(self.lineno-3)))}{self.lineno-3} │ {self.surrounding_lines[self.lineno-4]}\n" if (self.lineno-4) >= 0 else f""),
                (f" │ {' '*(len(str(self.lineno+2)) - len(str(self.lineno-2)))}{self.lineno-2} │ {self.surrounding_lines[self.lineno-3]}\n" if (self.lineno-3) >= 0 else f""),
                (f" │ {' '*(len(str(self.lineno+2)) - len(str(self.lineno-1)))}{self.lineno-1} │ {self.surrounding_lines[self.lineno-2]}\n" if (self.lineno-2) >= 0 else f""),
                (f" │ {' '*(len(str(self.lineno+2)) - len(str(self.lineno  )))}{self.lineno  } │ {self.surrounding_lines[self.lineno-1]}\n"),
                (f"   {' '*len(str(self.lineno+2))}  {' '*(self.token.startcol-1)} {'~'*(self.linecol-self.token.startcol)}\n"),
                (self.message)
                ]
            error_message = "".join(error_message)

        elif self.surrounding_lines is not None:
            error_message = [
                (f'File "{current_filename}", position <{self.lineno}:{self.linecol}>\n'),
                (f" │ {' '*(len(str(self.lineno+2)) - len(str(self.lineno-3)))}{self.lineno-3} │ {self.surrounding_lines[self.lineno-4]}\n" if (self.lineno-4) >= 0 else f""),
                (f" │ {' '*(len(str(self.lineno+2)) - len(str(self.lineno-2)))}{self.lineno-2} │ {self.surrounding_lines[self.lineno-3]}\n" if (self.lineno-3) >= 0 else f""),
                (f" │ {' '*(len(str(self.lineno+2)) - len(str(self.lineno-1)))}{self.lineno-1} │ {self.surrounding_lines[self.lineno-2]}\n" if (self.lineno-2) >= 0 else f""),
                (f" │ {' '*(len(str(self.lineno+2)) - len(str(self.lineno  )))}{self.lineno  } │ {self.surrounding_lines[self.lineno-1]}\n"),
                (f"   {' '*len(str(self.lineno+2))}  {' '*(self.linecol-1)} ^\n"),
                (self.message)
                ]
            error_message = "".join(error_message)
        else:
            error_message = (f'File "{current_filename}", position <{self.lineno}:{self.linecol}>\n'
                             f'   <error fetching lines>\n'
                             f'{self.message}')

        if _DEV_RAISE_ERROR_STACK:
            print(error_message)
            raise self
        else:
            print(error_message)
            exit()


class LexerError(Error):
    pass


class ParserError(Error):
    pass


class SemanticAnalyserError(Error):
    pass


###########################################
#                                         #
#  Node definitions                       #
#                                         #
###########################################

class Node:
    """
    Node base class

    Represents a node on an abstract syntax tree
    """
    def __str__(self) -> str:
        return str(self._print_children(self.__dict__))

    def _print_children(self, tree_dict: dict, depth: int = 1) -> str:
        """
        Recursive function to neatly print a node object and it's children

        Nodes look like:
        ```
        BinOp(
            left: Num(...),
            ...
        ),
        ```

        Lists look like:
        ```
        list_name: [
            ...
        ]
        ```

        Everything else looks like:
        ```
        object_name: object_value,
        ```

        """
        text = ""

        if depth == 1:
            text += self.__class__.__name__ + "(\n"

        # Looks ugly, will always look ugly, but the output looks great!
        for key, value in tree_dict.items():
            if isinstance(value, Node):
                text += "   " * depth + str(key) + ": " + str(value.__class__.__name__) + "(\n"
                text += self._print_children(value.__dict__, depth + 1)
                text += "   " * depth + "),\n"
            elif isinstance(value, list):
                text += "   " * depth + str(key) + ": [\n"
                for node in value:
                    if isinstance(node, Node):
                        text += "   " * (depth + 1) + node.__class__.__name__ + "(\n"
                        text += self._print_children(node.__dict__, depth + 2)
                        text += "   " * (depth + 1) + "),\n"
                    else:
                        raise TypeError(f"Cannot print type '{type(node)}'")
                text += "   " * depth + "],\n"
            else:
                text += ("   " * depth + str(key) + ": " + str(value) + ",\n")

        if depth == 1:
            text += ")"

        return text


"""
INTERIOR NODES
(in order of precedence)
"""


class Program(Node):
    def __init__(self):
        self.statements: list[Node] = []


class Compound(Node):
    def __init__(self):
        self.children: list[Node] = []


class VarDecl(Node):
    def __init__(self, type_node, var_node, assign_op=None, expr_node=None):
        self.type_node: TypeNode = type_node
        self.var_node: Var = var_node
        self.assign_op: Token | None = assign_op
        self.expr_node: Node | None = expr_node


class ProcedureDecl(Node):
    def __init__(self, procedure_var, params, compound_node, return_type=None):
        self.procedure_var: Var = procedure_var
        self.params: list[Param] = params
        self.return_type: TypeNode | None = return_type
        self.compound_node: Compound = compound_node


class AssignOp(Node):
    def __init__(self, left, op, right):
        self.left: Token = left
        self.op: Token = op
        self.right: Node = right


class UnaryOp(Node):
    def __init__(self, op, expr):
        self.op: Token = op
        self.expr: Node = expr


class BinOp(Node):
    def __init__(self, left, op, right):
        self.left: Node = left
        self.op: Token = op
        self.right: Node = right


class Param(Node):
    def __init__(self, var_node, type_node):
        self.var_node: Var = var_node
        self.type_node: TypeNode = type_node


"""
LEAF NODES
(in order of precedence)
"""


class TypeNode(Node):
    def __init__(self, token):
        self.token: Token = token
        self.id = self.token.type.name


class Var(Node):
    def __init__(self, token):
        self.token: Token = token
        self.id = self.token.id


class Num(Node):
    def __init__(self, token):
        self.token: Token = token
        self.id: int | str | None = self.token.id


class NoOp(Node):
    pass


###########################################
#                                         #
#   Symbol table code                     #
#                                         #
###########################################

class Symbol:
    """
    Symbol base class
    """
    def __init__(self, name, datatype=None):
        self.name: str = name
        self.type: BuiltinSymbol | None = datatype

    def __str__(self):
        return self.name


class BuiltinSymbol(Symbol):
    """
    Symbol which represents built in types
    """
    def __init__(self, name):
        super().__init__(name)

    def __str__(self):
        return f"<builtin> {self.name}"


class VarSymbol(Symbol):
    """
    Symbol which represents user-defined variables
    """
    def __init__(self, name, datatype):
        super().__init__(name, datatype)

    def __str__(self):
        return f"<variable> (id: '{self.name}', type: '{self.type.name}')"


class ProcedureSymbol(Symbol):
    """
    Symbol which represents procedure declarations
    """
    def __init__(self, name, params=[]):
        super().__init__(name)
        self.params: list[Param] = params

    def __str__(self):
        if len(self.params) == 0:
            return f"<procedure> (id: '{self.name}', parameters: <no params>)"
        else:
            # Okay, yes this is horrendous don't @me
            return (f"<procedure> (id: '{self.name}', parameters: "
                    f"{', '.join(list(map(lambda param: f'({param.var_node.id}, <{param.type_node.id}>)', self.params)))})")


class SymbolTable:
    """
    Class to store all the program symbols
    """
    def __init__(self, scope_name, scope_level, parent_scope=None):
        self._symbols: dict[str, Symbol] = {}
        self.scope_name: str = scope_name
        self.scope_level: int = scope_level
        self.parent_scope: SymbolTable | None = parent_scope

        if self.scope_level == 0:
            self.define(BuiltinSymbol("INTEGER"))
            self.define(BuiltinSymbol("FLOAT"))

    def __str__(self):
        # Add header information
        text = "\nSCOPE (SCOPED SYMBOL TABLE):\n"
        text += f"Scope name    : {self.scope_name}\n"
        text += f"Scope level   : {self.scope_level}\n"
        text += f"Parent scope  : {self.parent_scope.scope_name if self.parent_scope else '<none>'}\n\n"
        text += "Scope symbol table contents\n"
        text += "---------------------------\n\n"

        # Organise contents of symbol table by symbol type.
        # This excludes the built-in symbol type, which is always printed first.
        symbols = defaultdict(list)
        for _, val in sorted(self._symbols.items()):
            symbols[val.__class__.__name__].append(val)

        symbols: dict[str, list] = dict(symbols)

        # At this point `symbols` is a dictionary (dict[str, list])
        # containing the name of each (present) symbol type and
        # a list of all the objects which are that type.

        # Show the built-in symbols first
        builtin_types: list[BuiltinSymbol] = symbols.get(BuiltinSymbol.__name__)
        if builtin_types is not None:
            for builtin_type in builtin_types:
                text += "  " + str(builtin_type) + "\n"
            text += "\n"
            # Remove it from `symbols` so it is not shown again
            del symbols[BuiltinSymbol.__name__]

        # Now show the remaining symbols
        for _, symbols in symbols.items():
            for symbol in symbols:
                text += "  " + str(symbol) + "\n"
            text += "\n"

        # Simple code to add bars around the top and bottom of the table output string.
        # The width of the bar is dependent on the longest line in the string.
        text = text.split("\n")
        del text[-1]
        longest_string_length = len(max(text, key=len))
        text.insert(2, "=" * (longest_string_length + 1))
        text.append("=" * (longest_string_length + 1) + "\n")
        text = "\n".join(text)

        return text

    def define(self, symbol: Symbol):
        """
        Adds a symbol to the symbol table
        """
        self._symbols[symbol.name] = symbol

    def lookup(self, symbol_name: str, search_parent_scopes: bool = True) -> Symbol | None:
        """
        Will search for the given symbol name in `self._symbols` and
        then it will search its parent scopes.

        `search_parent_scopes` (bool): Determines whether the function will search in parent scopes. 
        """
        symbol = self._symbols.get(symbol_name)
        if symbol is not None:
            return symbol

        # Recursively search up the scopes to find symbols
        if self.parent_scope is not None and search_parent_scopes:
            return self.parent_scope.lookup(symbol_name)
        else:
            return None


###########################################
#                                         #
#   Node visitor code                     #
#                                         #
###########################################

class NodeVisitor:
    """
    NodeVisitor base class

    Base class for all classes which visit/walk through a syntax tree
    """

    def visit(self, node: Node) -> Any:
        """
        Executes the visit function associated with the given node
        """
        method_name = "visit_" + type(node).__name__
        visitor = getattr(self, method_name, self.default_visit)
        return visitor(node)

    def default_visit(self, node: Node):
        """
        Code gets executed when there is no `visit_(...)` function associated with a given `Node` object.
        """
        raise Exception(f"{self.__class__.__name__} :: No visit_{type(node).__name__} method")


###########################################
#                                         #
#   Lexer code                            #
#                                         #
###########################################

class Lexer:
    """
    Main lexer class

    The lexer is responsible for the tokenisation of the code.
    In other words, it splits the code up into its individual components.

    For example given the code:
    `2 + 2`

    The lexer will generate:
    ```
    Token[type = type.INTEGER_CONST, id = '2', start_pos = 0]
    Token[type = type.PLUS, id = '+', start_pos = 2]
    Token[type = type.INTEGER_CONST, id = '2', start_pos = 4]
    ```
    """
    def __init__(self, text):
        self.text: str = text
        self.text_lines: list[str] = self.text.split('\n')
        self.pos: int = 0
        self.lineno: int = 1
        self.linecol: int = 0
        self.current_char: str | None = self.text[self.pos]

        self.RESERVED_KEYWORDS: dict = {
            'int': TokenType.INTEGER,
            'float': TokenType.FLOAT,
            'def': TokenType.DEFINITION
        }

    # Utility functions

    def error(self):
        error = LexerError(ErrorCode.SYNTAX_ERROR, f"could not tokenise '{self.current_char}'", position=[self.lineno, self.linecol], surrounding_lines=self.text_lines)
        error.trigger()

    def advance(self):
        """Advance `self.pos` and set `self.current_char`"""

        if self.current_char == "\n":
            self.lineno += 1
            self.linecol = 0

        self.pos += 1
        if self.pos > len(self.text) - 1:
            self.current_char = None
        else:
            self.linecol += 1
            self.current_char = self.text[self.pos]
            
    def peek(self) -> None | str:
        """Peeks at the next character in the code and returns it"""
        peek_pos = self.pos + 1
        if peek_pos > len(self.text) - 1:
            return None
        else:
            return self.text[peek_pos]

    # Lexer functions

    def skip_whitespace(self):
        """Advances `self.pos` until a non-whitespace character has been reached"""
        while self.current_char is not None and self.current_char == " ":
            self.advance()

    def skip_multi_comment(self):
        """Advances `self.pos` until a comment terminator (*/) has been reached"""
        while not (self.current_char == "*" and self.peek() == "/"):
                self.advance()
        # Advance twice more to skip over the final "*/"
        self.advance()
        self.advance()

    def skip_comment(self):
        """Advances `self.pos` until a newline has been reached"""
        while self.current_char is not None and not self.current_char == "\n":
            self.advance()
        self.advance()

    def identifier(self) -> Token:
        """Creates and returns an identifier token"""
        result = ""
        start_pos = self.linecol
        while self.current_char is not None and (self.current_char.isalnum() or self.current_char == "_"):
            result += self.current_char
            self.advance()

        # Checks if `result` is a keyword or not and returns the appropriate type.
        # Gets the type associated with `result if applicable, else default to `type.IDENTIFIER`
        token_type = self.RESERVED_KEYWORDS.get(result, TokenType.IDENTIFIER)

        token = Token(token_type, result, self.lineno, self.linecol, startcol=start_pos)

        return token

    def number(self) -> Token:
        """Consumes a number from the input code and returns it"""
        number = ''
        start_pos = self.linecol
        while self.current_char is not None and self.current_char.isdigit():
            number += self.current_char
            self.advance()

        if self.current_char == ".":
            number += self.current_char
            self.advance()

            while self.current_char is not None and self.current_char.isdigit():
                number += self.current_char
                self.advance()

            token = Token(TokenType.FLOAT_CONST, float(number), self.lineno, self.linecol, startcol=start_pos)

        else:
            token = Token(TokenType.INTEGER_CONST, int(number), self.lineno, self.linecol, startcol=start_pos)

        return token

    def get_next_token(self) -> Token:
        """
        Responsible for breaking down and extracting 
        tokens out of code.
        """
        while self.current_char is not None:

            # Ignored characters

            if self.current_char == "\n":
                self.advance()
                continue

            elif self.current_char == " ":
                self.skip_whitespace()
                continue

            elif self.current_char == "/" and self.peek() == "*":
                self.skip_multi_comment()
                continue

            elif self.current_char == "/" and self.peek() == "/":
                self.skip_comment()
                continue

            # Terminals

            if self.current_char.isalpha() or self.current_char == "_":
                return self.identifier()

            elif self.current_char.isdigit():
                return self.number()

            # Operators

            elif self.current_char == "=":
                token = Token(TokenType.ASSIGN, self.current_char, self.lineno, self.linecol)
                self.advance()
                return token

            elif self.current_char == '*':
                token = Token(TokenType.MULT, self.current_char, self.lineno, self.linecol)
                self.advance()
                return token

            elif self.current_char == '/':

                if self.peek() != "/":
                    token = Token(TokenType.FLOAT_DIV, self.current_char, self.lineno, self.linecol)
                # Disabled in place of comments
                #else:
                #    token = Token(type.INTEGER_DIV, self.current_char, self.lineno, self.linecol)
                #    self.advance()

                self.advance()

                return token

            elif self.current_char == '+':
                token = Token(TokenType.PLUS, self.current_char, self.lineno, self.linecol)
                self.advance()
                return token

            elif self.current_char == '-':
                if self.peek() == ">":
                    token = Token(TokenType.RETURNS_OP, self.current_char, self.lineno, self.linecol)
                    self.advance()
                    self.advance()
                else:
                    token = Token(TokenType.MINUS, self.current_char, self.lineno, self.linecol)
                    self.advance()

                return token

            # Symbols

            elif self.current_char == ";":
                token = Token(TokenType.SEMI, self.current_char, self.lineno, self.linecol)
                self.advance()
                return token

            elif self.current_char == ":":
                token = Token(TokenType.COLON, self.current_char, self.lineno, self.linecol)
                self.advance()
                return token

            elif self.current_char == ",":
                token = Token(TokenType.COMMA, self.current_char, self.lineno, self.linecol)
                self.advance()
                return token

            elif self.current_char == '(':
                token = Token(TokenType.LPAREN, self.current_char, self.lineno, self.linecol)
                self.advance()
                return token

            elif self.current_char == ')':
                token = Token(TokenType.RPAREN, self.current_char, self.lineno, self.linecol)
                self.advance()
                return token

            elif self.current_char == "{":
                token = Token(TokenType.BEGIN, self.current_char, self.lineno, self.linecol)
                self.advance()
                return token

            elif self.current_char == "}":
                token = Token(TokenType.END, self.current_char, self.lineno, self.linecol)
                self.advance()
                return token

            self.error()

        return Token(TokenType.EOF, None, self.lineno, self.linecol)


###########################################
#                                         #
#   Parser code                           #
#                                         #
###########################################

class Parser:
    """
    Main parser class

    The class is responsible for parsing the tokens and turning them into syntax trees.
    These trees make it easier to process the code and understand the relationships between tokens.

    For example give the set of tokens (equivalent to `1 + 1`):
    ```
    Token[type = type.INTEGER_CONST, id = '2', start_pos = 0]
    Token[type = type.PLUS, id = '+', start_pos = 2]
    Token[type = type.INTEGER_CONST, id = '2', start_pos = 4]
    ```

    The parser will generate:
    ```
    Program(
        statements: [
            BinOp(
                left: Num(
                    token: Token[type = type.INTEGER_CONST, id = '1', start_pos = 0],
                    id: 1
                ),
                op: Token[type = type.PLUS, id = '+', start_pos = 2],
                right: Num(
                    token: Token[type = type.INTEGER_CONST, id = '2', start_pos = 4],
                    id: 2
                )
            )
        ]
    )
    ```
    """

    def __init__(self, text):
        self.text: str = text
        self.lexer: Lexer = Lexer(self.text)
        self.current_token: Token = self.lexer.get_next_token()

    def error(self, error_code: ErrorCode, token: Token, message):
        error = ParserError(error_code, message, token=token, surrounding_lines=self.lexer.text_lines)
        error.trigger()

    def eat(self, expected_type: TokenType):
        """
        Compares the current token type to the expected
        type and, if equal, 'eat' the current
        token and move onto the next token.
        """
        if PRINT_TOKENS:
            print(self.current_token, expected_type)
        if self.current_token.type == expected_type:
            self.current_token = self.lexer.get_next_token()
        else:
            if expected_type == TokenType.END:
                self.error(
                    error_code=ErrorCode.SYNTAX_ERROR,
                    token=self.current_token,
                    message=f"Unexpected type <{self.current_token.type.name}>"
                )
            elif expected_type == TokenType.SEMI:
                self.error(
                    error_code=ErrorCode.SYNTAX_ERROR,
                    token=self.current_token,
                    message=f"Expected type <{expected_type.name}> but got type <{self.current_token.type.name}>, perhaps you forgot a semicolon?"
                )
            else:
                self.error(
                    error_code=ErrorCode.SYNTAX_ERROR,
                    token=self.current_token,
                    message=f"Expected type <{expected_type.name}> but got type <{self.current_token.type.name}>"
                )

    # Could be a function native to `Token`
    def is_type(self) -> bool:
        """
        Check if the current token is a datatype
        """
        if self.current_token.type in [
            TokenType.INTEGER,
            TokenType.FLOAT
        ]:
            return True
        else:
            return False

    # Grammar definitions

    def program(self) -> Program:
        """
        program -> statement_list <`EOF`>
        """
        node = Program()

        node.statements = self.statement_list()

        if PRINT_EAT_STACK:
            print("(Parser) Calling eat() from line", getframeinfo(currentframe()).lineno)
        self.eat(TokenType.EOF)

        return node

    def statement_list(self) -> list[Node]:
        """
        statement_list -> statement `SEMI`
                        | statement `SEMI` statement_list
                        | empty
        """
        node = self.statement()

        results = [node]

        while self.current_token.type == TokenType.SEMI:
            if PRINT_EAT_STACK:
                print("(Parser) Calling eat() from line", getframeinfo(currentframe()).lineno)
            self.eat(TokenType.SEMI)
            statement = self.statement()

            # Specific error handling due to weird no-op behaviour

            if isinstance(statement, NoOp) and self.current_token.type == TokenType.SEMI:
                self.error(
                    error_code=ErrorCode.SYNTAX_ERROR,
                    token=self.current_token,
                    message="Too many semicolons!"
                )
            elif isinstance(statement, Compound) and self.current_token.type != TokenType.SEMI:
                self.error(
                    error_code=ErrorCode.SYNTAX_ERROR,
                    token=self.current_token,
                    message="Missing semicolon after compound"
                )
            elif isinstance(statement, ProcedureDecl) and self.current_token.type != TokenType.SEMI:
                self.error(
                    error_code=ErrorCode.SYNTAX_ERROR,
                    token=self.current_token,
                    message="Missing semicolon after procedure"
                )
            else:
                results.append(statement)

        return results

    def statement(self) -> Node:
        """
        statement -> compound_statement
                   | procedure_declaration
                   | variable_declaration
                   | variable_assignment
        """
        if self.current_token.type == TokenType.BEGIN:
            node = self.compound_statement()
        elif self.current_token.type == TokenType.DEFINITION:
            node = self.procedure_declaration()
        elif self.is_type():
            node = self.variable_declaration()
        elif self.current_token.type == TokenType.IDENTIFIER:
            node = self.variable_assignment()
        else:
            node = self.empty()
        return node

    def compound_statement(self) -> Compound:
        """
        compound_statement -> `BEGIN` statement_list `END`
        """
        if PRINT_EAT_STACK:
            print("(Parser) Calling eat() from line", getframeinfo(currentframe()).lineno)
        self.eat(TokenType.BEGIN)
        nodes = self.statement_list()
        if PRINT_EAT_STACK:
            print("(Parser) Calling eat() from line", getframeinfo(currentframe()).lineno)
        self.eat(TokenType.END)

        root = Compound()
        for node in nodes:
            root.children.append(node)

        return root

    def procedure_declaration(self) -> ProcedureDecl:
        """
        procedure_declaration -> `DEFINITION` variable `LPAREN` formal_parameter_list `RPAREN` compound_statement
                               | `DEFINITION` variable `LPAREN` formal_parameter_list `RPAREN` `RETURNS_OP` type_spec compound_statement
        """
        if PRINT_EAT_STACK:
            print("(Parser) Calling eat() from line", getframeinfo(currentframe()).lineno)
        self.eat(TokenType.DEFINITION)

        procedure_var = self.variable()

        if PRINT_EAT_STACK:
            print("(Parser) Calling eat() from line", getframeinfo(currentframe()).lineno)
        self.eat(TokenType.LPAREN)

        params = self.formal_parameter_list()

        if PRINT_EAT_STACK:
            print("(Parser) Calling eat() from line", getframeinfo(currentframe()).lineno)
        self.eat(TokenType.RPAREN)

        if self.current_token.type == TokenType.BEGIN:

            body = self.compound_statement()

            proc_decl = ProcedureDecl(procedure_var, params, body)

        elif self.current_token.type == TokenType.RETURNS_OP:

            if PRINT_EAT_STACK:
                print("(Parser) Calling eat() from line", getframeinfo(currentframe()).lineno)
            self.eat(TokenType.RETURNS_OP)

            return_type = self.type_spec()

            body = self.compound_statement()

            proc_decl = ProcedureDecl(procedure_var, params, body, return_type=return_type)

        else:
            self.error(
                error_code=ErrorCode.SYNTAX_ERROR,
                token=self.current_token,
                message=f"Invalid procedure declaration form"
            )

        return proc_decl

    def variable_declaration(self) -> VarDecl | Compound:
        """
        variable_declaration -> type_spec variable `ASSIGN` expr
                              | type_spec variable (`COMMA` variable)*

        """
        type_node = self.type_spec()

        var_node = self.variable()

        # type_spec variable `ASSIGN` expr
        if self.current_token.type == TokenType.ASSIGN:
            assign_op = self.current_token
            if PRINT_EAT_STACK:
                print("(Parser) Calling eat() from line", getframeinfo(currentframe()).lineno)
            self.eat(TokenType.ASSIGN)
            expr_node = self.expr()

            node = VarDecl(type_node, var_node, assign_op, expr_node)

        # type_spec variable (`COMMA` variable)*
        else:
            node = Compound()
            node.children.append(VarDecl(type_node, var_node))
            while self.current_token.type == TokenType.COMMA:
                if PRINT_EAT_STACK:
                    print("(Parser) Calling eat() from line", getframeinfo(currentframe()).lineno)
                self.eat(TokenType.COMMA)
                var_node = self.variable()
                node.children.append(VarDecl(type_node, var_node))

        return node

    def variable_assignment(self) -> AssignOp:
        """
        variable_assignment -> variable `ASSIGN` expr
        """
        var_node = self.current_token
        if PRINT_EAT_STACK:
            print("(Parser) Calling eat() from line", getframeinfo(currentframe()).lineno)
        self.eat(TokenType.IDENTIFIER)

        assign_op = self.current_token
        if PRINT_EAT_STACK:
            print("(Parser) Calling eat() from line", getframeinfo(currentframe()).lineno)
        self.eat(TokenType.ASSIGN)

        right = self.expr()
        node = AssignOp(var_node, assign_op, right)

        return node

    def formal_parameter_list(self) -> list[Param]:
        """
        formal_parameter_list -> formal_parameter
                               | formal_parameter `COMMA` formal_parameter_list
                               | empty
        """
        if self.current_token.type == TokenType.RPAREN:
            results = []

        else:
            node = self.formal_parameter()

            results = [node]

            while self.current_token.type == TokenType.COMMA:
                if PRINT_EAT_STACK:
                    print("(Parser) Calling eat() from line", getframeinfo(currentframe()).lineno)
                self.eat(TokenType.COMMA)
                results.append(self.formal_parameter())

            # Commented out due to unknown behaviour
            #if self.current_token.type == TokenType.IDENTIFIER:
            #    self.error()

        return results

    def formal_parameter(self) -> Param:
        """
        formal_parameter -> type_spec variable
        """
        type_node = self.type_spec()
        var_node = self.variable()

        param_node = Param(var_node, type_node)

        return param_node

    def type_spec(self) -> TypeNode:
        """
        type_spec -> `INTEGER` | `FLOAT`
        """
        token = self.current_token
        if self.is_type():
            if PRINT_EAT_STACK:
                print("(Parser) Calling eat() from line", getframeinfo(currentframe()).lineno)
            self.eat(token.type)
        else:
            self.error(
                error_code=ErrorCode.TYPE_ERROR,
                token=self.current_token,
                message=f"'{self.current_token.id}' is not a valid type!"
            )

        node = TypeNode(token)
        return node

    def empty(self) -> NoOp:
        """
        empty ->
        """
        return NoOp()

    def expr(self) -> Node:
        """
        expr -> term ((`PLUS`|`MINUS`) term)*
        """
        node = self.term()

        # term ((`PLUS`|`MINUS`) term)*
        while self.current_token.type in (TokenType.PLUS, TokenType.MINUS):
            token = self.current_token

            if token.type == TokenType.PLUS:
                if PRINT_EAT_STACK:
                    print("(Parser) Calling eat() from line", getframeinfo(currentframe()).lineno)
                self.eat(TokenType.PLUS)

            elif token.type == TokenType.MINUS:
                if PRINT_EAT_STACK:
                    print("(Parser) Calling eat() from line", getframeinfo(currentframe()).lineno)
                self.eat(TokenType.MINUS)

            node = BinOp(left=node, op=token, right=self.term())

        return node

    def term(self) -> Node:
        """
        term -> factor ((`MUL`|`INTEGER_DIV`|`FLOAT_DIV`) factor)*
        """
        node = self.factor()

        # factor ( (`MUL`|`DIV`) factor)*
        while self.current_token.type in (TokenType.MULT, TokenType.INTEGER_DIV, TokenType.FLOAT_DIV):
            token = self.current_token

            if token.type == TokenType.MULT:
                if PRINT_EAT_STACK:
                    print("(Parser) Calling eat() from line", getframeinfo(currentframe()).lineno)
                self.eat(TokenType.MULT)

            elif token.type == TokenType.INTEGER_DIV:
                if PRINT_EAT_STACK:
                    print("(Parser) Calling eat() from line", getframeinfo(currentframe()).lineno)
                self.eat(TokenType.INTEGER_DIV)

            elif token.type == TokenType.FLOAT_DIV:
                if PRINT_EAT_STACK:
                    print("(Parser) Calling eat() from line", getframeinfo(currentframe()).lineno)
                self.eat(TokenType.FLOAT_DIV)

            node = BinOp(left=node, op=token, right=self.factor())

        return node

    def factor(self) -> Node:
        """
        factor -> `PLUS` factor
                | `MINUS` factor
                | `INTEGER_CONST`
                | `FLOAT_CONST` 
                | `LPAREN` expr `RPAREN`
                | variable
        """
        token = self.current_token

        # `PLUS` factor
        if token.type == TokenType.PLUS:
            if PRINT_EAT_STACK:
                print("(Parser) Calling eat() from line", getframeinfo(currentframe()).lineno)
            self.eat(TokenType.PLUS)
            node = UnaryOp(token, self.factor())
            return node

        # `MINUS` factor
        elif token.type == TokenType.MINUS:
            if PRINT_EAT_STACK:
                print("(Parser) Calling eat() from line", getframeinfo(currentframe()).lineno)
            self.eat(TokenType.MINUS)
            node = UnaryOp(token, self.factor())
            return node

        # `INTEGER_CONST`
        elif token.type == TokenType.INTEGER_CONST:
            if PRINT_EAT_STACK:
                print("(Parser) Calling eat() from line", getframeinfo(currentframe()).lineno)
            self.eat(TokenType.INTEGER_CONST)
            return Num(token)

        # `FLOAT_CONST`
        elif token.type == TokenType.FLOAT_CONST:
            if PRINT_EAT_STACK:
                print("(Parser) Calling eat() from line", getframeinfo(currentframe()).lineno)
            self.eat(TokenType.FLOAT_CONST)
            return Num(token)

        # `LPAREN` expr `RPAREN`
        elif token.type == TokenType.LPAREN:
            if PRINT_EAT_STACK:
                print("(Parser) Calling eat() from line", getframeinfo(currentframe()).lineno)
            self.eat(TokenType.LPAREN)
            node = self.expr()
            if PRINT_EAT_STACK:
                print("(Parser) Calling eat() from line", getframeinfo(currentframe()).lineno)
            self.eat(TokenType.RPAREN)
            return node

        # variable
        else:
            node = self.variable()
            return node

    def variable(self) -> Var:
        """
        variable -> `IDENTIFIER`
        """
        node = Var(self.current_token)
        if PRINT_EAT_STACK:
            print("(Parser) Calling eat() from line", getframeinfo(currentframe()).lineno)
        self.eat(TokenType.IDENTIFIER)
        return node

    def parse(self) -> Node:
        """Main Parser method

        Here is the program grammar:

        ```
        program -> statement_list <`EOF`>

        statement_list -> statement `SEMI`
                        | statement `SEMI` statement_list
                        | empty

        statement -> compound_statement
                   | procedure_declaration
                   | variable_declaration
                   | variable_assignment

        compound_statement -> `BEGIN` statement_list `END`

        procedure_declaration -> `DEFINITION` variable `LPAREN` formal_parameter_list `RPAREN` compound_statement
                               | `DEFINITION` variable `LPAREN` formal_parameter_list `RPAREN` `RETURNS_OP` type_spec compound_statement

        variable_declaration -> type_spec variable `ASSIGN` expr
                              | type_spec variable (`COMMA` variable)*

        variable_assignment -> variable `ASSIGN` expr

        formal_parameter_list -> formal_parameter
                               | formal_parameter `COMMA` formal_parameter_list
                               | empty

        formal_parameter -> type_spec variable

        type_spec -> `INTEGER` | `FLOAT`

        empty ->
        // What did you expect cuh

        expr -> term ((`PLUS`|`MINUS`) term)*

        term -> factor ((`MUL`|`INTEGER_DIV`|`FLOAT_DIV`) factor)*

        factor -> `PLUS` factor
                | `MINUS` factor
                | `INTEGER_CONST`
                | `FLOAT_CONST` 
                | `LPAREN` expr `RPAREN`
                | variable

        variable -> `IDENTIFIER`
        ```
        """
        node = self.program()
        if self.current_token.type != TokenType.EOF:
            self.error(
                error_code=ErrorCode.SYNTAX_ERROR,
                token=self.current_token,
                message=f"Program terminated with <{self.current_token.type.value}>, not <{TokenType.EOF}>"
            )

        return node


###########################################
#                                         #
#   Semantic analysis (type checking)     #
#                                         #
###########################################

class SemanticAnalyser(NodeVisitor):
    """
    Constructs the symbol table and performs type-checks before runtime
    """
    def __init__(self, text):
        self.text_lines: list[str] = text.split('\n')
        self.current_scope: SymbolTable | None = None

    def error(self, error_code: ErrorCode, token: Token, message):
        error = SemanticAnalyserError(error_code, message, token, surrounding_lines=self.text_lines)
        error.trigger()

    def visit_Program(self, node: Program):
        builtin_scope = SymbolTable(scope_name="<builtins>", scope_level=0)
        global_scope = SymbolTable(scope_name="<global>", scope_level=1, parent_scope=builtin_scope)
        self.current_scope = global_scope

        if PRINT_SCOPE:
            print(builtin_scope)

        for child in node.statements:
            self.visit(child)

        if PRINT_SCOPE:
            print(global_scope)

        # Return to global scope
        self.current_scope = global_scope

    def visit_Compound(self, node: Compound):
        # TODO: Implement scoping around compound statements
        for child in node.children:
            self.visit(child)

    def visit_VarDecl(self, node: VarDecl):
        type_symbol = self.visit(node.type_node)

        var_id = node.var_node.id

        if self.current_scope.lookup(var_id, search_parent_scopes=False) is not None:
            self.error(
                error_code=ErrorCode.NAME_ERROR,
                token=node.var_node.token,
                message="Cannot initialise variable with same name"
            )

        var_symbol = VarSymbol(var_id, type_symbol)
        self.current_scope.define(var_symbol)

        if node.expr_node is not None:
            self.visit(node.expr_node)

    def visit_ProcedureDecl(self, node: ProcedureDecl):
        proc_name = node.procedure_var.id
        proc_params: list[Param] = node.params

        if self.current_scope.lookup(proc_name) is not None:
            self.error(
                error_code=ErrorCode.NAME_ERROR,
                token=node.procedure_var.token,
                message="Cannot declare procedure with same name"
            )

        proc_symbol = ProcedureSymbol(proc_name, proc_params)
        self.current_scope.define(proc_symbol)

        proc_scope = SymbolTable(scope_name=proc_name, scope_level=self.current_scope.scope_level + 1,
                                 parent_scope=self.current_scope)
        self.current_scope = proc_scope

        for param in proc_params:
            param_type = self.current_scope.lookup(param.type_node.id)
            param_name = param.var_node.id
            var_symbol = VarSymbol(param_name, param_type)
            self.current_scope.define(var_symbol)

        self.visit(node.compound_node)

        if PRINT_SCOPE:
            print(self.current_scope)

        # Return to parent scope
        self.current_scope = self.current_scope.parent_scope

    def visit_AssignOp(self, node: AssignOp):
        var_id = node.left.id
        var_symbol = self.current_scope.lookup(var_id)

        if var_symbol is None:
            self.error(
                error_code=ErrorCode.NAME_ERROR,
                token=node.left,
                message=f"Variable {repr(var_id)} does not exist"
            )

        self.visit(node.right)

    def visit_UnaryOp(self, node: UnaryOp):
        self.visit(node.expr)

    def visit_BinOp(self, node: BinOp):
        self.visit(node.left)
        self.visit(node.right)

    def visit_TypeNode(self, node: TypeNode):
        type_id = node.id
        type_symbol = self.current_scope.lookup(type_id)

        if type_symbol is None:
            self.error(
                error_code=ErrorCode.NAME_ERROR,
                token=node.token,
                message=f"Unrecognised type {repr(type_id)}"
            )
        else:
            return type_symbol

    def visit_Var(self, node: Var):
        var_id = node.id
        var_symbol = self.current_scope.lookup(var_id)

        if var_symbol is None:
            self.error(
                error_code=ErrorCode.NAME_ERROR,
                token=node.token,
                message=f"Variable {repr(var_id)} does not exist"
            )
        else:
            return var_symbol

    def visit_Num(self, node):
        pass

    def visit_NoOp(self, node):
        pass


###########################################
#                                         #
#   Interpreter code                      #
#                                         #
###########################################

# Currently some unloved garbárge
class Interpreter(NodeVisitor):
    """
    Main interpreter class

    The interpreter is responsible for processing abstract syntax trees
    and compiling (not machine code) them into a final result.
    It works by 'visiting' each node in the tree and processing it based on its attributes and surrounding nodes.

    It also handles type-checking at runtime
    """

    global_scope = GlobalScope()

    def interpret(self, tree: Node):
        """
        Initiates the recursive descent algorithm,
        generates a syntax tree,
        and executes the code.
        """
        return self.visit(tree)

    def visit_Program(self, node: Program):
        for child in node.statements:
            self.visit(child)

    def visit_Compound(self, node: Compound):
        for child in node.children:
            self.visit(child)

    def visit_VarDecl(self, node: VarDecl):
        variable_id = node.var_node.id
        variable_type_name = node.type_node.id

        if node.expr_node is not None:
            self.global_scope[variable_id] = [variable_type_name, self.visit(node.expr_node)]
        else:
            self.global_scope[variable_id] = [variable_type_name, None]

    def visit_ProcedureDecl(self, node):
        pass

    def visit_AssignOp(self, node: AssignOp):
        variable_id = node.left.id
        if variable_id in self.global_scope:
            self.global_scope[variable_id][1] = self.visit(node.right)
        else:
            raise ValueError("Interpreter :: Attempted to assign value to uninitialised variable!")

    def visit_UnaryOp(self, node: UnaryOp):
        if node.op.type == TokenType.PLUS:
            return +self.visit(node.expr)
        elif node.op.type == TokenType.MINUS:
            return -self.visit(node.expr)

    def visit_BinOp(self, node: BinOp):
        if node.op.type == TokenType.PLUS:
            return self.visit(node.left) + self.visit(node.right)
        elif node.op.type == TokenType.MINUS:
            return self.visit(node.left) - self.visit(node.right)
        elif node.op.type == TokenType.MULT:
            return self.visit(node.left) * self.visit(node.right)
        elif node.op.type == TokenType.INTEGER_DIV:
            return int(self.visit(node.left) // self.visit(node.right))
        elif node.op.type == TokenType.FLOAT_DIV:
            return self.visit(node.left) / self.visit(node.right)

    def visit_TypeNode(self, node: TypeNode):
        # Not utilised yet
        pass

    def visit_Var(self, node: Var):
        variable_id = node.id
        val = self.global_scope.get(variable_id)
        if val is None:
            raise NameError("Interpreter :: " + repr(variable_id))
        else:
            return val[1]

    def visit_Num(self, node: Num):
        return node.id

    def visit_NoOp(self, node: NoOp):
        pass


###########################################
#                                         #
#   Driver code                           #
#                                         #
###########################################

class Driver:
    """
    Driver code to execute the program
    """
    def __init__(self):
        self.filename: str = DEFAULT_FILENAME
        self.mode: str = "cmdline"

    def run_program(self):
        """
        Calls the relevant function for the given mode
        """
        self._process_arguments()
        if self.mode == "cmdline":
            self.cmdline_input()
        elif self.mode == "file":
            # Set the global filename, used by error handler
            global current_filename
            current_filename = self.filename
            self.file_input(self.filename)
        else:
            raise ValueError(f"mode {repr(self.mode)} is not a valid mode.")

    def _process_arguments(self):
        """
        Will configure execution based on command line arguments.

        Very basic implementation right now, will improve later
        """
        if len(sys.argv) == 1:
            # self.mode = "cmdline"
            # NOTE: cmdline disabled while testing to make execution quicker (Since I click run about 100 times/day (JOKE (satire)))
            # All of the following code should be removed in prod (not that I will ever reach that stage)
            if os.path.isfile(DEFAULT_FILENAME):
                self.filename = DEFAULT_FILENAME
                self.mode = "file"
            else:
                raise Exception(f"file {repr(DEFAULT_FILENAME)} does not exist!")

        elif len(sys.argv) == 2:
            path = sys.argv[1]
            print(path)
            if os.path.isfile(path):
                print("here")
                self.filename = path
                self.mode = "file"
        else:
            raise Exception("Unrecognised arguments!")

    def _process(self, code: str):
        parser = Parser(code)
        symbol_table = SemanticAnalyser(code)
        interpreter = Interpreter()

        tree = parser.parse()

        if PRINT_TREE:
            print(tree)

        symbol_table.visit(tree)
        interpreter.interpret(tree)

        print()
        print("Global vars (doesn't account for functions):")
        print(interpreter.global_scope)
        print()

    def cmdline_input(self):
        """
        Run interpreter in command line interface mode
        """
        while 1:

            try:
                text = input(">>> ")
            except KeyboardInterrupt:
                # Silently exit
                return

            if not text:
                continue

            self._process(text)

    def file_input(self, filename: str):
        """
        Run interpreter in file mode
        """
        file = open(filename, "r")
        text = file.read()

        if not text:
            return

        self._process(text)


if __name__ == '__main__':
    driver = Driver()
    driver.run_program()
