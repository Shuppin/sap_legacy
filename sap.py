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

DEFAULT_FILENAME = "procedure_calls.sap"

PRINT_TOKENS = False
PRINT_EAT_STACK = False
PRINT_TREE = False
PRINT_SCOPE = False
PRINT_CALL_STACK = True

# This defines how errors are treated
# If true, it will raise an error with the full stack trace,
# useful for debugging purposes
# If false, it will print the error normally
_DEV_RAISE_ERROR_STACK = False
# Strict semicolons are treated as line terminators and
# are required after every statement
# Non-strict semicolons are treated as statement sperators
# and are only required between statements
_DEV_STRICT_SEMICOLONS = True

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
    # These values do not represent how the lexer identifies tokens,
    # they are just represent what these tokens look like
    # symbols
    MULT                = '*'
    INTEGER_DIV         = '//'  # Currently not in use, may be removed in future
    FLOAT_DIV           = '/'
    PLUS                = '+'
    MINUS               = '-'
    RETURNS_OP          = '->'
    LPAREN              = '('
    RPAREN              = ')'
    ASSIGN              = ':='
    SEMI                = ';'
    COLON               = ':'
    COMMA               = ','
    BEGIN               = '{'
    END                 = '}'
    # reserved keywords
    INTEGER             = 'int'
    FLOAT               = 'float'
    DEFINITION          = 'def'
    # dynamic token types
    INTEGER_LITERAL     = 'INTEGER_LITERAL'
    FLOAT_LITERAL       = 'FLOAT_LITERAL'
    IDENTIFIER          = 'IDENTIFIER'
    EOF                 = 'EOF'


class ActivationRecordType(Enum):
    PROGRAM = "PROGRAM"


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
    """
    Temporary class to act as a symbol table for the interpreter,
    actual symbol tables have not yet been implemented into the interpreter.
    """
    def __init_subclass__(cls):
        return super().__init_subclass__()

    def __str__(self) -> str:
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
        return f"Token[type = {self.type}, id = {repr(self.id)}, position = <{self.lineno}:{self.linecol}>]"

    def __repr__(self) -> str:
        return repr(self.__str__())


###########################################
#                                         #
#  Error handler                          #
#                                         #
###########################################

class BaseError(Exception):
    """
    Error base class
    Inherits from Excpetion so it can be raised using python syntax
    """
    def __init__(self, error_code: ErrorCode, message: str, token:Token=None, position:list[int]=None, surrounding_lines:list[str]=None):
        self.error_code: ErrorCode = error_code
        self.message: str = f'({self.__class__.__name__[:-5]}) {self.error_code.value}: {message}'
        self.token: Token | None = token
        self.surrounding_lines: list[str] | None = surrounding_lines
        # We need the position at which the error occured,
        # It is either extracted from a given token or
        # passed directly as an array
        self.lineno: int
        self.linecol: int
        if token is not None and position is None:
            self.lineno = token.lineno
            self.linecol = token.linecol
        elif position is not None and token is None:
            self.lineno = position[0]
            self.linecol = position[1]
        elif token is not None and position is not None:
            raise ValueError("Too much information passed into Error, either token or position must be given, not both")
        else:
            raise ValueError("Not enough information passed into Error, either token or position must be given")

    def trigger(self):
        """
        Prints out error along with various bit of information.
        Output looks like this:
        ```
        File '.\integers.sap', position <2:16>
        │ 1 │ int x := 5;
        │ 2 │ int y := x + 3 +
                              ^
        (Parser) SyntaxError: Expected type <IDENTIFIER> but got type <EOF>
        ```
        """
        # Creates the message with the '~~~' highlighter
        # For example:
        # | 1 | inte x := 4;
        #       ~~~~
        if self.surrounding_lines is not None and self.token is not None and self.token.startcol is not None:
            error_message = [
                (f'File "{current_filename}", position <{self.lineno}:{self.linecol}>\n'),
                # The if clauses here will ensure it prints the surrounding lines only if it exists
                (f" │ {' '*(len(str(self.lineno+2)) - len(str(self.lineno-3)))}{self.lineno-3} │ {self.surrounding_lines[self.lineno-4]}\n" if (self.lineno-4) >= 0 else f""),
                (f" │ {' '*(len(str(self.lineno+2)) - len(str(self.lineno-2)))}{self.lineno-2} │ {self.surrounding_lines[self.lineno-3]}\n" if (self.lineno-3) >= 0 else f""),
                (f" │ {' '*(len(str(self.lineno+2)) - len(str(self.lineno-1)))}{self.lineno-1} │ {self.surrounding_lines[self.lineno-2]}\n" if (self.lineno-2) >= 0 else f""),
                (f" │ {' '*(len(str(self.lineno+2)) - len(str(self.lineno  )))}{self.lineno  } │ {self.surrounding_lines[self.lineno-1]}\n"),
                (f"   {' '*len(str(self.lineno+2))}  {' '*(self.token.startcol)} {'~'*(self.linecol-self.token.startcol)}\n"),
                (self.message)
                ]
            error_message = "".join(error_message)

        # Creates the message with the '^' highlighter
        # For example:
        # | 1 | int x := 4 $ 3
        #                  ^
        elif self.surrounding_lines is not None:
            error_message = [
                (f'File "{current_filename}", position <{self.lineno}:{self.linecol}>\n'),
                # The if clauses here will ensure it prints the surrounding lines only if it exists
                (f" │ {' '*(len(str(self.lineno+2)) - len(str(self.lineno-3)))}{self.lineno-3} │ {self.surrounding_lines[self.lineno-4]}\n" if (self.lineno-4) >= 0 else f""),
                (f" │ {' '*(len(str(self.lineno+2)) - len(str(self.lineno-2)))}{self.lineno-2} │ {self.surrounding_lines[self.lineno-3]}\n" if (self.lineno-3) >= 0 else f""),
                (f" │ {' '*(len(str(self.lineno+2)) - len(str(self.lineno-1)))}{self.lineno-1} │ {self.surrounding_lines[self.lineno-2]}\n" if (self.lineno-2) >= 0 else f""),
                (f" │ {' '*(len(str(self.lineno+2)) - len(str(self.lineno  )))}{self.lineno  } │ {self.surrounding_lines[self.lineno-1]}\n"),
                (f"   {' '*len(str(self.lineno+2))}  {' '*(self.linecol)} ^\n"),
                (self.message)
                ]
            error_message = "".join(error_message)
        # If no surrounding lines were passed for whatever reason,
        # just print without surrounding lines
        else:
            error_message = (f'File "{current_filename}", position <{self.lineno}:{self.linecol}>\n'
                             f'   <error fetching lines>\n'
                             f'{self.message}')

        # Raise error or just print it normally
        if _DEV_RAISE_ERROR_STACK:
            print(error_message)
            raise self
        else:
            print(error_message)
            exit()


class LexerError(BaseError):
    """
    For lexer specific errors
    """
    pass


class ParserError(BaseError):
    """
    For parser specific errors
    """
    pass


class SemanticAnalyserError(BaseError):
    """
    For semantic specific errors
    """
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
                        raise TypeError(f"Cannot print type {repr(type(node))}")
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
    """
    Program() represents a whole program
    """
    def __init__(self):
        self.statements: list[Node] = []


class Compound(Node):
    """
    Compound() represents a list of statements surrounded by curly brackets
    """
    def __init__(self):
        self.children: list[Node] = []


class VarDecl(Node):
    """
    VarDecl() represents a variable declaration statement
    """
    def __init__(self, type_node, var_node, assign_op=None, expr_node=None):
        self.type_node: TypeNode = type_node
        self.var_node: Var = var_node
        self.assign_op: Token | None = assign_op
        self.expr_node: Node | None = expr_node


class ProcedureDecl(Node):
    """
    ProcedureDecl() represents a procedure declaration statement
    """
    def __init__(self, procedure_var, params, compound_node, return_type=None):
        self.procedure_var: Var = procedure_var
        self.params: list[Param] = params
        self.return_type: TypeNode | None = return_type
        self.compound_node: Compound = compound_node


class ProcedureCall(Node):
    """
    ProcedureCall() represents a procedure call statement
    """
    def __init__(self, procedure_var, literal_params):
        self.procedure_var: Var = procedure_var
        self.literal_params: list[Param] = literal_params


class AssignOp(Node):
    """
    AssignOp() represents an assignment operation
    """
    def __init__(self, left, op, right):
        self.left: Token = left
        self.op: Token = op
        self.right: Node = right


class UnaryOp(Node):
    """
    UnaryOp() represents a unary operation (one-sided operation) such as `-1`
    """
    def __init__(self, op, expr):
        self.op: Token = op
        self.expr: Node = expr


class BinOp(Node):
    """
    BinOp() represents a binary operation (two-sided operation) such as `1+2`
    """
    def __init__(self, left, op, right):
        self.left: Node = left
        self.op: Token = op
        self.right: Node = right


class Param(Node):
    """
    Param() represents a defined argument within a procedure declaration
    """
    def __init__(self, var_node, type_node):
        self.var_node: Var = var_node
        self.type_node: TypeNode = type_node


"""
LEAF NODES
(in order of precedence)
"""


class TypeNode(Node):
    """
    TypeNode() represents a data type
    """
    def __init__(self, token):
        self.token: Token = token
        self.id = self.token.type.name


class Var(Node):
    """
    Var() represents a variable
    """
    def __init__(self, token):
        self.token: Token = token
        self.id = self.token.id


class Num(Node):
    """
    Num() represents any number-like literal such as `23` or `3.14`
    """
    def __init__(self, token):
        self.token: Token = token
        self.id: int | str | None = self.token.id


class NoOp(Node):
    """
    NoOp() represents an empty statement,
    for example there would be a NoOp between `;;` because semicolons act as seperators
    """
    pass


###########################################
#                                         #
#   Symbols                               #
#                                         #
###########################################


class BaseSymbol:
    """
    Symbol base class
    """
    def __init__(self, name, datatype=None):
        self.name: str = name
        self.type: BuiltinSymbol | None = datatype

    def __str__(self) -> str:
        return self.name


class BuiltinSymbol(BaseSymbol):
    """
    Symbol which represents built in types
    """
    def __init__(self, name):
        super().__init__(name)

    def __str__(self) -> str:
        return f"<builtin> {self.name}"


class VarSymbol(BaseSymbol):
    """
    Symbol which represents user-defined variables
    """
    def __init__(self, name, datatype):
        super().__init__(name, datatype)

    def __str__(self) -> str:
        return f"<variable> (id: {repr(self.name)}, type: {repr(self.type.name)})"


class ProcedureSymbol(BaseSymbol):
    """
    Symbol which represents procedure declarations
    """
    def __init__(self, name, params=[]):
        super().__init__(name)
        self.params: list[Param] = params

    def __str__(self) -> str:
        if len(self.params) == 0:
            return f"<procedure> (id: {repr(self.name)}, parameters: <no params>)"
        else:
            # Okay, yes this is (slightly less) horrendous don't @me
            parameter_list = ', '.join(
                list(
                    map(
                        lambda param: f"({repr(param.var_node.id)}, <{param.type_node.id}>)",
                        self.params
                    )
                )
            )
            return (f"<procedure> (id: {repr(self.name)}, parameters: {parameter_list}")


###########################################
#                                         #
#   Symbol table code                     #
#                                         #
###########################################


class SymbolTable:
    """
    Class to store all the program symbols
    """
    def __init__(self, scope_name, scope_level, parent_scope=None):
        self._symbols: dict[str, BaseSymbol] = {}
        self.scope_name: str = scope_name
        self.scope_level: int = scope_level
        self.parent_scope: SymbolTable | None = parent_scope

        # The only table with a scope level of 0 should be the
        # <builtins> table, which represents all the built-in
        # types. These are defined internally here:
        if self.scope_level == 0:
            self.define(BuiltinSymbol("INTEGER"))
            self.define(BuiltinSymbol("FLOAT"))

    def __str__(self) -> str:
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

    def define(self, symbol: BaseSymbol):
        """
        Adds a symbol to the symbol table
        """
        self._symbols[symbol.name] = symbol

    def lookup(self, symbol_name: str, search_parent_scopes: bool = True) -> BaseSymbol | None:
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
#   Memory system                         #
#                                         #
###########################################


class Member:
    def __init__(self, name: str, value: Any, datatype: str):
        self.name: str = name
        self.value: Any = value
        # May update to Symbol in future
        self.datatype: str = datatype

    def __str__(self):
        return f"<{self.datatype}> {self.name} = {repr(self.value)}"


class ActivationRecord:
    def __init__(self, name: str, ar_type: ActivationRecordType, scope_level: int):
        self.name: str = name
        self.ar_type: ActivationRecordType = ar_type
        self.scope_level: int = scope_level
        self._members: dict[str, Member] = {}

    def __str__(self):
        message = "\nACTIVATION RECORD:\n"
        message += f"Scope name    : {self.name}\n"
        message += f"Scope level   : {self.scope_level}\n"
        message += f"AR type       : {self.ar_type.value}\n\n"
        message += "Activation record contents\n"
        message += "--------------------------\n"
        if self._members == {}:
            message += "\n    <empty>\n"
        else:
            member_objects: list[Member]
            _, member_objects = zip(*self._members.items())
            grouped_members = defaultdict(list)
            for member in member_objects:
                grouped_members[member.datatype].append(member)

            for _, members in grouped_members.items():
                message+="\n"
                for member in members:
                    message += "    " + str(member) + "\n"

        message = message.split("\n")
        longest_string_length = len(max(message, key=len))
        message.insert(2, "=" * (longest_string_length + 1))
        message.append("=" * (longest_string_length + 1) + "\n")
        message = "\n".join(message)

        return message

    def set(self, member: Member):
        self._members[member.name] = member

    def get(self, key: str) -> Member:
        return self._members.get(key)


class CallStack:
    def __init__(self):
        self._records: list[ActivationRecord] = []

    def __str__(self) -> str:
        message = f"\nCALL STACK (File \"{current_filename}\")\n"
        for record in self._records:
            message += str(record)
        return message

    def push(self, ar: ActivationRecord):
        self._records.append(ar)

    def pop(self) -> ActivationRecord:
        return self._records.pop()

    def peek(self) -> ActivationRecord:
        return self._records[-1]


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
        # This function uses some python trickery
        # to call a specific visit_(...) function
        # based on the class name from the given
        # 'node' argument.

        # For example if a given node has the name
        # 'Var' this function would call 'visit_Var()'

        # If there is no visit_(...) function that
        # matches the name of a given node then
        # visitor_not_found() is called instead.
        method_name = "visit_" + type(node).__name__
        visitor = getattr(self, method_name, self.visitor_not_found)
        return visitor(node)

    def visitor_not_found(self, node: Node):
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
    Token[type = type.INTEGER_LITERAL, id = '2', start_pos = 0]
    Token[type = type.PLUS, id = '+', start_pos = 2]
    Token[type = type.INTEGER_LITERAL, id = '2', start_pos = 4]
    ```
    """
    def __init__(self, text):
        self.text: str = text
        self.pos: int = 0
        self.lineno: int = 1
        self.linecol: int = 0
        self.current_char: str | None = self.text[self.pos]

        # Just self.text split up into a list of each line
        # Used by the error reporter
        self.text_lines: list[str] = self.text.split('\n')

        # Used to keep track how many times
        # the lexer tries to get the next token
        # even though the pointer is at the end
        # of the code
        self.reached_end_counter = 0

        self.RESERVED_KEYWORDS: dict = {
            'int': TokenType.INTEGER,
            'float': TokenType.FLOAT,
            'def': TokenType.DEFINITION
        }

    # Utility functions

    def error(self, message=None, char_pos=None):
        """
        Create and raise a LexerError object
        """
        # Set default error message
        if message is None:
            message = f"Could not tokenise {repr(self.current_char)}"
        # Set default position
        if char_pos is None:
            char_pos=[self.lineno, self.linecol]
        error = LexerError(
            ErrorCode.SYNTAX_ERROR,
            message,
            position=char_pos,
            surrounding_lines=self.text_lines
        )
        error.trigger()

    def advance(self):
        """Advance `self.pos` and set `self.current_char`"""
        self.pos += 1
        self.linecol += 1

        # If newline
        if self.current_char == "\n":
            self.lineno += 1
            self.linecol = 0

        # If `self.pos` has reached the end of the code
        if self.pos > len(self.text) - 1:
            # After the lexer has tried multiple times to get
            # the next token while being at the end of the code.
            # This behaviour occurs when there is an error with
            # the lexeical analysis stage.
            if self.reached_end_counter > 3:
                print("(Lexer) [CRITICAL] Lexer has reached end of code but is still trying to advance")
                exit()
            self.reached_end_counter += 1
            self.current_char = None
        # Else advance as normal
        else:
            self.current_char = self.text[self.pos]
            
    def peek(self) -> None | str:
        """Peeks at the next character in the code and returns it"""
        peek_pos = self.pos + 1
        if peek_pos > len(self.text) - 1:
            return None
        else:
            return self.text[peek_pos]

    # Lexer utility functions

    def skip_whitespace(self):
        """Advances `self.pos` until a non-whitespace character has been reached"""
        while self.current_char is not None and self.current_char == " ":
            self.advance()

    def skip_multi_comment(self):
        """Advances `self.pos` until a comment terminator (*/)  or <EOF> has been reached"""
        start_pos = [self.lineno, self.linecol]
        while not (self.current_char == "*" and self.peek() == "/"):
            self.advance()
            # If we have reached the end of the code
            # without reaching a '*/' comment terminator
            if self.current_char is None:
                self.error(
                    message="Unterminated multiline comment",
                    char_pos=start_pos
                )
        # Advance twice more to skip over the final "*/"
        self.advance()
        self.advance()

    def skip_comment(self):
        """Advances `self.pos` until a newline has been reached"""
        while self.current_char is not None and not self.current_char == "\n":
            self.advance()
        # Advance once more over the final '\n'
        self.advance()

    def identifier(self) -> Token:
        """Creates and returns an identifier token"""
        result = ""
        start_pos = self.linecol
        # While the current char is alpha numeric or '_'
        while self.current_char is not None and (self.current_char.isalnum() or self.current_char == "_"):
            result += self.current_char
            self.advance()

        # Gets the token type associated with `result` if applicable, else default to `type.IDENTIFIER`
        token_type = self.RESERVED_KEYWORDS.get(result, TokenType.IDENTIFIER)

        token = Token(token_type, result, self.lineno, self.linecol, startcol=start_pos)

        return token

    def number(self) -> Token:
        """Consumes a number from the input code and returns it"""
        number = ''
        start_pos = self.linecol
        # While the current char is a digit
        while self.current_char is not None and self.current_char.isdigit():
            number += self.current_char
            self.advance()

        # If there is a decimal that would indicate a float.
        if self.current_char == ".":
            number += self.current_char
            self.advance()

            # It is possible for a float to have no leading or
            # trailing digits, which is handled with the
            # `has_decimals` check
            has_decimals = False

            while self.current_char is not None and self.current_char.isdigit():
                number += self.current_char
                has_decimals = True
                self.advance()

            if has_decimals == False:
                self.error(
                    message="Incomplete float",
                    char_pos=[self.lineno, self.linecol-1]
                )

            token = Token(TokenType.FLOAT_LITERAL, float(number), self.lineno, self.linecol, startcol=start_pos)

        else:
            token = Token(TokenType.INTEGER_LITERAL, int(number), self.lineno, self.linecol, startcol=start_pos)

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

            # Size-variant symbols

            if self.current_char.isalpha() or self.current_char == "_":
                return self.identifier()

            elif self.current_char.isdigit():
                return self.number()

            # Operators

            elif self.current_char == ":" and self.peek() == "=":
                token = Token(TokenType.ASSIGN, ':=', self.lineno, self.linecol)
                self.advance()
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
                    token = Token(TokenType.RETURNS_OP, "->", self.lineno, self.linecol)
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
    Token[type = type.INTEGER_LITERAL, id = '2', start_pos = 0]
    Token[type = type.PLUS, id = '+', start_pos = 2]
    Token[type = type.INTEGER_LITERAL, id = '2', start_pos = 4]
    ```

    The parser will generate:
    ```
    Program(
        statements: [
            BinOp(
                left: Num(
                    token: Token[type = type.INTEGER_LITERAL, id = '1', start_pos = 0],
                    id: 1
                ),
                op: Token[type = type.PLUS, id = '+', start_pos = 2],
                right: Num(
                    token: Token[type = type.INTEGER_LITERAL, id = '2', start_pos = 4],
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
        # Previous token refers to the token before the current token
        # It is initially set to an empty token
        # Exclusively used by the error reporter
        self.previous_token: Token = Token(None,None,0,0)

    def error(self, error_code: ErrorCode, token: Token, message):
        """
        Create and raise a ParserError object
        """
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
            self.previous_token = self.current_token
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

    def semicolon_check(self, statement):
        """
        Specific error handling due to weird no-op behaviour

        Specifically, it checks the following:
        - Multiple semicolons next to each other
        - Missing semicolons after compound statements
        - Missing semicolons after procedure declarations
        - Missing semicolons after any other statement

        Note: compounds and procedures are handled seprately for more accurate error reporting
        """
        if isinstance(statement, NoOp) and self.current_token.type == TokenType.SEMI:
            self.error(
                error_code=ErrorCode.SYNTAX_ERROR,
                token=self.current_token,
                message="Too many semicolons!"
            )
        elif isinstance(statement, Compound) and self.current_token.type != TokenType.SEMI:
            self.error(
                error_code=ErrorCode.SYNTAX_ERROR,
                token=self.previous_token,
                message="Missing semicolon after compound"
            )
        elif isinstance(statement, ProcedureDecl) and self.current_token.type != TokenType.SEMI:
            self.error(
                error_code=ErrorCode.SYNTAX_ERROR,
                token=self.previous_token,
                message="Missing semicolon after procedure"
            )
        elif (not isinstance(statement, NoOp)) and self.current_token.type != TokenType.SEMI:
            self.error(
                error_code=ErrorCode.SYNTAX_ERROR,
                token=self.previous_token,
                message=f"Missing semicolon" 
            )

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
        """
        node = self.statement()
        if _DEV_STRICT_SEMICOLONS:
            self.semicolon_check(node)

        results = [node]

        while self.current_token.type == TokenType.SEMI:
            if PRINT_EAT_STACK:
                print("(Parser) Calling eat() from line", getframeinfo(currentframe()).lineno)
            self.eat(TokenType.SEMI)
            statement = self.statement()
            if _DEV_STRICT_SEMICOLONS:
                self.semicolon_check(statement)
            results.append(statement)

        return results

    def statement(self) -> Node:
        """
        statement -> compound_statement
                   | procedure_declaration
                   | procedure_call
                   | variable_declaration
                   | variable_assignment
                   | empty
        """
        if self.current_token.type == TokenType.BEGIN:
            node = self.compound_statement()
        elif self.current_token.type == TokenType.DEFINITION:
            node = self.procedure_declaration()
        elif self.is_type():
            node = self.variable_declaration()
        elif self.current_token.type == TokenType.IDENTIFIER:
            assert len(TokenType.LPAREN.value) == 1  # Assert that `LPAREN` is a single character symbol
            if self.lexer.current_char == TokenType.LPAREN.value:
                node = self.procedure_call()
            else:
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

    def procedure_call(self) -> ProcedureCall:
        """
        procedure_call -> variable `LPAREN` (empty | expr (`COMMA` expr)*) `RPAREN`
        """
        procedure_var = self.current_token

        if PRINT_EAT_STACK:
            print("(Parser) Calling eat() from line", getframeinfo(currentframe()).lineno)
        self.eat(TokenType.IDENTIFIER)

        if PRINT_EAT_STACK:
            print("(Parser) Calling eat() from line", getframeinfo(currentframe()).lineno)
        self.eat(TokenType.LPAREN)

        literal_params = []
        if self.current_token.type != TokenType.RPAREN:
            # Parameters can either be single variables, or full arithmetic expressions,
            # both of which are valid `expr()`'s
            literal_params.append(self.expr())

            # If there is a comma after the first param,
            # that means there should be more parameters after it
            while self.current_token.type == TokenType.COMMA:
                if PRINT_EAT_STACK:
                    print("(Parser) Calling eat() from line", getframeinfo(currentframe()).lineno)
                self.eat(TokenType.COMMA)
                literal_params.append(self.expr())

        if PRINT_EAT_STACK:
            print("(Parser) Calling eat() from line", getframeinfo(currentframe()).lineno)
        self.eat(TokenType.RPAREN)

        node = ProcedureCall(procedure_var, literal_params)

        return node

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
        
        # type_spec variable 
        elif self.current_token.type == TokenType.SEMI:
            node = VarDecl(type_node, var_node)

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
                message=f"{repr(self.current_token.id)} is not a valid type!"
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
        factor -> `MINUS` factor
                | `INTEGER_LITERAL`
                | `FLOAT_CONST` 
                | `LPAREN` expr `RPAREN`
                | variable
        """
        token = self.current_token

        # `MINUS` factor
        if token.type == TokenType.MINUS:
            if PRINT_EAT_STACK:
                print("(Parser) Calling eat() from line", getframeinfo(currentframe()).lineno)
            self.eat(TokenType.MINUS)
            node = UnaryOp(token, self.factor())
            return node

        # `INTEGER_LITERAL`
        elif token.type == TokenType.INTEGER_LITERAL:
            if PRINT_EAT_STACK:
                print("(Parser) Calling eat() from line", getframeinfo(currentframe()).lineno)
            self.eat(TokenType.INTEGER_LITERAL)
            return Num(token)

        # `FLOAT_CONST`
        elif token.type == TokenType.FLOAT_LITERAL:
            if PRINT_EAT_STACK:
                print("(Parser) Calling eat() from line", getframeinfo(currentframe()).lineno)
            self.eat(TokenType.FLOAT_LITERAL)
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

        statement -> compound_statement
                   | procedure_declaration
                   | procedure_call
                   | variable_declaration
                   | variable_assignment
                   | empty

        compound_statement -> `BEGIN` statement_list `END`

        procedure_declaration -> `DEFINITION` variable `LPAREN` formal_parameter_list `RPAREN` compound_statement
                               | `DEFINITION` variable `LPAREN` formal_parameter_list `RPAREN` `RETURNS_OP` type_spec compound_statement

        procedure_call -> variable `LPAREN` (empty | expr (`COMMA` expr)*) `RPAREN`

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

        factor -> `MINUS` factor
                | `INTEGER_LITERAL`
                | `FLOAT_CONST` 
                | `LPAREN` expr `RPAREN`
                | variable

        variable -> `IDENTIFIER`
        ```
        """
        node = self.program()

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
    def __init__(self, text: str):
        self.text_lines: list[str] = text.split('\n')
        self.current_scope: SymbolTable | None = None

    def error(self, error_code: ErrorCode, token: Token, message):
        """
        Create and raise a SemanticAnalyserError object
        """
        error = SemanticAnalyserError(error_code, message, token, surrounding_lines=self.text_lines)
        error.trigger()

    def visit_Program(self, node: Program):
        # Create new symbol tables for the program
        # Might move <builtins> declaration to a seperate place when modules are added
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

    def visit_ProcedureCall(self, node: ProcedureCall):
        for param_node in node.literal_params:
            self.visit(param_node)

    def visit_AssignOp(self, node: AssignOp):
        var_id = node.left.id
        var_symbol = self.current_scope.lookup(var_id)

        if var_symbol is None:
            self.error(
                error_code=ErrorCode.NAME_ERROR,
                token=node.left,
                message=f"Variable {repr(var_id)} was never initialised"
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
                message=f"Variable {repr(var_id)} was never initialised"
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

    def __init__(self):
        self.call_stack = CallStack()

    def interpret(self, tree: Node):
        """
        Initiates the recursive descent algorithm,
        generates a syntax tree,
        and executes the code.
        """
        return self.visit(tree)

    def visit_Program(self, node: Program):
        program_name = current_filename
        if program_name.endswith(".sap"):
            program_name = program_name[:-4]
        ar = ActivationRecord(
            name=program_name,
            ar_type=ActivationRecordType.PROGRAM,
            scope_level=1
        )

        self.call_stack.push(ar)

        for child in node.statements:
            self.visit(child)

        print(ar)

        self.call_stack.pop()

    def visit_Compound(self, node: Compound):
        for child in node.children:
            self.visit(child)

    def visit_VarDecl(self, node: VarDecl):
        variable_id = node.var_node.id
        variable_type_name = node.type_node.id

        current_ar = self.call_stack.peek()

        if node.expr_node is not None:
            var_value = self.visit(node.expr_node)

            current_ar.set(
                Member(
                    name=variable_id,
                    value=var_value,
                    datatype=variable_type_name
                )
            )
        else:
            current_ar.set(
                Member(
                    name=variable_id,
                    value=None,
                    datatype=variable_type_name
                )
            )

    def visit_ProcedureDecl(self, node):
        pass

    def visit_ProcedureCall(self, node):
        pass

    def visit_AssignOp(self, node: AssignOp):
        variable_id = node.left.id
        variable_value = self.visit(node.right)

        current_ar = self.call_stack.peek()
        current_ar.set(
            Member(
                variable_id,
                variable_value,
                "[DEV] Add typing already!"
            )
        )
        #if variable_id in self.global_scope:
        #    self.global_scope[variable_id][1] = self.visit(node.right)
        #else:
        #    raise ValueError("Interpreter :: Attempted to assign value to uninitialised variable!")

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
        current_ar = self.call_stack.peek()
        variable = current_ar.get(variable_id)
        if variable is None:
            raise NameError(f"Interpreter :: could not find {repr(variable_id)} in current frame")
        elif variable.value is None:
            raise NameError(f"Interpreter :: {repr(variable_id)} has no value!")
        else:
            return variable.value

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
            if os.path.isfile(path):
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

        #print()
        #print("Global vars (doesn't account for functions):")
        #print(interpreter.global_scope)
        #print()

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
