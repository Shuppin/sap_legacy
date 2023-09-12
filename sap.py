from __future__ import annotations

import logging

from collections    import defaultdict
from enum           import Enum
from inspect        import currentframe
from inspect        import getframeinfo
from os             import system
from os.path        import isfile
from os             import name as system_name
from sys            import argv
from sys            import stdout
from typing         import Any
from time           import time as current_time

from modules.builtin_methods    import mapping as builtin_method_mapping
from modules.config             import ConfigParser

from modules.operand        import *
from modules.builtin_types  import *

# TODO:
# * Make boolean type parsing a little stricter?
# * Add syntax warnings
# * Add package framework
#   - Run a file in module mode and spit out a symbol table,
#   - which is then fed into the main program
# * Add proper argument parsing for procedures (built-in and user defined)
# * Add procedure return functionality
# * Add more documentation
# * Add more built-in functions
#   - Add type parsing methods such as str() and int()
# * Add proper erroring for zero division
# * Add distinction between seperators and newlines
#   - In future this will allow for statements with curly brackets
#     to have the the brackets on a new line

__version__ = "1.0.0"

# Load config information first
config_path = "config.toml"
config = ConfigParser(config_path, override_logfile=True)

# Basic implementation of file name tracking
current_filename = "<cmdline>"


###########################################
#                                         #
#   Enums                                 #
#                                         #
###########################################

class TokenType(Enum):
    """
    Enum class which holds all the token types for  the SAP language
    """
    # These values do not represent how the lexer identifies tokens,
    # they are just represent what these tokens look like
    # symbols
    MULT            = '*'
    INTEGER_DIV     = '//'  # Currently not in use, may be removed in future
    FLOAT_DIV       = '/'
    PLUS            = '+'
    SUB             = '-'
    MOD             = "%"
    NOT             = '!'
    RETURNS_OP      = '->'
    LPAREN          = '('
    RPAREN          = ')'
    ASSIGN          = '='
    COLON           = ':'
    COMMA           = ','
    BEGIN           = '{'
    END             = '}'
    EQUAL           = '=='
    INEQUAL         = '~='
    LESS            = '<'
    LESSEQ          = '<='
    MORE            = '>'
    MOREEQ          = ">="
    # reserved keywords
    DEFINITION      = 'fn'
    INTEGER         = 'int'
    FLOAT           = 'float'
    BOOLEAN         = 'bool'
    STRING          = 'string'
    AND             = 'and'
    OR              = 'or'
    INCREMENT       = 'dec'
    DECREMENT       = 'inc'
    WHILE           = 'while'
    IF              = 'if'
    ELSEIF          = 'elseif'
    ELSE            = 'else'
    # dynamic token types
    SEP             = '<SEP>'  # Note, newline characters are also treated as seperators
    STRING_LITERAL  = '<STRING_LITERAL>'
    INTEGER_LITERAL = '<INTEGER_LITERAL>'
    FLOAT_LITERAL   = '<FLOAT_LITERAL>'
    BOOLEAN_LITERAL = '<BOOLEAN_LITERAL>'
    IDENTIFIER      = '<IDENTIFIER>'
    # other
    EOF             = '<EOF>'


class ActivationRecordType(Enum):
    """
    Enum class which holds all the types an activation record can have
    """
    PROGRAM     = "PROGRAM"
    PROCEDURE   = "PROCEDURE"


class ErrorCode(Enum):
    """
    Enum class which holds all the error code for the SAP language
    """
    SYNTAX_ERROR    = "SyntaxError"
    NAME_ERROR      = "NameError"
    TYPE_ERROR      = "TypeError"


###########################################
#                                         #
#   Data classes                          #
#                                         #
###########################################

class Token:
    """
    Token data class

    Simple data class to hold information about a token
    """
    def __init__(self, tokentype: TokenType, id: Any, lineno: int, linecol: int, startcol: int | None = None):
        self.type: TokenType = tokentype
        self.id: Any = id
        self.lineno: int = lineno
        self.linecol: int = linecol
        self.startcol: int | None = startcol
        log(f"Token.__init__(): created {str(self)}", level=LogLevel.ALL, stackoffset=1)

    def __str__(self) -> str:
        return f"Token[type = {self.type}, id = {repr(self.id)}, position = <{self.lineno}:{self.linecol}>]"

    def __repr__(self) -> str:
        return repr(self.__str__())


# Change the value to just return empty if not running on windows
# This prevents weird looking output on other platforms
_filter = lambda string: string if system_name == 'nt' else ''

class TermColour:
    BOLD    = _filter("\033[1m")
    CYAN    = _filter("\033[96m")
    DEFAULT = _filter("\033[0m")
    LIME    = _filter("\033[92m")
    RED     = _filter("\033[91m")
    WHITE   = _filter("\033[97m")

# Remove it after since we don't want it being used anywhere else
del _filter

class LogLevel:
    """
    Data class which holds all the logging levels used by the logger

    This avoids calling the config.get() function evertime we want to access a logging level
    """
    CRITICAL        = config.getint("logging.levels.CRITICAL")
    INFO            = config.getint("logging.levels.INFO")
    DEBUG           = config.getint("logging.levels.DEBUG")
    VERBOSE         = config.getint("logging.levels.VERBOSE")
    HIGHLY_VERBOSE  = config.getint("logging.levels.HIGHLY_VERBOSE")
    EAT_STACK       = config.getint("logging.levels.EAT_STACK")
    ALL             = config.getint("logging.levels.ALL")


class Member:
    """
    Member data class

    Data class to represent item within activation record
    """
    def __init__(self, name: str, value: Any):
        # May update to Symbol in future
        self.name: str = name
        self.value: Any = value

    def __str__(self):
        return f"<{self.value.__class__.__name__}> {self.name} = {str(self.value)}"


###########################################
#                                         #
#  Error handler                          #
#                                         #
###########################################

class BaseError(Exception):
    """
    Error base class
    Inherits from Exception, so it can be raised using python syntax
    """
    def __init__(
            self,
            error_code: ErrorCode,
            message: str,
            token: Token | list[Token] | None = None,
            position: list[int] = None,
            surrounding_lines: list[str] = None
    ):
        self.error_code: ErrorCode = error_code
        self.message: str = f' {TermColour.CYAN}{TermColour.BOLD}{self.__class__.__name__[:-5]} {TermColour.WHITE}:: {TermColour.RED}{self.error_code.value}{TermColour.DEFAULT}{TermColour.WHITE}: {message}{TermColour.DEFAULT}'
        self.token: Token | list[Token] | None= token
        self.surrounding_lines: list[str] | None = surrounding_lines
        # We need the position at which the error occurred,
        # It is either extracted from a given token or
        # passed directly as an array
        self.lineno: int
        self.linecol: int
        
        # Awful hacky code up ahead, you have been warned
        if isinstance(token, list):
            # Sort tokens by line number, then by column
            tokens = sorted(
                token,
                key = lambda token: (token.lineno, token.linecol)
            )
            self.lineno = tokens[-1].lineno
            self.linecol = tokens[-1].linecol
            
            # Filter out all tokens not on the same line, then select the first token in this new list
            first_token_on_same_line = list(
                filter(
                    lambda token: token.lineno == self.lineno,
                    tokens
                )
            )[0]
            
            # Access the start column if it exists,
            # otherwise grab the linecol and subtract the length of the token to achieve the same value
            start_col = first_token_on_same_line.startcol or first_token_on_same_line.linecol - len(first_token_on_same_line.id)
            
            self.token = Token(
                None,
                None,
                None,
                None,
                start_col
            )
        
        else:
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
        log(f"{type(self).__name__}.__init__() complete", stackoffset=1)

    def trigger(self):
        """
        Prints out error along with various bit of information.
        Output looks like this:
        ```
        File '.\\integers.sap', position <2:16>
        │ 1 │ int x = 5;
        │ 2 │ int y = x + 3 +
                            ^
        (Parser) SyntaxError: Expected type <IDENTIFIER> but got type <EOF>
        ```
        """

        # Checks if current output has utf encoding,
        # allowing for this special vertical bar character to be used
        if stdout.encoding.lower().startswith("utf"):
            VERT_BAR = "│"
        else:
            VERT_BAR = "|"
        
        # Warning: The following code has been heavily condensed and
        # is a mess to look at, however it saves about 100 lines of bloat

        # Create the surrounding lines and header of the error message
        error_message = [
            (f'{TermColour.BOLD}{TermColour.RED} Error{TermColour.WHITE} aborting execution due to error{TermColour.DEFAULT}\n\n'),
            # The if clauses here will ensure it prints the surrounding lines only if it exists
            (f" {TermColour.DEFAULT}{TermColour.CYAN}{' '*(len(str(self.lineno+2)))}-->{TermColour.WHITE} {current_filename}:{self.lineno}:{self.linecol}\n"),
            (f" {TermColour.CYAN}{TermColour.BOLD}{' '*(len(str(self.lineno+2)))} {VERT_BAR}\n"),
            (f" {TermColour.CYAN}{TermColour.BOLD}{' '*(len(str(self.lineno+2)) - len(str(self.lineno-3)))}{self.lineno-3} {VERT_BAR}{TermColour.DEFAULT} {self.surrounding_lines[self.lineno-4]}\n" if (self.lineno-4) >= 0 else f""),
            (f" {TermColour.CYAN}{TermColour.BOLD}{' '*(len(str(self.lineno+2)) - len(str(self.lineno-2)))}{self.lineno-2} {VERT_BAR}{TermColour.DEFAULT} {self.surrounding_lines[self.lineno-3]}\n" if (self.lineno-3) >= 0 else f""),
            (f" {TermColour.CYAN}{TermColour.BOLD}{' '*(len(str(self.lineno+2)) - len(str(self.lineno-1)))}{self.lineno-1} {VERT_BAR}{TermColour.DEFAULT} {self.surrounding_lines[self.lineno-2]}\n" if (self.lineno-2) >= 0 else f""),
            ]
        
        # Creates the message with the '~~~' highlighter, and the relevant text coloured in
        # For example:
        # | 1 | inte x = 4;
        #       ~~~~
        if self.surrounding_lines is not None and self.token is not None and self.token.startcol is not None:
            highlighted_line = f"{self.surrounding_lines[self.lineno-1][:self.token.startcol]}{TermColour.BOLD}{TermColour.LIME}{self.surrounding_lines[self.lineno-1][self.token.startcol:self.linecol]}{TermColour.DEFAULT}{self.surrounding_lines[self.lineno-1][self.linecol:]}"
            error_message.append(
                (f" {TermColour.CYAN}{TermColour.BOLD}{' '*(len(str(self.lineno+2)) - len(str(self.lineno  )))}{self.lineno  } {VERT_BAR}{TermColour.DEFAULT} {highlighted_line}\n"),
            )
            error_message.append(
                f" {TermColour.CYAN}{TermColour.BOLD}{' '*(len(str(self.lineno+2)))} {VERT_BAR}{' ' * self.token.startcol} {TermColour.LIME}{'~' * (self.linecol - self.token.startcol)}{TermColour.DEFAULT}\n\n"
            )
            error_message.append(self.message)
            error_message = "".join(error_message)  # Turn the list of lines into one big string
            
        # This code breaks under certain circumstances
        # Creates the message with the '^' highlighter, and the relevant text coloured in
        # For example:
        # | 1 | int x = 4 $ 3
        #                 ^
        elif self.surrounding_lines is not None:
            try:
                self.surrounding_lines[self.lineno-1][self.linecol]
                offset = 0
            except IndexError:
                offset = 1
            highlighted_line = f"{self.surrounding_lines[self.lineno-1][:self.linecol-offset]}{TermColour.BOLD}{TermColour.LIME}{self.surrounding_lines[self.lineno-1][self.linecol-offset]}{TermColour.DEFAULT}{self.surrounding_lines[self.lineno-1][self.linecol+1:]}"
            error_message.append(
                (f" {TermColour.CYAN}{TermColour.BOLD}{' '*(len(str(self.lineno+2)) - len(str(self.lineno  )))}{self.lineno  } {VERT_BAR}{TermColour.DEFAULT} {highlighted_line}\n"),
            )
            error_message.append(
                f" {TermColour.CYAN}{TermColour.BOLD}{' '*(len(str(self.lineno+2)))} {VERT_BAR}{' '*self.linecol}{TermColour.LIME}{' ' if offset==0 else ''}^{TermColour.DEFAULT}\n\n"
            )

            error_message.append(self.message)
            error_message = "".join(error_message)  # Turn the list of lines into one big string

        # If no surrounding lines were passed for whatever reason,
        # just print a little error message saying '<error fetching lines>' instead
        else:
            error_message = (f'{TermColour.BOLD}{TermColour.RED} Error{TermColour.WHITE} aborting execution due to error{TermColour.DEFAULT}\n\n'
                             f'{TermColour.BOLD}{TermColour.RED}<error fetching lines>{TermColour.DEFAULT}\n\n'
                             f'{self.message}')
        
        log(f"{type(self).__name__}: Successfully constructed error message")
        log(f"{type(self).__name__}: Program terminating with a success state", level=LogLevel.INFO)

        # Raise error or just print it normally
        if config.getbool("dev.raise_error_stack"):
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


class InterpreterError(BaseError):
    """
    For interpreter specific errors
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
    
    def _get_sub_tokens(self) -> list[Token]:
        tokens = []
        # Access the dir() of self and filter out any dunder (double under score e.g. '__init__') attributes
        # and anything that is not callable (i.e. methods)
        # Then, get the instance of those values and store them
        self_attributes = [getattr(self, a) for a in dir(self) if not a.startswith('__') and not callable(getattr(self, a))]
        
        # Go through attributes
        for attr in self_attributes:
            
            # Add any tokens we see
            if isinstance(attr, Token):
                tokens.append(attr)
            
            # Recursively explore any nodes we see
            elif issubclass(type(attr), Node):
                tokens += attr._get_sub_tokens()
                
            # A few nodes have lists, for now they are all one-dimensional
            # so this nested code should suffice
            elif isinstance(attr, list):
                
                for child in attr:
                    if isinstance(child, Token):
                        tokens.append(child)
                    elif issubclass(type(child), Node):
                        tokens += child._get_sub_tokens()
                
        return tokens

    def find_nth_token(self, n) -> Token:
        """
        Searches through all the tokens in this node and it's children
        then sorts it into order and return the `n`th item in that list
        """
        tokens = self._get_sub_tokens()
        # sort by line number, then line column
        tokens = sorted(
            tokens,
            key=lambda token: (token.lineno, token.linecol)
        )
        try:
            return tokens[n]
        except IndexError:
            return None
                

class InteriorNode(Node):
    """
    Interior nodes will always have other children
    nodes connected to them (like branches on a tree)
    """
    pass


class LeafNode(Node):
    """
    Leaf nodes do not have any children (like leaves on a tree)
    """
    def __init__(self, token: Token):
        self.token: Token = token


class Program(InteriorNode):
    """
    Program() represents a whole program
    """
    def __init__(self):
        self.statements: list[Node] = []


class Compound(InteriorNode):
    """
    Compound() represents a list of statements surrounded by curly brackets
    """
    def __init__(self):
        self.children: list[Node] = []


class VarDecl(InteriorNode):
    """
    VarDecl() represents a variable declaration statement
    """
    def __init__(
        self,
        type_node: TypeNode,
        var_node: VarNode,
        assign_op: Token = None,
        expr_node: Node = None
    ):
        self.type_node: TypeNode = type_node
        self.var_node: VarNode = var_node
        self.assign_op: Token | None = assign_op
        self.expr_node: Node | None = expr_node


class ProcedureDecl(InteriorNode):
    """
    ProcedureDecl() represents a procedure declaration statement
    """
    def __init__(
        self,
        procedure_var: VarNode,
        params: list[Param],
        compound_node: Compound,
        return_type: TypeNode = None
    ):
        self.procedure_var: VarNode = procedure_var
        self.params: list[Param] = params
        self.return_type: TypeNode | None = return_type
        self.compound_node: Compound = compound_node


class ProcedureCall(InteriorNode):
    """
    ProcedureCall() represents a procedure call statement
    """
    def __init__(self, procedure_var: VarNode, literal_params: list[Param]):
        self.procedure_var: VarNode = procedure_var
        self.literal_params: list[Param] = literal_params
        self.procedure_symbol: ProcedureSymbol | BuiltinProcedureSymbol | None = None


class WhileStatement(InteriorNode):
    """
    WhileStatement() represents a while loop statement
    """
    def __init__(self, condition: Node, compound: Compound):
        self.condition: Node = condition
        self.compound: Compound = compound


class SelectionStatement(InteriorNode):
    """
    SelectionStatement() represents a collection of condtionals e.g. if, elseif and else
    """
    def __init__(self, conditionals: list[Conditional], else_conditional: Conditional|None = None):
        self.conditionals: list[Conditional] = conditionals
        self.else_conditional: Compound = else_conditional


class IncrementalStatement(InteriorNode):
    """
    IncrementalStatement() represent a increment or decrement operation
    """
    def __init__(self, op, var):
        self.op: TokenType = op
        self.var: VarNode = var


class AssignOp(InteriorNode):
    """
    AssignOp() represents an assignment operation
    """
    def __init__(self, left, op, right):
        self.left: VarNode = left
        self.op: Token = op
        self.right: Node = right


class UnaryOp(InteriorNode):
    """
    UnaryOp() represents a unary operation (one-sided operation) such as `-1`
    """
    def __init__(self, op: Token, expr: Node):
        self.op: Token = op
        self.expr: Node = expr


class BinOp(InteriorNode):
    """
    BinOp() represents a binary operation (two-sided operation) such as `1+2`
    """
    def __init__(self, left: Node, op: Token, right: Node):
        self.left: Node = left
        self.op: Token = op
        self.right: Node = right


class Param(InteriorNode):
    """
    Param() represents a parameter within a procedure declaration
    """
    def __init__(self, var_node: VarNode, type_node: TypeNode):
        self.var_node: VarNode = var_node
        self.type_node: TypeNode = type_node


class Conditional(InteriorNode):
    """
    Conditional() represents an if, elseif or else statement
    """
    def __init__(self, condition: Node, compound: Compound):
        self.condition: Node = condition
        self.compound: Compound = compound


class NoOp(InteriorNode):
    """
    NoOp() represents an empty statement,
    for example there would be a NoOp between `;;` because semicolons
    act as separators, which need something to seperate.
    """
    pass


class TypeNode(LeafNode):
    """
    TypeNode() represents a data type literal
    """
    def __init__(self, token: Token):
        self.token: Token = token
        self.id: int | str | None | Type = self.token.type


class VarNode(LeafNode):
    """
    VarNode() represents a variable
    """
    def __init__(self, token: Token):
        self.token: Token = token
        self.id: int | str | None | Type = self.token.id


class NumNode(LeafNode):
    """
    NumNode() represents any number-like literal such as `23` or `3.14`
    """
    def __init__(self, token: Token):
        self.token: Token = token
        self.id: int | str | None | Type = self.token.id


class BoolNode(LeafNode):
    """
    BoolNode() represents boolean literals `True` and `False`
    """
    def __init__(self, token: Token):
        self.token: Token = token
        self.id: int | str | None | Type = self.token.id


class StringNode(LeafNode):
    """
    StringNode() represents string literals like "abcdef" and "Hello world"
    """
    def __init__(self, token: Token):
        self.token: Token = token
        self.id: int | str | None | Type = self.token.id
    

###########################################
#                                         #
#   Symbols                               #
#                                         #
###########################################

class BaseSymbol:
    """
    Symbol base class

    Symbols are the component which makes up a symbol table
    """
    def __init__(self, id):
        self.id = id
        self.scope_level = 0

    def __str__(self) -> str:
        return self.name

class BuiltinProcedureSymbol(BaseSymbol):
    def __init__(self, id: str, callable: callable):
        super().__init__(id)
        self.callable: callable = callable
        
    def __str__(self) -> str:
        return f"<builtin-proc> {repr(self.id)} <function {self.callable.__name__}>"


class VarSymbol(BaseSymbol):
    """
    Symbol which represents user-defined variables
    """
    def __init__(self, id, var_type):
        super().__init__(id)
        self.type: Type = var_type

    def __str__(self) -> str:
        return f"<variable> (id: {repr(self.id)}, type: {self.type})"


class ProcedureSymbol(BaseSymbol):
    """
    Symbol which represents procedure declarations
    """
    def __init__(self, name: str, procedure_node: ProcedureDecl):
        super().__init__(name)
        self.procedure_node: ProcedureDecl = procedure_node

    def __str__(self) -> str:
        if len(self.procedure_node.params) == 0:
            return f"<procedure> (id: {repr(self.id)}, parameters: <no params>)"
        else:
            # Okay, yes this is (slightly less) horrendous don't @me
            parameter_list = ', '.join(
                list(
                    map(
                        lambda param: f"({repr(param.var_node.id)}, {param.type_node.token.id})",
                        self.procedure_node.params
                    )
                )
            )
            return f"<procedure> (id: {repr(self.id)}, parameters: [{parameter_list}])"


###########################################
#                                         #
#   Symbol table code                     #
#                                         #
###########################################

class SymbolTable:
    """
    Class to store all the program symbols

    The symbol table is responsible for keeping track of semantics of symbols.
    It stores information about type, scope level and various other symbol-specific attributes.

    There are 3 types of symbols:
    - VarSymbols store information about variables.
    - BuiltinSymbols store information about built-in data types
    - ProcedureSymbols store information about a procedure's parameters, and the contents of the procedure
    """
    def __init__(self, scope_name, scope_level, parent_scope=None):
        self._symbols: dict[str, BaseSymbol] = {}
        self.scope_name: str = scope_name
        self.scope_level: int = scope_level
        self.parent_scope: SymbolTable | None = parent_scope

    def __str__(self) -> str:
        # Add header information
        text = "\nSCOPE (SCOPED SYMBOL TABLE):\n"
        text += f"Scope name    : {self.scope_name if self.scope_name.startswith('<') else repr(self.scope_name)}\n"
        text += f"Scope level   : {self.scope_level}\n"
        text += f"Parent scope  : {(self.parent_scope.scope_name if self.parent_scope.scope_name.startswith('<') else repr(self.parent_scope.scope_name)) if self.parent_scope else '<none>'}\n\n"
        text += "Scope symbol table contents\n"
        text += "---------------------------\n\n"

        # Organise contents of symbol table by symbol type.
        # This excludes the built-in symbol type, which is always printed first.
        symbols = defaultdict(list)
        
        # This is awful i know, TODO: Make this look neater
        stringified_keys = list(
            map(
                lambda key_value_tuple: (
                    getattr(key_value_tuple[0], "__name__", str(key_value_tuple[0])),  # Try to use __name__ attribute if present, else use str()
                    key_value_tuple[1]  # Leave the value as it is
                ),
                self._symbols.items()
            )
        )
        
        for _, val in sorted(stringified_keys):
            symbols[val.__class__.__name__].append(val)

        symbols: dict[str, list] = dict(symbols)

        # At this point `symbols` is a dictionary (dict[str, list])
        # containing the name of each (present) symbol type and
        # a list of all the objects which are that type.

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
        symbol.scope_level = self.scope_level
        log(f"SymbolTable {repr(self.scope_name)}: define {repr(str(symbol))} into scope {repr(self.scope_name)}", level=LogLevel.ALL)
        self._symbols[symbol.id] = symbol

    def lookup(self, symbol_name: str, search_parent_scopes: bool = True) -> BaseSymbol | None:
        """
        Will search for the given symbol name in `self._symbols` and
        then it will search its parent scopes.

        `search_parent_scopes` (bool): Determines whether the function will search in parent scopes. 
        """
        symbol = self._symbols.get(symbol_name)
        if symbol is not None:
            log(f"SymbolTable {repr(self.scope_name)}: lookup {repr(symbol_name)} returned {repr(str(symbol))} in scope {repr(self.scope_name)}", level=LogLevel.ALL)
            return symbol

        # Recursively search up the scopes to find symbols
        if self.parent_scope is not None and search_parent_scopes:
            return self.parent_scope.lookup(symbol_name)
        else:
            log(f"SymbolTable {repr(self.scope_name)}: lookup {repr(symbol_name)} returned None in scope {repr(self.scope_name)}", level=LogLevel.HIGHLY_VERBOSE)
            return None


###########################################
#                                         #
#   Memory system                         #
#                                         #
###########################################

class ActivationRecord:
    """
    A simple class which represents a change in scope, and information about the new scope.

    ActivationRecords make up the call stack, which is used to keep track of memory and current scope.
    """
    def __init__(self, name: str, ar_type: ActivationRecordType, nesting_level: int):
        self.name: str = name
        self.ar_type: ActivationRecordType = ar_type
        self.scope_level: int = nesting_level
        self._members: dict[str, Member] = {}

    def __str__(self):
        message = "\nACTIVATION RECORD:\n"
        message += f"Scope name    : {self.name if self.name.startswith('<') else repr(self.name)}\n"
        message += f"AR type       : {self.ar_type.value}\n"
        message += f"Nesting level : {self.scope_level}\n\n"
        message += "Activation record contents\n"
        message += "--------------------------\n"
        if self._members == {}:
            message += "\n    <empty>\n"
        else:
            member_objects: list[Member]
            _, member_objects = zip(*self._members.items())
            grouped_members = defaultdict(list)
            for member in member_objects:
                grouped_members[member.value.__class__.__name__].append(member)

            for _, members in grouped_members.items():
                message += "\n"
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
    """
    The call stack is responsible for keeping track of the program memory and
    all the scope information about the program in it's current state.

    It is made up of activation records which contain properties about each scope.

    The activation record at the top of the stack is the current scope.
    """
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
        """
        Return the value on the top of the stack without affecting the stack
        """
        return self._records[-1]

    def get(self, variable_id: str, search_up_stack: bool = True):
        pop_stack = []
        while len(self._records) > 0:
            current_ar = self.pop()
            pop_stack.append(current_ar)
            value = current_ar.get(variable_id)
            if value is not None:
                while len(pop_stack) > 0:
                    self._records.append(pop_stack.pop())
                return value

            if not search_up_stack:
                self._records.append(pop_stack.pop())
                return None
            
        while len(pop_stack) > 0:
            self._records.append(pop_stack.pop())

        return None


###########################################
#                                         #
#   Node visitor code                     #
#                                         #
###########################################

class NodeVisitor:
    """
    NodeVisitor base class

    Base class for all classes which visits/walks through the syntax tree generated by the parser.
    """
    def __init__(self):
        log("NodeVisitor.__init__() complete", stackoffset=1)

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
        if hasattr(node, 'token'):
            log(f"{type(self).__name__}: visiting {type(node).__name__} <{node.token.lineno}:{node.token.linecol}>", stackoffset=1, level=LogLevel.ALL)
        else:
            log(f"{type(self).__name__}: visiting {type(node).__name__}", stackoffset=1, level=LogLevel.ALL)
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
    def __init__(self, src):
        self.src: str = src
        self.pos: int = 0
        self.lineno: int = 1
        self.linecol: int = 0
        self.current_char: str | None = self.src[self.pos]

        # Just self.text split up into a list of each line
        # Used by the error reporter
        self.src_lines: list[str] = self.src.split('\n')

        # Used to keep track how many times
        # the lexer tries to get the next token
        # even though the pointer is at the end
        # of the code
        self.reached_end_counter = 0

        self.RESERVED_KEYWORDS: dict[str, tuple[TokenType, Any]] = {
            'fn'       :   (TokenType.DEFINITION, 'fn'),
            'while'     :   (TokenType.WHILE, 'while'),
            'if'        :   (TokenType.IF, 'if'),
            'elseif'    :   (TokenType.ELSEIF, 'elseif'),
            'else'      :   (TokenType.ELSE, 'else'),
            'and'       :   (TokenType.AND, 'and'),
            'or'        :   (TokenType.OR, 'or'),
            'inc'       :   (TokenType.INCREMENT, 'inc'),
            'dec'       :   (TokenType.DECREMENT, 'dec'),
            'int'       :   (TokenType.INTEGER, Int),
            'float'     :   (TokenType.FLOAT, Float),
            'bool'      :   (TokenType.BOOLEAN, Bool),
            'string'    :   (TokenType.STRING, String),
            'True'      :   (TokenType.BOOLEAN_LITERAL, Bool(1)),
            'False'     :   (TokenType.BOOLEAN_LITERAL, Bool(0))
        }
        log("Lexer.__init__(): created `RESERVED_KEYWORDS` table")
        log("Lexer.__init__() complete", stackoffset=1)

    # Utility functions

    def error(self, message=None, char_pos=None):
        """
        Create and raise a LexerError object
        """
        # Set default error message
        if message is None:
            message = f"Could not tokenise '{self.current_char}'"
        # Set default position
        if char_pos is None:
            char_pos = [self.lineno, self.linecol]
        error = LexerError(
            ErrorCode.SYNTAX_ERROR,
            message,
            position=char_pos,
            surrounding_lines=self.src_lines
        )
        log(f"Lexer: displaying SyntaxError: {repr(message)} at <{self.lineno}:{self.linecol}>", stackoffset=1)
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
        if self.pos > len(self.src) - 1:
            # After the lexer has tried multiple times to get
            # the next token while being at the end of the code.
            # This behaviour occurs when there is an error with
            # the lexical analysis stage.
            if self.reached_end_counter > 3:
                print("(Lexer) [CRITICAL] Lexer has reached end of code but is still trying to advance")
                log("Lexer: Lexer has reached end of code but is still trying to advance", level=LogLevel.CRITICAL)
                log("Lexer: Program terminating with an errored state", level=LogLevel.CRITICAL)
                exit()
            self.reached_end_counter += 1
            self.current_char = None
        # Else advance as normal
        else:
            self.current_char = self.src[self.pos]

    def peek(self) -> None | str:
        """Peeks at the next character in the code and returns it"""
        peek_pos = self.pos + 1
        if peek_pos > len(self.src) - 1:
            return None
        else:
            return self.src[peek_pos]

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
        # While the current char is alphanumeric or '_'
        while self.current_char is not None and (self.current_char.isalnum() or self.current_char == "_"):
            result += self.current_char
            self.advance()

        # Gets the token type associated with `result` if applicable, else default to `type.IDENTIFIER`
        token_type, type_ref = self.RESERVED_KEYWORDS.get(result, (TokenType.IDENTIFIER, None))
        
        if type_ref is None:
            type_ref = result

        token = Token(token_type, type_ref, self.lineno, self.linecol, startcol=start_pos)

        return token

    def number(self) -> Token:
        """Consumes a number from the input code and returns it"""
        number = ''
        start_pos = self.linecol
        # While the current char is a digit
        while self.current_char is not None and self.current_char.isdigit():
            number += self.current_char
            self.advance()

        # If number is longer than one character and startswith a 0
        if len(number) > 1 and number[0] == "0":
            self.error(
                message=f"Number cannot have leading zeros",
                char_pos=[self.lineno, start_pos]
            )

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

            if not has_decimals:
                self.error(
                    message="Incomplete float",
                    char_pos=[self.lineno, self.linecol-1]
                )

            token = Token(TokenType.FLOAT_LITERAL, Float(number), self.lineno, self.linecol, startcol=start_pos)

        else:
            token = Token(TokenType.INTEGER_LITERAL, Int(number), self.lineno, self.linecol, startcol=start_pos)

        return token

    def string(self) -> Token:
        
        start_pos = self.pos
        start_col = self.linecol
        start_line = self.lineno

        # Ignore opening '"'
        self.advance()
        
        string = ""
        
        while self.current_char != '"':
            if self.current_char == '\\' and self.peek() == '"':
                self.advance()
                self.advance()
                string += '"'
            elif self.current_char == '\n' or self.current_char is None:
                self.error(
                    message=f"Unterminated string literal at line {self.lineno}",
                    char_pos=[start_line, start_col]
                )
            else:
                string += self.current_char
                self.advance()
                
        # Skip over closing '"'
        self.advance()
        
        return Token(TokenType.STRING_LITERAL, String(string), self.lineno, self.linecol, start_pos)
            
    def get_next_token(self) -> Token:
        """
        Responsible for breaking down and extracting 
        tokens out of code.
        """
        while self.current_char is not None:
            
            # Special
            
            if self.current_char == ";" or self.current_char == "\n":
                token = Token(TokenType.SEP, self.current_char, self.lineno, self.linecol)
                self.advance()
                return token

            # Ignored characters

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

            elif self.current_char == '"':
                return self.string()
                
            # Operators

            elif self.current_char == "=":
                if self.peek() == "=":
                    token = Token(TokenType.EQUAL, '==', self.lineno, self.linecol+2, startcol=self.linecol)
                    self.advance()
                else:
                    token = Token(TokenType.ASSIGN, '=', self.lineno, self.linecol)
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
                    token = Token(TokenType.RETURNS_OP, "->", self.lineno, self.linecol+2, startcol=self.linecol)
                    self.advance()
                else:
                    token = Token(TokenType.SUB, self.current_char, self.lineno, self.linecol)
                self.advance()
                return token

            elif self.current_char == '%':
                token = Token(TokenType.MOD, self.current_char, self.lineno, self.linecol)
                self.advance()
                return token

            elif self.current_char == '!':
                token = Token(TokenType.NOT, self.current_char, self.lineno, self.linecol)
                self.advance()
                return token
            
            elif self.current_char == '~':
                if self.peek() == '=':
                    token = Token(TokenType.INEQUAL, '~=', self.lineno, self.linecol+2, startcol=self.linecol)
                    self.advance()
                    self.advance()
                    return token

            elif self.current_char == "<":
                if self.peek() == "=":
                    token = Token(TokenType.LESSEQ, "<=", self.lineno,self.linecol+2, startcol=self.linecol)
                    self.advance()
                else:
                    token = Token(TokenType.LESS, "<", self.lineno, self.linecol)
                self.advance()
                return token

            elif self.current_char == ">":
                if self.peek() == "=":
                    token = Token(TokenType.MOREEQ, ">=", self.lineno, self.linecol+2, startcol=self.linecol)
                    self.advance()
                else:
                    token = Token(TokenType.MORE, ">", self.lineno, self.linecol)
                self.advance()
                return token

            # Symbols

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

    For example give the set of tokens (equivalent to `2 + 2`):
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
                    token: Token[type = type.INTEGER_LITERAL, id = '2', start_pos = 0],
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

    def __init__(self, src):
        self.text: str = src
        self.lexer: Lexer = Lexer(self.text)
        log("Parser.__init__(): pre-loading first token")
        self.current_token: Token = self.lexer.get_next_token()
        # Previous token refers to the token before the current token
        # It is initially set to an empty token
        # Exclusively used by the error reporter
        log("Parser.__init__(): setting `self.previous_token` to empty token")
        self.previous_token: Token = Token(None, None, 0, 0)
        log("Parser.__init__() complete", stackoffset=1)

    def error(self, error_code: ErrorCode, token: Token, message):
        """
        Create and raise a ParserError object
        """
        log(f"Parser.error(): displaying {error_code.value}: {repr(message)} at <{token.lineno}:{token.linecol}>", stackoffset=1)
        error = ParserError(error_code, message, token=token, surrounding_lines=self.lexer.src_lines)
        error.trigger()

    def eat(self, expected_type: TokenType):
        """
        Compares the current token type to the expected
        type and, if equal, 'eat' the current
        token and move onto the next token.
        """
        log(f"Parser.eat(): (current type: {self.current_token.type}, expected type: {expected_type})", level=LogLevel.EAT_STACK, stackoffset=1)
        if self.current_token.type == expected_type:
            self.previous_token = self.current_token
            self.current_token = self.lexer.get_next_token()
        else:
            if expected_type.value.startswith("<") and expected_type.value.endswith(">"):
                expected_type_str = f"type {expected_type.value}"
            else:
                expected_type_str = f"'{expected_type.value}'"
            if expected_type == TokenType.END or expected_type == TokenType.EOF:
                self.error(
                    error_code=ErrorCode.SYNTAX_ERROR,
                    token=self.current_token,
                    message=f"Unexpected type <{self.current_token.type.name}>"
                )
            elif expected_type == TokenType.SEP:
                self.error(
                    error_code=ErrorCode.SYNTAX_ERROR,
                    token=self.current_token,
                    message=f"Expected {expected_type_str} but got {repr(self.current_token.id)}, perhaps you forgot a semicolon?"
                )
            else:
                self.error(
                    error_code=ErrorCode.SYNTAX_ERROR,
                    token=self.current_token,
                    message=f"Expected {expected_type_str} but got {repr(self.current_token.id)}"
                )

    def is_type(self) -> bool:
        """
        Check if the current token is a datatype
        """
        if self.current_token.type in [
            TokenType.INTEGER,
            TokenType.FLOAT,
            TokenType.BOOLEAN,
            TokenType.STRING,
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

        self.eat(TokenType.EOF)

        log(f"Parser: created {node.__class__.__name__}() <{self.current_token.lineno}:{self.current_token.linecol}> through {getframeinfo(currentframe()).function}()", level=LogLevel.ALL)
        return node

    def statement_list(self) -> list[Node]:
        """
        statement_list -> statement `SEMI`
                        | statement `SEMI` statement_list
        """
        node = self.statement()

        results = [node]

        while self.current_token.type == TokenType.SEP:
            self.eat(TokenType.SEP)
            statement = self.statement()
            results.append(statement)

        log(f"Parser: created list[Node]({len(results)}) <{self.current_token.lineno}:{self.current_token.linecol}> through {getframeinfo(currentframe()).function}()", level=LogLevel.ALL)
        return results

    def statement(self) -> Node:
        """
        statement -> compound_statement
                   | procedure_declaration
                   | procedure_call
                   | variable_declaration
                   | variable_assignment
                   | while_statement
                   | selection_statement
                   | incremental_statement
                   | empty
        """
        token = self.current_token.type
        if token == TokenType.BEGIN:
            node = self.compound_statement()
        elif token == TokenType.DEFINITION:
            node = self.procedure_declaration()
        elif self.is_type():
            node = self.variable_declaration()
        elif token == TokenType.IDENTIFIER:
            assert len(TokenType.LPAREN.value) == 1  # Assert that `LPAREN` is a single character symbol
            if self.lexer.current_char == TokenType.LPAREN.value:
                node = self.procedure_call()
            else:
                node = self.variable_assignment()
        elif token == TokenType.WHILE:
            node = self.while_statement()
        elif token == TokenType.IF:
            node = self.selection_statement()
        elif token == TokenType.INCREMENT or token == TokenType.DECREMENT:
            node = self.incremental_statement()
        else:
            node = self.empty()
        log(f"Parser: created {node.__class__.__name__}() <{self.current_token.lineno}:{self.current_token.linecol}> through {getframeinfo(currentframe()).function}()", level=LogLevel.ALL)
        return node

    def compound_statement(self) -> Compound:
        """
        compound_statement -> `BEGIN` statement_list `END`
        """
        self.eat(TokenType.BEGIN)
        nodes = self.statement_list()
        self.eat(TokenType.END)

        root = Compound()
        for node in nodes:
            root.children.append(node)

        log(f"Parser: created {root.__class__.__name__}()({len(root.children)}) <{self.current_token.lineno}:{self.current_token.linecol}> through {getframeinfo(currentframe()).function}()", level=LogLevel.ALL)
        return root

    def procedure_declaration(self) -> ProcedureDecl:
        """
        procedure_declaration -> `DEFINITION` variable `LPAREN` formal_parameter_list `RPAREN` compound_statement
                               | `DEFINITION` variable `LPAREN` formal_parameter_list `RPAREN` `RETURNS_OP` type_spec compound_statement
        """
        self.eat(TokenType.DEFINITION)

        procedure_var = self.variable()

        self.eat(TokenType.LPAREN)

        params = self.formal_parameter_list()

        self.eat(TokenType.RPAREN)

        if self.current_token.type == TokenType.BEGIN:

            body = self.compound_statement()

            proc_decl = ProcedureDecl(procedure_var, params, body)

        elif self.current_token.type == TokenType.RETURNS_OP:

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

        log(f"Parser: created {proc_decl.__class__.__name__}() <{self.current_token.lineno}:{self.current_token.linecol}>  through {getframeinfo(currentframe()).function}()", level=LogLevel.ALL)
        return proc_decl

    def procedure_call(self) -> ProcedureCall:
        """
        procedure_call -> variable `LPAREN` (empty | expr (`COMMA` expr)*) `RPAREN`
        """
        procedure_var = VarNode(
            self.current_token
        )

        self.eat(TokenType.IDENTIFIER)

        self.eat(TokenType.LPAREN)

        literal_params = []
        if self.current_token.type != TokenType.RPAREN:
            # Parameters can either be single variables, or full arithmetic expressions,
            # both of which are valid `expr()`'s
            literal_params.append(self.expr())

            # If there is a comma after the first param,
            # that means there should be more parameters after it
            while self.current_token.type == TokenType.COMMA:
                self.eat(TokenType.COMMA)
                literal_params.append(self.expr())

        self.eat(TokenType.RPAREN)

        node = ProcedureCall(procedure_var, literal_params)

        log(f"Parser: created {node.__class__.__name__}() <{self.current_token.lineno}:{self.current_token.linecol}> through {getframeinfo(currentframe()).function}()", level=LogLevel.ALL)
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
            self.eat(TokenType.ASSIGN)
            expr_node = self.expr()

            node = VarDecl(type_node, var_node, assign_op, expr_node)

        # type_spec variable 
        elif self.current_token.type == TokenType.SEP:
            node = VarDecl(type_node, var_node)

        # type_spec variable (`COMMA` variable)*
        elif self.current_token.type == TokenType.COMMA:
            node = Compound()
            node.children.append(VarDecl(type_node, var_node))
            while self.current_token.type == TokenType.COMMA:
                self.eat(TokenType.COMMA)
                var_node = self.variable()
                node.children.append(VarDecl(type_node, var_node))

        else:
            # If the token that resulted in this error is on the same line,
            # it is likely an issue inside the declaration statement
            if self.current_token.lineno == self.previous_token.lineno:
                self.error(
                    error_code=ErrorCode.SYNTAX_ERROR,
                    token=self.current_token,
                    message=f"Declaration statement has an invalid form"
                )
            # If it is on a different line, it is likely that the user has
            # just forgotten a semicolon
            else:
                self.error(
                    error_code=ErrorCode.SYNTAX_ERROR,
                    token=self.previous_token,
                    message=f"Declaration statement has an invalid form, perhaps you forgot a semicolon?"
                )

        log(f"Parser: created {node.__class__.__name__}() <{self.current_token.lineno}:{self.current_token.linecol}> through {getframeinfo(currentframe()).function}()", level=LogLevel.ALL)
        return node

    def variable_assignment(self) -> AssignOp:
        """
        variable_assignment -> variable `ASSIGN` expr
        """
        var_node = VarNode(self.current_token)
        self.eat(TokenType.IDENTIFIER)

        assign_op = self.current_token
        self.eat(TokenType.ASSIGN)

        right = self.expr()
        node = AssignOp(var_node, assign_op, right)

        log(f"Parser: created {node.__class__.__name__}() <{self.current_token.lineno}:{self.current_token.linecol}> through {getframeinfo(currentframe()).function}()", level=LogLevel.ALL)
        return node

    def while_statement(self) -> WhileStatement:
        """
        while_statement -> `WHILE` expr compound_statment
        """
        self.eat(TokenType.WHILE)
        return WhileStatement(
            condition=self.expr(),
            compound=self.compound_statement()
        )

    def selection_statement(self) -> SelectionStatement:
        """
        selection_statement -> `IF` expr compound_statement (`ELSEIF` expr compound_statement)*
                             | `IF` expr compound_statement (`ELSEIF` expr compound_statement)* `ELSE` expr compound_statement
        """
        # `IF` expr compound_statement
        self.eat(TokenType.IF)
        conditionals = [
            Conditional(
                condition=self.expr(),
                compound=self.compound_statement()
            )
        ]

        # (`ELSEIF` expr compound_statement)*
        while self.current_token.type == TokenType.ELSEIF:
            self.eat(TokenType.ELSEIF)
            conditionals.append(
                Conditional(
                    condition=self.expr(),
                    compound=self.compound_statement()
                )
            )

        # Initialise else conditional to None in case of no else statement.
        else_conditional = None

        # `ELSE` expr compound_statement
        if self.current_token.type == TokenType.ELSE:
            self.eat(TokenType.ELSE)
            else_conditional = self.compound_statement()

        if self.current_token.type in [TokenType.ELSE, TokenType.ELSEIF]:
            self.error(
                error_code=ErrorCode.SYNTAX_ERROR,
                token=self.current_token,
                message=f"Mismatched selection statement"
            )

        return SelectionStatement(
            conditionals=conditionals,
            else_conditional=else_conditional
        )

    def incremental_statement(self) -> IncrementalStatement:
        """
        incremental_statement -> (`INCREMENT` | `DECREMENT`) variable
        """
        op = self.current_token.type
        if op == TokenType.INCREMENT:
            self.eat(TokenType.INCREMENT)
        elif op == TokenType.DECREMENT:
            self.eat(TokenType.DECREMENT)
            
        var = self.current_token
        
        # Make sure the next token is actually a variable
        if var.type != TokenType.IDENTIFIER:
            self.error(
                error_code=ErrorCode.SYNTAX_ERROR,
                token=self.current_token,
                message=f"Can only {'increment' if op==TokenType.INCREMENT else 'decrement'} values on numerical-like variables"
            )
        
        # Eat token
        self.eat(TokenType.IDENTIFIER)
        
        # If the next token isn't a statement termination token,
        # then the variable is just the first component of an expression, which is not what we want.
        
        #                                 Statement termination tokens
        if self.current_token.type not in [TokenType.END, TokenType.SEP, TokenType.EOF]:
            self.error(
                error_code=ErrorCode.SYNTAX_ERROR,
                token=self.current_token,
                message=f"Unexpected <{self.current_token.type.name}>, can only {'increment' if op==TokenType.INCREMENT else 'decrement'} values on numerical-like variables"
            )
        
        return IncrementalStatement(
            op=op,
            var=VarNode(var)
        )

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
                self.eat(TokenType.COMMA)
                results.append(self.formal_parameter())

            # Commented out due to unknown behaviour
            #if self.current_token.type == TokenType.IDENTIFIER:
            #    self.error()

        log(f"Parser: created list[Param]({len(results)}) <{self.current_token.lineno}:{self.current_token.linecol}> through {getframeinfo(currentframe()).function}()", level=LogLevel.ALL)
        return results

    def formal_parameter(self) -> Param:
        """
        formal_parameter -> type_spec variable
        """
        type_node = self.type_spec()
        var_node = self.variable()

        param_node = Param(var_node, type_node)

        log(f"Parser: created {param_node.__class__.__name__}() <{self.current_token.lineno}:{self.current_token.linecol}> through {getframeinfo(currentframe()).function}()", level=LogLevel.ALL)
        return param_node

    def type_spec(self) -> TypeNode:
        """
        type_spec -> `INTEGER` | `FLOAT`
        """
        token = self.current_token
        if self.is_type():
            self.eat(token.type)
        else:
            self.error(
                error_code=ErrorCode.TYPE_ERROR,
                token=self.current_token,
                message=f"{repr(self.current_token.id)} is not a valid type!"
            )

        node = TypeNode(token)
        log(f"Parser: created {node.__class__.__name__}() <{self.current_token.lineno}:{self.current_token.linecol}> through {getframeinfo(currentframe()).function}()", level=LogLevel.ALL)
        return node

    def empty(self) -> NoOp:
        """
        empty ->
        """
        log(f"Parser: created NoOp() <{self.current_token.lineno}:{self.current_token.linecol}> through {getframeinfo(currentframe()).function}()", level=LogLevel.ALL)
        return NoOp()

    def expr(self) -> Node:
        """
        expr -> comp_expr ((`AND`|`OR`) comp_expr)*
        """
        node = self.comp_expr()
        
        # ((`AND`|`OR`) num_expr)*
        while self.current_token.type in (TokenType.AND, TokenType.OR):
            token = self.current_token
            
            if token.type == TokenType.AND:
                self.eat(TokenType.AND)
                
            elif token.type == TokenType.OR:
                self.eat(TokenType.OR)
                
            node = BinOp(
                left=node,
                op=token,
                right=self.comp_expr()
            )
            
        log(f"Parser: created {node.__class__.__name__}() <{self.current_token.lineno}:{self.current_token.linecol}> through {getframeinfo(currentframe()).function}()", level=LogLevel.ALL)
        return node
        
    def comp_expr(self) -> Node:
        """
        comp_expr -> num_expr ((`EQUAL`|`INEQUAL`|`LESS`|`MORE`|`LESSEQ`|`MOREEQ`) num_expr)*
        """
        node = self.num_expr()

        while self.current_token.type in [
            TokenType.EQUAL,
            TokenType.INEQUAL,
            TokenType.LESS,
            TokenType.MORE,
            TokenType.LESSEQ,
            TokenType.MOREEQ
        ]:
            token = self.current_token
            self.eat(token.type)

            node = BinOp(
                left=node,
                op=token,
                right=self.num_expr()
            )

        log(f"Parser: created {node.__class__.__name__}() <{self.current_token.lineno}:{self.current_token.linecol}> through {getframeinfo(currentframe()).function}()", level=LogLevel.ALL)
        return node
        
    def num_expr(self) -> Node:
        """
        num_expr -> mod_expr ((`PLUS`|`MINUS`) mod_expr)*
        """
        node = self.mod_expr()

        # ((`PLUS`|`MINUS`) mod_expr)*
        while self.current_token.type in (TokenType.PLUS, TokenType.SUB):
            token = self.current_token

            if token.type == TokenType.PLUS:
                self.eat(TokenType.PLUS)

            elif token.type == TokenType.SUB:
                self.eat(TokenType.SUB)

            node = BinOp(
                left=node,
                op=token,
                right=self.mod_expr()
            )

        log(f"Parser: created {node.__class__.__name__}() <{self.current_token.lineno}:{self.current_token.linecol}> through {getframeinfo(currentframe()).function}()", level=LogLevel.ALL)
        return node

    def mod_expr(self) -> Node:
        """
        mod_expr -> term (`MOD` term)*
        """
        node = self.term()

        # (`MOD` term)*
        while self.current_token.type == TokenType.MOD:
            token = self.current_token

            if token.type == TokenType.MOD:
                self.eat(TokenType.MOD)

            node = BinOp(
                left=node,
                op=token,
                right=self.term()
            )

        log(f"Parser: created {node.__class__.__name__}() <{self.current_token.lineno}:{self.current_token.linecol}> through {getframeinfo(currentframe()).function}()", level=LogLevel.ALL)
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
                self.eat(TokenType.MULT)

            elif token.type == TokenType.INTEGER_DIV:
                self.eat(TokenType.INTEGER_DIV)

            elif token.type == TokenType.FLOAT_DIV:
                self.eat(TokenType.FLOAT_DIV)

            node = BinOp(left=node, op=token, right=self.factor())

        log(f"Parser: created {node.__class__.__name__}() <{self.current_token.lineno}:{self.current_token.linecol}> through {getframeinfo(currentframe()).function}()", level=LogLevel.ALL)
        return node

    def factor(self) -> Node:
        """
        factor -> `MINUS` factor
                | `NOT` factor
                | `INTEGER_LITERAL`
                | `FLOAT_LITERAL` 
                | `BOOLEAN_LITERAL`
                | `STRING_LITERAL`
                | `LPAREN` expr `RPAREN`
                | variable
        """
        token = self.current_token

        # `MINUS` factor
        if token.type == TokenType.SUB:
            self.eat(TokenType.SUB)
            node = UnaryOp(token, self.factor())
            
        # `NOT` factor
        elif token.type == TokenType.NOT:
            self.eat(TokenType.NOT)
            node = UnaryOp(token, self.factor())

        # `INTEGER_LITERAL`
        elif token.type == TokenType.INTEGER_LITERAL:
            self.eat(TokenType.INTEGER_LITERAL)
            node = NumNode(token)

        # `FLOAT_CONST`
        elif token.type == TokenType.FLOAT_LITERAL:
            self.eat(TokenType.FLOAT_LITERAL)
            node = NumNode(token)

        # `BOOLEAN_LITERAL`
        elif token.type == TokenType.BOOLEAN_LITERAL:
            self.eat(TokenType.BOOLEAN_LITERAL)
            node = BoolNode(token)

        # `STRING_LITERAL`
        elif token.type == TokenType.STRING_LITERAL:
            self.eat(TokenType.STRING_LITERAL)
            node = StringNode(token)

        # `LPAREN` expr `RPAREN`
        elif token.type == TokenType.LPAREN:
            self.eat(TokenType.LPAREN)
            node = self.expr()
            self.eat(TokenType.RPAREN)

        # variable
        else:
            node = self.variable()

        log(f"Parser: created {node.__class__.__name__}() <{self.current_token.lineno}:{self.current_token.linecol}> through {getframeinfo(currentframe()).function}()", level=LogLevel.ALL)
        return node

    def variable(self) -> VarNode:
        """
        variable -> `IDENTIFIER`
        """
        node = VarNode(self.current_token)
        self.eat(TokenType.IDENTIFIER)
        log(f"Parser: created {node.__class__.__name__}() <{self.current_token.lineno}:{self.current_token.linecol}> through {getframeinfo(currentframe()).function}()", level=LogLevel.ALL)
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
                   | while_statement
                   | selection_statement
                   | incremental_statement
                   | empty

        compound_statement -> `BEGIN` statement_list `END`

        procedure_declaration -> `DEFINITION` variable `LPAREN` formal_parameter_list `RPAREN` compound_statement
                               | `DEFINITION` variable `LPAREN` formal_parameter_list `RPAREN` `RETURNS_OP` type_spec compound_statement

        procedure_call -> variable `LPAREN` (empty | expr (`COMMA` expr)*) `RPAREN`

        variable_declaration -> type_spec variable `ASSIGN` expr
                              | type_spec variable (`COMMA` variable)*

        variable_assignment -> variable `ASSIGN` expr

        while_statement -> `WHILE` expr compound_statement

        selection_statement -> `IF` expr compound_statement (`ELSEIF` expr compound_statement)*
                             | `IF` expr compound_statement (`ELSEIF` expr compound_statement)* `ELSE` expr compound_statement

        incremental_statement -> (`INCREMENT` | `DECREMENT`) variable

        formal_parameter_list -> formal_parameter
                               | formal_parameter `COMMA` formal_parameter_list
                               | empty

        formal_parameter -> type_spec variable

        type_spec -> `INTEGER` | `FLOAT` | `BOOL`

        empty ->
        // What did you expect cuh
                       
        expr -> comp_expr ((`AND`|`OR`) comp_expr)*

        comp_expr -> num_expr ((`EQUAL`|`INEQUAL`|`LESS`|`MORE`|`LESSEQ`|`MOREEQ`) num_expr)*
    
        num_expr -> mod_expr ((`PLUS`|`MINUS`) mod_expr)*

        mod_expr -> term (`MOD` term)*

        term -> factor ((`MUL`|`INTEGER_DIV`|`FLOAT_DIV`) factor)*

        factor -> `MINUS` factor
                | `NOT` factor
                | `INTEGER_LITERAL`
                | `FLOAT_LITERAL`
                | `BOOLEAN_LITERAL` 
                | `LPAREN` expr `RPAREN`
                | variable

        variable -> `IDENTIFIER`
        ```
        """
        log(f"Parser.parse(): parsing file {repr(current_filename)}")
        node = self.program()
        log(f"Parser.parse(): finished parsing file {repr(current_filename)}")
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
    def __init__(self, src: str, builtin_scope: SymbolTable):
        super().__init__()
        self.src_lines: list[str] = src.split('\n')
        self.builtin_scope = builtin_scope
        self.current_scope: SymbolTable | None = None
        log("SemanticAnalyser.__init__() complete", stackoffset=1)

    def error(self, error_code: ErrorCode, token: Token, message):
        """
        Create and raise a SemanticAnalyserError object
        """
        log(f"SemanticAnalyser: displaying {error_code.value}: {repr(message)} at <{token.lineno}:{token.linecol}>")
        error = SemanticAnalyserError(error_code, message, token, surrounding_lines=self.src_lines)
        error.trigger()

    def analyse(self, tree: Node):
        """
        Performs semantic analysis before executing the code
        """
        log("SemanticAnalyser: Performing analysis...")
        self.visit(tree)
        log("SemanticAnalyser: Analysis complete!")

    def visit_Program(self, node: Program):
        
        # Log the contents of the built-ins symbol table
        log(f"SemanticAnalyser: Loaded {repr(self.builtin_scope.scope_name)} symbol table", level=LogLevel.VERBOSE)
        log("SemanticAnalyser: SCOPE", repr(self.builtin_scope.scope_name), level=LogLevel.HIGHLY_VERBOSE)
        log(str(self.builtin_scope), level=LogLevel.HIGHLY_VERBOSE, prefix_per_line=" |   ")
        log("SemanticAnalyser: SCOPE", repr(self.builtin_scope.scope_name), "END", level=LogLevel.HIGHLY_VERBOSE)
        
        # Create global scoped symbol table
        log("SemanticAnalyser: Creating '<global>' symbol table", level=LogLevel.VERBOSE)
        global_scope = SymbolTable(scope_name="<global>", scope_level=1, parent_scope=self.builtin_scope)
        self.current_scope = global_scope
        
        log("SemanticAnalyser: Base symbol tables created, proceeding to analyse tree...", level=LogLevel.VERBOSE)

        for child in node.statements:
            self.visit(child)

        log("SemanticAnalyser: Finished analysing tree!", level=LogLevel.VERBOSE)

        # Log the contents of the global scoped symbol table after the program has been analysed
        log("SemanticAnalyser: SCOPE", repr(global_scope.scope_name), level=LogLevel.HIGHLY_VERBOSE)
        log(str(global_scope), level=LogLevel.HIGHLY_VERBOSE, prefix_per_line=" |   ")
        log("SemanticAnalyser: SCOPE", repr(global_scope.scope_name), "END", level=LogLevel.HIGHLY_VERBOSE)

        # Return to global scope
        self.current_scope = global_scope

    def visit_Compound(self, node: Compound):
        for child in node.children:
            self.visit(child)

    def visit_VarDecl(self, node: VarDecl):
        self.visit(node.type_node)

        var_id = node.var_node.id
        var_type = node.type_node.token.id
        
        if var_id == "_":
            return

        if self.current_scope.lookup(var_id, search_parent_scopes=False) is not None:
            self.error(
                error_code=ErrorCode.NAME_ERROR,
                token=node.var_node.token,
                message="Cannot initialise variable with same name"
            )

        var_symbol = VarSymbol(var_id, var_type)
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

        proc_symbol = ProcedureSymbol(proc_name, node)
        self.current_scope.define(proc_symbol)

        log(f"SemanticAnalyser: Creating {repr(proc_name)} symbol table", level=LogLevel.VERBOSE)
        proc_scope = SymbolTable(scope_name=proc_name, scope_level=self.current_scope.scope_level + 1,
                                 parent_scope=self.current_scope)
        self.current_scope = proc_scope

        for param in proc_params:
            param_name = param.var_node.id
            param_type = param.type_node.token.id
            var_symbol = VarSymbol(param_name, param_type)
            self.current_scope.define(var_symbol)

        self.visit(node.compound_node)
        log(f"SemanticAnalyser: Successfully created {repr(proc_name)} symbol table", level=LogLevel.VERBOSE)
        log("SemanticAnalyser: SCOPE", repr(self.current_scope.scope_name), level=LogLevel.HIGHLY_VERBOSE)
        log(str(self.current_scope), level=LogLevel.HIGHLY_VERBOSE, prefix_per_line=" |   ")
        log("SemanticAnalyser: SCOPE", repr(self.current_scope.scope_name), "END", level=LogLevel.HIGHLY_VERBOSE)

        # Return to parent scope
        self.current_scope = self.current_scope.parent_scope

    def visit_ProcedureCall(self, node: ProcedureCall):
        procedure_id = node.procedure_var.id
        procedure_symbol: ProcedureSymbol = self.current_scope.lookup(procedure_id)
        if procedure_symbol is None:
            self.error(
                error_code=ErrorCode.NAME_ERROR,
                token=node.procedure_var.token,
                message=f"Procedure {repr(procedure_id)} was never initialised"
            )

        for param_node in node.literal_params:
            self.visit(param_node)

        node.procedure_symbol = procedure_symbol

    def visit_WhileStatement(self, node: WhileStatement):
        self.visit(node.condition)
        self.visit(node.compound)

    def visit_SelectionStatement(self, node: SelectionStatement):
        for conditional in node.conditionals:
            self.visit(conditional.condition)
            self.visit(conditional.compound)
            
        if node.else_conditional is not None:
            self.visit(node.else_conditional)
    
    def visit_IncrementalStatement(self, node: IncrementalStatement):
        self.visit(node.var)
    
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
        type_id = node.token.id

        if not issubclass(type_id, Type):
            self.error(
                error_code=ErrorCode.NAME_ERROR,
                token=node.token,
                message=f"Unrecognised type {repr(type_id.__name__)}"
            )
        else:
            return type_id

    def visit_VarNode(self, node: VarNode):
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

    def visit_NumNode(self, node):
        pass

    def visit_BoolNode(self, node):
        pass

    def visit_StringNode(self, node):
        pass

    def visit_NoOp(self, node):
        pass


###########################################
#                                         #
#   Interpreter code                      #
#                                         #
###########################################

# Currently some (even more slightly less) unloved garbárge
class Interpreter(NodeVisitor):
    """
    Main interpreter class

    The interpreter is responsible for processing abstract syntax trees
    and compiling (not machine code) them into a final result.
    It works by 'visiting' each node in the tree and processing it based on its attributes and surrounding nodes.

    It also handles type-checking at runtime (for now).
    """
    def __init__(self, src: str):
        super().__init__()
        self.call_stack = CallStack()
        self.src_lines: list[str] = src.split('\n')
        log("Interpreter.__init__() complete", stackoffset=1)

    def interpret(self, tree: Node):
        """
        Initiates the recursive descent algorithm and executes the code.
        """
        return self.visit(tree)
    
    def error(self, error_code: ErrorCode, token: Token, message):
        """
        Create and raise a InterpreterError object
        """
        log(f"Interpreter: displaying {error_code.value}: {repr(message)} at <{token.lineno}:{token.linecol}>")
        error = InterpreterError(error_code, message, token, surrounding_lines=self.src_lines)
        error.trigger()
        
    def execute_builtin_procedure(self, procedure_call: ProcedureCall):
        func = procedure_call.procedure_symbol.callable
        # visit/evaluate each parameter and map it to a new list
        params = list(map(
            lambda param: self.visit(param),
            procedure_call.literal_params
        ))
        # Actually execute the function
        func(*params)

    def visit_Program(self, node: Program):
        log(f"Interpreter: Interpreting tree", repr(node))
        ar = ActivationRecord(
            name="<program>",
            ar_type=ActivationRecordType.PROGRAM,
            nesting_level=1
        )
        log(f"Interpreter: created AR", repr(ar.name), repr(ar), level=LogLevel.VERBOSE)

        self.call_stack.push(ar)

        log(f"Interpreter: pushed AR {repr(ar.name)} onto call stack", level=LogLevel.VERBOSE)

        for child in node.statements:
            self.visit(child)

        log("Interpreter: ACTIVATION RECORD", repr(ar.name), level=LogLevel.HIGHLY_VERBOSE)
        log(str(ar), level=LogLevel.HIGHLY_VERBOSE, prefix_per_line=" |   ")
        log("Interpreter: ACTIVATION RECORD", repr(ar.name), "END", level=LogLevel.HIGHLY_VERBOSE)

        self.call_stack.pop()
        log(f"Interpreter: lifted AR {repr(ar.name)} from call stack", level=LogLevel.VERBOSE)
        log(f"Interpreter: Finished interpreting tree")

    def visit_Compound(self, node: Compound):
        for child in node.children:
            self.visit(child)

    def visit_VarDecl(self, node: VarDecl):
        variable_id = node.var_node.id
        variable_type = node.type_node.token.id

        # "_" is a place holder variable so we can ignore it
        if variable_id == "_":
            return

        current_ar = self.call_stack.peek()

        # If the variable actually has an expression to evaluate and assign.
        if node.expr_node is not None:
            expression_value = self.visit(node.expr_node)
            
            # Ensure type is correct
            if variable_type != type(expression_value):
                
                # If not, attempt to parse to expected type
                new_expression_value = expression_value.to(variable_type)
                
                # If that fails, error
                if new_expression_value is None:
                    self.error(
                        error_code=ErrorCode.TYPE_ERROR,
                        token=node.var_node.token,
                        message=f"Attempted to assign expression with type <{type(expression_value).__name__}> to var with incompatible type <{variable_type.__name__}>"
                    )
                    
                # A new var is created to ensure error prints correct type 
                expression_value = new_expression_value

            current_ar.set(
                Member(
                    name=variable_id,
                    value=expression_value
                )
            )
            log(f"Interpreter: VarDecl <{variable_type.__name__}> {repr(variable_id)} = {expression_value}", level=LogLevel.ALL)

        # Else just give the variable a NoneType value
        else:
            current_ar.set(
                Member(
                    name=variable_id,
                    value=NoneType()
                )
            )
            log(f"Interpreter: VarDecl <{variable_type.__name__}> {repr(variable_id)} = NoneType()", level=LogLevel.ALL)

    def visit_ProcedureDecl(self, node):
        # No code needed since all information we need
        # about a procedure is stored in the symbol table.
        # Function exists so that walking the tree does not error
        pass

    def visit_ProcedureCall(self, node: ProcedureCall):
        
        # Check if the procedure is built-in, and execute a special function if so
        if isinstance(node.procedure_symbol, BuiltinProcedureSymbol):
            self.execute_builtin_procedure(node)
            return
        
        procedure_name = node.procedure_var.id
        log(f'Interpreter: calling procedure {repr(procedure_name)}', level=LogLevel.VERBOSE)

        # Create the activation record for the procedure we are about to enter
        ar = ActivationRecord(
            name=procedure_name,
            ar_type=ActivationRecordType.PROCEDURE,
            nesting_level=node.procedure_symbol.scope_level + 1
        )
        
        log(f"Interpreter: created AR", repr(ar.name), repr(ar), level=LogLevel.VERBOSE)

        # TODO: Add proper argument vs parameters checking
        # i.e. Type comparison and length comparison
        
        # Grab parameters from the procedure call and the procedure declaration
        formal_params = node.procedure_symbol.procedure_node.params
        literal_params = node.literal_params

        for formal_param, literal_param in zip(formal_params, literal_params):
            # Evaluate each parameter in the procedure call
            param_value = self.visit(literal_param)
            # Add it to the scope of the procedure we are about to enter
            ar.set(
                Member(
                    name=formal_param.var_node.id,
                    value=param_value
                )
            )
            log(f"Interpreter: ProcedureCall param {formal_param.type_node.token.id} {repr(formal_param.var_node.id)} = {repr(param_value)}", level=LogLevel.HIGHLY_VERBOSE)

        # Push the new scope to the call stack
        self.call_stack.push(ar)
        log(f"Interpreter: pushed AR {repr(ar.name)} onto call stack", level=LogLevel.VERBOSE)

        log(f"Interpreter: executing procedure {repr(procedure_name)}", level=LogLevel.VERBOSE)
        # Enter procedure
        self.visit(node.procedure_symbol.procedure_node.compound_node)
        log(f"Interpreter: execution complete for procedure {repr(procedure_name)}", level=LogLevel.VERBOSE)

        # Log activation record
        log("Interpreter: ACTIVATION RECORD", repr(ar.name), level=LogLevel.HIGHLY_VERBOSE)
        log(str(ar), level=LogLevel.HIGHLY_VERBOSE, prefix_per_line=" |   ")
        log("Interpreter: ACTIVATION RECORD", repr(ar.name), "END", level=LogLevel.HIGHLY_VERBOSE)

        # Once finished, remove the scope from the call stack and return to our original scope
        self.call_stack.pop()
        log(f"Interpreter: lifted AR {repr(ar.name)} from call stack", level=LogLevel.VERBOSE)

    def visit_WhileStatement(self, node: WhileStatement):
        def eval_condition():
            evaluated_value = self.visit(node.condition)
        
            # Parse code
            if not isinstance(evaluated_value, Bool):
                # Attempt to parse to bool
                evaluated_value = evaluated_value.to(Bool)
                
                # If this returns None, the parse has failed, so error
                if evaluated_value is None:
                    self.error(
                        error_code=ErrorCode.TYPE_ERROR,
                        token=node.condition.find_nth_token(0),
                        message=f"Could not parse expression to Bool"
                    )
            
            # If the value is equal to 1, then the condition is True
            if evaluated_value.value == 1:
                return True
            
            return False
        
        # Interpreted languages written in interpeted languages are so funny
        while eval_condition():
            self.visit(node.compound)

    def visit_SelectionStatement(self, node: SelectionStatement):
        
        visted_conditional = False
        
        # Iterate through all the conditions (in order of appearance) and test if they're true.
        for conditional in node.conditionals:
            evaluated_value = self.visit(conditional.condition)
            
            if not isinstance(evaluated_value, Bool):
                
                # Attempt to parse to bool
                evaluated_value = evaluated_value.to(Bool)
                
                if evaluated_value is None:
                    self.error(
                        error_code=ErrorCode.TYPE_ERROR,
                        token=conditional.condition.find_nth_token(0),
                        message=f"Could not parse expression to Bool"
                    )
            
            # If the evaluated expression it true, execute it!
            if evaluated_value.value == 1:
                visted_conditional = True
                self.visit(conditional.compound)
                log(f"Interpreter: SelectionStatement selected if condition", level=LogLevel.ALL)
                break
            
        if node.else_conditional is not None and not visted_conditional:
            log(f"Interpreter: SelectionStatement selected else condition", level=LogLevel.ALL)
            self.visit(node.else_conditional)

    def visit_IncrementalStatement(self, node: IncrementalStatement):
        # Decide what operator to pass in operand()
        op_str = '+' if node.op == TokenType.INCREMENT else '-'
        
        # Get the value of the variable
        original_value = self.visit(node.var)
        
        # Perform the operation on it
        value = operand(op_str, original_value, Int(1))
        
        # If this returns none, the operation was between invalid datatypes, so error
        if value is None:
            self.error(
                error_code=ErrorCode.TYPE_ERROR,
                token=node.var.token,
                message=f"Cannot {'increment' if node.op==TokenType.INCREMENT else 'decrement'} variable of type {original_value.__class__.__name__}"
            )
        
        # If the type of the new value doesn't match the type of the old value,
        # attempt to parse it back to the oriognal value
        if type(original_value) != type(value):
            value = value.to(type(original_value))
            
            # Not sure if this code is reachable
            if value is None:
                self.error(
                    error_code=ErrorCode.TYPE_ERROR,
                    token=node.var.token,
                    message=f"Could not parse {repr(node.var.id)} back to original type after {'incrementing' if node.op==TokenType.INCREMENT else 'decrementing'}"
                )
        
        # Access the current record
        current_ar = self.call_stack.peek()
        
        # Update current variable on activation record
        current_ar.set(
            Member(
                node.var.id,
                value
            )
        )

    def visit_AssignOp(self, node: AssignOp):
        current_ar = self.call_stack.peek()
        variable = self.visit(node.left)
        variable_type = type(variable)
        variable_id = node.left.id
        evaluated_value = self.visit(node.right)
        
        # Used for error printing
        original_value_name = type(evaluated_value).__name__
        
        # Ensure type is correct
        if variable_type != type(evaluated_value) and variable_type != NoneType:
            
            # Attempt to parse to expected type
            evaluated_value = evaluated_value.to(variable_type)
            
            # If that fails, error
            if evaluated_value is None:
                self.error(
                    error_code=ErrorCode.TYPE_ERROR,
                    token=node.left,
                    message=f"Attempted to assign value with type <{original_value_name}> to var with incompatible type <{variable_type.__name__}>"
                )
                
        # Else, continue on as normal
                
        # Update current variable on activation record
        current_ar.set(
            Member(
                variable_id,
                evaluated_value
            )
        )
        log(f"Interpreter: AssignOp <{variable_type.__name__}> {repr(variable_id)} = {evaluated_value}", level=LogLevel.ALL)

    def visit_UnaryOp(self, node: UnaryOp):
        # Negate operation
        if node.op.type == TokenType.SUB:
            value = self.visit(node.expr)
            result = operand('-', value)
            
            if result is None:
                self.error(
                    error_code=ErrorCode.TYPE_ERROR,
                    token=node.op,
                    message=f"Unsupported operation {repr(node.op.id)} for <{type(value).__name__}>"
                )
                
            return result
        
        # Boolean not operation
        elif node.op.type == TokenType.NOT:
            value = self.visit(node.expr)
            original_value_name = type(value).__name__
            
            if not isinstance(value, Bool):
                value = value.to(Bool)
                
            result = operand('not', value)
                
            if result is None:
                self.error(
                    error_code=ErrorCode.TYPE_ERROR,
                    token=node.op,
                    message=f"Unsupported operation {repr(node.op.id)} for type <{original_value_name}>"
                )
                
            return result

    def visit_BinOp(self, node: BinOp):
        
        left_value = self.visit(node.left)
        right_value = self.visit(node.right)
        
        op_str = node.op.id
            
        value = operand(op_str, left_value, right_value)

        if value is None:
            self.error(
                error_code=ErrorCode.TYPE_ERROR,
                token=node.op,
                message=f"Unsupported operation {repr(node.op.id)} between <{type(left_value).__name__}> and <{type(right_value).__name__}>"
            )
            
        return value

    def visit_TypeNode(self, node: TypeNode):
        # Already checked by semantic analyser so we don't need to worry about this
        pass

    def visit_VarNode(self, node: VarNode):
        variable_id = node.id
        variable = self.call_stack.get(variable_id)
        if variable is None:
            # The semantic analyser would miss this check if variables are
            # defined in conditional statements (which is very bad practice)
            self.error(
                error_code=ErrorCode.NAME_ERROR,
                token=node.token,
                message=f"Variable {repr(variable_id)} does not exist!"
            )
        # Not sure if this states is even reachable
        # since semantic analyser *should* catch errors like this
        elif variable.value is None:
            self.error(
                error_code=ErrorCode.NAME_ERROR,
                token=node.token,
                message=f"Variable {repr(variable_id)} has no value!"
            )
        else:
            return variable.value

    def visit_NumNode(self, node: NumNode):
        return node.id

    def visit_BoolNode(self, node: BoolNode):
        return node.id

    def visit_StringNode(self, node: StringNode):
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
    Driver code to initiate and execute all the classes in the program.
    """
    def __init__(self):
        self.filename: str = config.getstr("dev.default_filename")
        self.mode: str = config.getstr("behaviour.read_mode")
        log("Driver.__init__() complete")

    def run_program(self):
        """
        Calls the relevant function for the given mode
        """
        log("Driver.run_program(): executing")
        if self.mode == "file":
            self._process_arguments()

        # self._process_arguments() modifies self.mode and self.filename,
        # so we need to check self.mode again
        if self.mode == "cmdline":
            self.cmdline_input()
        elif self.mode == "file":
            # Set the global filename, used by error handler
            global current_filename
            current_filename = self.filename
            log(f"Driver.run_program(): Set global `current_filename` to {repr(self.filename)}")
            print(f"{TermColour.LIME}{TermColour.BOLD} Running{TermColour.WHITE} `{current_filename}`{TermColour.DEFAULT}")
            self.file_input(self.filename)
        else:
            log(f"ValueError: mode {repr(self.mode)} is not a valid mode.", level=LogLevel.CRITICAL)
            raise ValueError(f"mode {repr(self.mode)} is not a valid mode.")

        log("Driver.run_program(): Program terminating with a success state", level=LogLevel.INFO)

    def _process_arguments(self):
        """
        Will configure execution based on command line arguments.

        Very basic implementation right now, will improve later
        """
        log("Driver._process_arguments(): Processing command line arguments")
        if len(argv) == 1:
            # self.mode = "cmdline"
            # NOTE: cmdline disabled while testing to make execution quicker (Since I click run about 100 times/day (JOKE (satire)))
            # All of the following code should be removed in prod (not that I will ever reach that stage)
            dev_default_filname = config.getstr("dev.default_filename")
            if isfile(dev_default_filname):
                self.filename = dev_default_filname
                self.mode = "file"
            else:
                log(f"Driver._process_arguments(): Exception: file {repr(dev_default_filname)} does not exist!", level=LogLevel.CRITICAL)
                raise Exception(f"file {repr(dev_default_filname)} does not exist!")

        elif len(argv) == 2:
            path = argv[1]
            if isfile(path):
                self.filename = path
                self.mode = "file"
        else:
            log("Driver._process_arguments(): Exception: Unrecognised arguments:", *argv, level=LogLevel.CRITICAL)
            raise Exception("Unrecognised arguments!")

        log("Driver._process_arguments(): Driver.mode is now", repr(self.mode))

    def _process(self, code: str):
        log("Driver._process(): Initialising modules")
        parser = Parser(src=code)
        
        # All builtin symbols are inserted into the program at this stage
        builtin_symbol_table = SymbolTable(
            scope_name='<builtins>',
            scope_level=0
        )
        for key, value in builtin_method_mapping.items():
            builtin_symbol_table.define(
                BuiltinProcedureSymbol(key, value["callable"])
            )
        log("Driver._process(): Defined built-in symbols for", repr(builtin_symbol_table.scope_name), level=LogLevel.VERBOSE)
        
        symbol_table = SemanticAnalyser(
            src=code,
            builtin_scope=builtin_symbol_table
        )
        interpreter = Interpreter(code)

        log("Driver._process(): All modules initialised")
        log("Driver._process(): Evoking parser")
        tree = parser.parse()

        log(f"SYNTAX TREE ({repr(current_filename)})", level=LogLevel.HIGHLY_VERBOSE)
        log(str(tree), level=LogLevel.HIGHLY_VERBOSE, prefix_per_line=" |   ")
        log("SYNTAX TREE END", level=LogLevel.HIGHLY_VERBOSE)

        log("Driver._process(): Evoking semantic analyser")
        symbol_table.analyse(tree)
        log("Driver._process(): Evoking interpreter")
        interpreter.interpret(tree)

        log("Driver._process(): Processing complete!")

    def cmdline_input(self):
        """
        NOT IMPLEMENTED
        
        Run interpreter in command line interface mode
        """
        log("Driver: Attempted to execute program in command-line mode, which is not implemented")
        log("Driver: Terminating program")
        print("Command-line interface not implemented")
        exit()
        log("Driver.cmdline_input(): Executing program in command line mode")
        while 1:

            log("Driver.cmdline_input(): Set state to waiting for input")

            try:
                text = input(">>> ")
            except KeyboardInterrupt:
                # Silently exit
                log("Driver.cmdline_input(): Keyboard exit code (ctrl-c) fired, program terminating silently", level=LogLevel.INFO)
                return

            if not text:
                log("Driver.cmdline_input(): Line is empty, not processing")
                continue

            log("Driver.cmdline_input(): Processing line", repr(text))

            self._process(text)

    def file_input(self, filename: str):
        """
        Run interpreter in file mode
        """
        log("Driver.file_input(): Executing program in file input mode")
        file = open(filename, "r")
        text = file.read()
        log("Driver.file_input(): Successfully read file", repr(filename))

        if not text:
            log("Driver.file_input(): File is empty, not processing")
            return

        log("Driver.file_input(): Processing file")
        self._process(text)


###########################################
#                                         #
#   Main body                             #
#                                         #
########################################### 

def log(*message, level: int = LogLevel.DEBUG, stackoffset: int = 0, prefix_per_line: str = ""):
    """
    Logs a given message (or list of messages) to the default logger.

    Example usage:
    ```
    log("Hello\\nworld", level=10, stackoffset=0, prefix_per_line="LOGGER: ")
    ```
    
    Params:
    - `*message`: The message to be logged
    - `level` (optional): The logging level at which to write to the log
    - `stackoffset` (optional): The number at which to offset the stack. Used by the logger
    to obtain information like line number or the name of the current function
    - `prefix_per_line` (optional): What to prefix each line with, particularly useful
    for multi-line messages
    """
    if config.getbool("behaviour.logging_enabled"):
        message = " ".join(map(str, message))
        message = message.split("\n")
        for line in message:
            logging.log(msg=prefix_per_line+line, level=level, stacklevel=3+stackoffset)

if config.getbool("behaviour.logging_enabled"):
    # Hard coded because it is also hard coded in config.py
    logging.addLevelName(LogLevel.VERBOSE, "VERBOSE")
    logging.addLevelName(LogLevel.HIGHLY_VERBOSE, "HVERBOSE")
    logging.addLevelName(LogLevel.EAT_STACK, "EATSTACK")
    logging.addLevelName(LogLevel.ALL, "ALL")
    logging.basicConfig(
        filename="logs/" + config.getstr("logging.destination"),
        filemode='a',
        format=config.getstr("logging.format"),
        datefmt=config.getstr("logging.datefmt"),
        level=config.getint(f"logging.levels.{config.getstr('logging.level')}")
    )

if __name__ == '__main__':
    execution_start_time = current_time()

    if system_name == 'nt':
        system('color')
        
    log("Initialising", level=LogLevel.INFO)

    driver = Driver()
    driver.run_program()

    execution_finish_time = round(current_time()-execution_start_time, 5)
    log(f"Execution finished in {execution_finish_time}s", level=LogLevel.INFO)
    print(f"\n{TermColour.LIME}{TermColour.BOLD} Finished{TermColour.WHITE} execution in {execution_finish_time}s{TermColour.DEFAULT}")
