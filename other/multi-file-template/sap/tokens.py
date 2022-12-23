from typing import Any
from enum import Enum


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