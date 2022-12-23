"""
All error types and error codes
"""

from enum import Enum

from sap.tokens import Token

from sap import _DEV_RAISE_ERROR_STACK, current_filename


class ErrorCode(Enum):
    SYNTAX_ERROR        = "SyntaxError"
    NAME_ERROR          = "NameError"
    TYPE_ERROR          = "TypeError"


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
                (f"   {' '*len(str(self.lineno+2))}  {' '*(self.linecol)} ^\n"),
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
