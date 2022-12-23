from sap.errors import LexerError, ErrorCode
from sap.tokens import Token, TokenType


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
