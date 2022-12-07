from enum import Enum

class Type(Enum):
    INTEGER = object()
    STR = object()
    MULT = object()
    DIV = object()
    PLUS = object()
    MINUS = object()
    LPAREN = object()
    RPAREN = object()
    IDENTIFIER = object()
    VARIABLE = object()
    BOOL = object()
    ASSIGN = object()
    COMMENT = object()
    SEMI = object()
    EOF = object()
    NEWLINE = object()
    

class Token():
    def __init__(self, startPos, type, id) -> None:
        self.startPos: int = startPos
        self.type: Type = type
        self.id: str = id

    def __str__(self) -> str:
        return f"Token[type = {self.type}, id = '{self.id}', startPos = {self.startPos}]"


class Lexer():
    def __init__(self, text):
        self.text: str = text
        self.pos: int = 0
        self.lineno: int = 1
        self.current_char: str = self.text[self.pos]

    def error(self):
        raise Exception(f'Error parsing input on line {self.lineno}, pos {self.pos}\n   {text}\n   {" "*self.pos}^ Here')

    def advance(self):
        """Advance `self.pos` and set `self.current_char`"""
        self.pos += 1
        if self.pos > len(self.text) - 1:
            self.current_char = None
        else:
            self.current_char = self.text[self.pos]

    def skip_whitespace(self):
        while self.current_char is not None and self.current_char == " ":
            self.advance()

    def integer(self):
        """Consume an integer from the input code and return it"""
        integer = ''
        while self.current_char is not None and self.current_char.isdigit():
            integer += self.current_char
            self.advance()
        return int(integer)

    def operator(self):
        """Consume an integer from the input code and return it"""
        base_operators = ['+', '-', '/', '*']
        operator = ''

        if self.current_char in base_operators:
            operator += self.current_char
            self.advance()

        return operator

    def get_next_token(self):
        """Lexer

        Responsible for breaking down and extracting 
        tokens out of code.
        """
        while self.current_char is not None:

            if self.current_char == " ":
                self.skip_whitespace()

            elif self.current_char.isdigit():
                return Token(self.pos, Type.INTEGER, self.integer())

            elif self.current_char == '*':
                token = Token(self.pos, Type.MULT, self.current_char)
                self.advance()
                return token

            elif self.current_char == '/':
                token = Token(self.pos, Type.DIV, self.current_char)
                self.advance()
                return token

            elif self.current_char == '+':
                token = Token(self.pos, Type.PLUS, self.current_char)
                self.advance()
                return token

            elif self.current_char == '-':
                token = Token(self.pos, Type.MINUS, self.current_char)
                self.advance()
                return token

            elif self.current_char == '(':
                token = Token(self.pos, Type.LPAREN, self.current_char)
                self.advance()
                return token

            elif self.current_char == ')':
                token = Token(self.pos, Type.RPAREN, self.current_char)
                self.advance()
                return token
            
            else:
                self.error()

        return Token(self.pos, Type.EOF, None)


class Interpreter():
    def __init__(self, text) -> None:
        self.text: str = text
        if not self.text:
            return None
        self.lexer: Lexer = Lexer(self.text)
        self.current_token: Token = self.lexer.get_next_token()

    def error(self):
        raise Exception(f"Error parsing input on line {self.lexer.lineno}, pos {self.lexer.pos}\n   {text}\n   {' '*(self.lexer.pos-1)}^ Here\nUnexpected type <'{self.current_token.type.name}'>")

    def eat(self, expected_type: Type):
        # Compares the current token type to the expected
        # type and if they're equal then 'eat' the current
        # token and move onto the next token.
        print(self.current_token, expected_type)
        if self.current_token.type == expected_type:
            self.current_token = self.lexer.get_next_token()
        else:
            self.error()

    def factor(self):
        """Return an `INTEGER` token value
        
        factor -> `INTEGER` | `LPAREN` expr `RPAREN`
        """
        token = self.current_token

        if token.type == Type.INTEGER:
            self.eat(Type.INTEGER)
            return token.id

        elif token.type == Type.LPAREN:
            self.eat(Type.LPAREN)
            result = self.expr()
            self.eat(Type.RPAREN)
            return result

    def term(self):
        """
        term -> factor ( (`MUL`|`DIV`) factor)*
        """
        
        result = self.factor()

        while self.current_token.type in (Type.MULT, Type.DIV):
            token = self.current_token

            if token.type == Type.MULT:
                self.eat(Type.MULT)
                result = result * self.factor()

            elif token.type == Type.DIV:
                self.eat(Type.DIV)
                result = result / self.factor()

        return result



    def expr(self):
        """Parser method

        expr   -> term ((`PLUS`|`MINUS`) term)*

        term   -> factor ((`MUL`|`DIV`) factor)*
        
        factor -> `INTEGER` | `LPAREN` expr `RPAREN`
        """

        #self.current_token = self.lexer.get_next_token()

        result = self.term()

        while self.current_token.type in (Type.PLUS, Type.MINUS):
            token = self.current_token

            if token.type == Type.PLUS:
                self.eat(Type.PLUS)
                result = result + self.term()

            elif token.type == Type.MINUS:
                self.eat(Type.MINUS)
                result = result - self.term()

        return result

if __name__ == '__main__':
    environment = []
    while 1:
        text = input(">>> ")
        if not text:
            continue
        try:
            environment.append(text)
            interpreter = Interpreter(text)
            result = interpreter.expr()
            print(result)
        except Exception as e:
            print(e)
        