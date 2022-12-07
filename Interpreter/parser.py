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

class NodeVisitor(object):
    def visit(self, node):
        method_name = "visit_" + type(node).__name__
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        raise Exception(f"No visit_{type(node).__name__} method")


class AST(object):
    pass


class BinOp(AST):
    def __init__(self, left, op, right):
        self.left = left
        self.op: Token = op
        self.right = right


class Int(AST):
    def __init__(self, token: Token):
        self.token = token
        self.id = token.id


class Interpreter(NodeVisitor):
    def __init__(self, parser):
        self.parser = parser

    def interpret(self):
        tree = self.parser.parse()
        return self.visit(tree)

    def visit_BinOp(self, node: BinOp):
        if node.op.type == Type.PLUS:
            return self.visit(node.left) + self.visit(node.right)
        elif node.op.type == Type.MINUS:
            return self.visit(node.left) - self.visit(node.right)
        elif node.op.type == Type.MULT:
            return self.visit(node.left) * self.visit(node.right)
        elif node.op.type == Type.DIV:
            return self.visit(node.left) / self.visit(node.right)

    def visit_Int(self, node: Int):
        return node.id


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


class Parser():
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
            return Int(token)

        elif token.type == Type.LPAREN:
            self.eat(Type.LPAREN)
            node = self.expr()
            self.eat(Type.RPAREN)
            return node

    def term(self):
        """
        term -> factor ( (`MUL`|`DIV`) factor)*
        """
        
        node = self.factor()

        while self.current_token.type in (Type.MULT, Type.DIV):
            token = self.current_token

            if token.type == Type.MULT:
                self.eat(Type.MULT)

            elif token.type == Type.DIV:
                self.eat(Type.DIV)

            node = BinOp(left=node, op=token, right=self.factor())

        return node



    def expr(self):
        """
        expr   -> term ((`PLUS`|`MINUS`) term)*

        term   -> factor ((`MUL`|`DIV`) factor)*
        
        factor -> `INTEGER` | `LPAREN` expr `RPAREN`
        """

        node = self.term()

        while self.current_token.type in (Type.PLUS, Type.MINUS):
            token = self.current_token

            if token.type == Type.PLUS:
                self.eat(Type.PLUS)

            elif token.type == Type.MINUS:
                self.eat(Type.MINUS)

            node = BinOp(left=node, op=token, right=self.term())

        return node

    def parse(self):
        """Main Parser method"""
        return self.expr()

if __name__ == '__main__':
    environment = []
    while 1:
        text = input(">>> ")
        if not text:
            continue
        try:
            environment.append(text)
            parser = Parser(text)
            interpreter = Interpreter(parser)
            result = interpreter.interpret()
            print(result)
        except Exception as e:
            print(e)
        