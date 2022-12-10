from inspect import currentframe, getframeinfo
from enum import Enum

PRINT_TOKENS = True
PRINT_EAT_STACK = False
PRINT_TREE = True

# Modes:
# "cmdline"
# "file"

MODE = "file"

typeof = type


class type(Enum):
    """
    Enum class to hold all the data types for SAP language
    (Not all types listed here have been implemented)
    """
    INTEGER = object()
    STR = object()
    MULT = object()
    DIV = object()
    PLUS = object()
    MINUS = object()
    LPAREN = object()
    RPAREN = object()
    IDENTIFIER = object()
    BOOL = object()
    ASSIGN = object()
    COMMENT = object()
    SEMI = object()
    BEGIN = object()
    END = object()
    SOF = object()
    EOF = object()
    NEWLINE = object()
    

class Token():
    """
    Token data class

    Simple data class to hold information about a token
    """
    def __init__(self, startPos: int, type: type, id: str | int) -> None:
        self.startPos: int = startPos
        self.type: type = type
        self.id: str | int = id

    def __str__(self) -> str:
        return f"Token[type = {self.type}, id = '{self.id}', startPos = {self.startPos}]"

    def __repr__(self) -> str:
        return repr(self.__str__())


class Node(object):
    """
    Represents a node on an abstract syntax tree
    """
    def __str__(self):
        return str(self._print_children(self.__dict__))

    def _print_children(self, tree_dict: dict, depth: int = 1):
        text = ""
        if depth == 1:
            text += self.__class__.__name__ + "(\n"
        for key, value in tree_dict.items():
            if isinstance(value, Node):
                text += "   " * depth + str(key) + ": " + str(value.__class__.__name__) + "(\n"
                text += self._print_children(value.__dict__, depth+1)
                text += "   " * depth + "),\n"
            elif isinstance(value, list):
                text += "   " * depth + str(key) + ": [\n"
                for node in value:
                    text += "   " * (depth+1) + node.__class__.__name__ + "(\n"
                    text += self._print_children(node.__dict__, depth+2)
                    text += "   " * (depth+1) + "),\n"

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

class Compound(Node):
    def __init__(self):
        self.children: list[Node] = []


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

"""
LEAF NODES
(in order of precedence)
"""

class Var(Node):
    def __init__(self, token):
        self.token: Token = token
        self.id = self.token.id

class Int(Node):
    def __init__(self, token):
        self.token: Token = token
        self.id: int = self.token.id

class NoOp(Node):
    pass


class Interpreter():
    """
    Main interpreter class

    The interpreter is resposible for processing abstract syntax trees and compiling (not machine code) them into a final result.
    It works by 'visiting' each node in the tree and processing it based on it's attributes and surrounding nodes.
    """
    
    global_scope = {}

    def __init__(self, text):
        self.parser: Parser = Parser(text)

    def interpret(self):
        tree = self.parser.parse()
        if PRINT_TREE:
            print(tree)
        return self.visit(tree)

    def visit(self, node):
        method_name = "visit_" + typeof(node).__name__
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        raise Exception(f"No visit_{typeof(node).__name__} method")

    def visit_Compound(self, node: Compound):
        for child in node.children:
            self.visit(child)

    def visit_AssignOp(self, node: AssignOp):
        variable_id = node.left.id
        self.global_scope[variable_id] = self.visit(node.right)

    def visit_UnaryOp(self, node: UnaryOp):
        if node.op.type == type.PLUS:
            return +self.visit(node.expr)
        elif node.op.type == type.MINUS:
            return -self.visit(node.expr)

    def visit_BinOp(self, node: BinOp):
        if node.op.type == type.PLUS:
            return self.visit(node.left) + self.visit(node.right)
        elif node.op.type == type.MINUS:
            return self.visit(node.left) - self.visit(node.right)
        elif node.op.type == type.MULT:
            return self.visit(node.left) * self.visit(node.right)
        elif node.op.type == type.DIV:
            return self.visit(node.left) // self.visit(node.right)

    def visit_Var(self, node: Var):
        varaible_id = node.id
        val = self.global_scope.get(varaible_id)
        if val is None:
            raise NameError(repr(varaible_id))
        else:
            return val

    def visit_Int(self, node: Int):
        return node.id

    def visit_NoOp(self, node: NoOp):
        pass


class Lexer():
    """
    Main lexer class

    The lexer is resposible for the tokenization of the code. (Side note: I think that is the british spelling)
    In other words, it splits the code up into it's individual components.

    For example given the code:
    `2 + 2`

    The lexer will generate:
    ```
    Token[type = type.INTEGER, id = '2', startPos = 0]
    Token[type = type.PLUS, id = '+', startPos = 1]
    Token[type = type.INTEGER, id = '2', startPos = 2]
    ```
    """
    def __init__(self, text):
        self.text: str = text
        self.pos: int = 0
        self.lineno: int = 1
        self.linepos: int = 0
        self.current_char: str = self.text[self.pos]

    def error(self):
        raise Exception(f'Error parsing input on line {self.lineno}, pos {self.linepos}\n')

    def advance(self):
        """Advance `self.pos` and set `self.current_char`"""
        self.pos += 1
        self.linepos += 1
        if self.pos > len(self.text) - 1:
            self.current_char = None
        else:
            self.current_char = self.text[self.pos]

    def peek(self):
        """Peeks at the next character in the code and returns it"""
        peek_pos = self.pos + 1
        if peek_pos > len(self.text) - 1:
            return None
        else:
            return self.text[peek_pos]

    def _identifier(self):
        result = ""
        start_position = self.pos
        while self.current_char is not None and (self.current_char.isalnum() or self.current_char == "_"):
            result += self.current_char
            self.advance()

        return Token(start_position, type.IDENTIFIER, result)

    def skip_whitespace(self):
        "Advances `self.pos` until a non-whitespace character has been reached"
        while self.current_char is not None and self.current_char == " ":
            self.advance()

    def integer(self):
        """Consumes an integer from the input code and returns it"""
        integer = ''
        while self.current_char is not None and self.current_char.isdigit():
            integer += self.current_char
            self.advance()
        return int(integer)

    def operator(self):
        """Consumes an operator from the input code and returns it"""

        raise NotImplemented("operator() function not implemented")

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
                continue

            elif self.current_char == "\n":
                self.lineno += 1
                self.linepos = 0
                self.advance()
                continue

            if self.current_char.isalpha() or self.current_char == "_":
                return self._identifier()

            elif self.current_char.isdigit():
                return Token(self.pos, type.INTEGER, self.integer())

            elif self.current_char == "=":
                token = Token(self.pos, type.ASSIGN, self.current_char)
                self.advance()
                return token

            elif self.current_char == "{":
                token = Token(self.pos, type.BEGIN, self.current_char)
                self.advance()
                return token

            elif self.current_char == "}":
                token = Token(self.pos, type.END, self.current_char)
                self.advance()
                return token

            elif self.current_char == ";":
                token = Token(self.pos, type.SEMI, self.current_char)
                self.advance()
                return token

            elif self.current_char == '*':
                token = Token(self.pos, type.MULT, self.current_char)
                self.advance()
                return token

            elif self.current_char == '/':
                token = Token(self.pos, type.DIV, self.current_char)
                self.advance()
                return token

            elif self.current_char == '+':
                token = Token(self.pos, type.PLUS, self.current_char)
                self.advance()
                return token

            elif self.current_char == '-':
                token = Token(self.pos, type.MINUS, self.current_char)
                self.advance()
                return token

            elif self.current_char == '(':
                token = Token(self.pos, type.LPAREN, self.current_char)
                self.advance()
                return token

            elif self.current_char == ')':
                token = Token(self.pos, type.RPAREN, self.current_char)
                self.advance()
                return token
            
            self.error()

        return Token(self.pos, type.EOF, None)


class Parser():
    """
    Main parser class

    The class is responsible for parsing the tokens and turning them into syntax trees.
    These trees make it easier to process the code and understand the relationships between tokens.
    """
    def __init__(self, text) -> None:
        self.text: str = text
        if not self.text:
            return None
        self.lexer: Lexer = Lexer(self.text)
        self.current_token: Token = self.lexer.get_next_token()

    def error(self):
        raise Exception(f"Error parsing input on line {self.lexer.lineno}, pos {self.lexer.linepos}\nUnexpected type <'{self.current_token.type.name}'>")

    def eat(self, expected_type: type):
        # Compares the current token type to the expected
        # type and if they're equal then 'eat' the current
        # token and move onto the next token.
        if PRINT_TOKENS:
            print(self.current_token, expected_type)
        if self.current_token.type == expected_type:
            self.current_token = self.lexer.get_next_token()
        else:
            self.error()

    def factor(self):
        """Return an `INTEGER` token value
        
        factor -> `PLUS` factor
                | `MINUS` factor
                | `INTEGER` 
                | `LPAREN` expr `RPAREN`
                | variable

        """
        token = self.current_token

        # `PLUS` factor
        if token.type == type.PLUS:
            if PRINT_EAT_STACK: print("Calling eat() from line", getframeinfo(currentframe()).lineno)
            self.eat(type.PLUS)
            node = UnaryOp(token, self.factor())
            return node
        
        # `MINUS` factor
        elif token.type == type.MINUS:
            if PRINT_EAT_STACK: print("Calling eat() from line", getframeinfo(currentframe()).lineno)
            self.eat(type.MINUS)
            node = UnaryOp(token, self.factor())
            return node

        # `INTEGER`
        elif token.type == type.INTEGER:
            if PRINT_EAT_STACK: print("Calling eat() from line", getframeinfo(currentframe()).lineno)
            self.eat(type.INTEGER)
            return Int(token)

        # `LPAREN` expr `RPAREN`
        elif token.type == type.LPAREN:
            if PRINT_EAT_STACK: print("Calling eat() from line", getframeinfo(currentframe()).lineno)
            self.eat(type.LPAREN)
            node = self.expr()
            if PRINT_EAT_STACK: print("Calling eat() from line", getframeinfo(currentframe()).lineno)
            self.eat(type.RPAREN)
            return node

        # variable
        else:
            node = self.variable()
            return node

    def term(self):
        """
        term -> factor ( (`MUL`|`DIV`) factor)*
        """
        
        node = self.factor()

        # factor ( (`MUL`|`DIV`) factor)*
        while self.current_token.type in (type.MULT, type.DIV):
            token = self.current_token

            if token.type == type.MULT:
                if PRINT_EAT_STACK: print("Calling eat() from line", getframeinfo(currentframe()).lineno)
                self.eat(type.MULT)

            elif token.type == type.DIV:
                if PRINT_EAT_STACK: print("Calling eat() from line", getframeinfo(currentframe()).lineno)
                self.eat(type.DIV)

            node = BinOp(left=node, op=token, right=self.factor())

        return node

    def expr(self):
        """
        expr -> term ((`PLUS`|`MINUS`) term)*
        """

        node = self.term()

        # term ((`PLUS`|`MINUS`) term)*
        while self.current_token.type in (type.PLUS, type.MINUS):
            token = self.current_token

            if token.type == type.PLUS:
                if PRINT_EAT_STACK: print("Calling eat() from line", getframeinfo(currentframe()).lineno)
                self.eat(type.PLUS)

            elif token.type == type.MINUS:
                if PRINT_EAT_STACK: print("Calling eat() from line", getframeinfo(currentframe()).lineno)
                self.eat(type.MINUS)

            node = BinOp(left=node, op=token, right=self.term())

        return node

    def compound_statement(self):
        """
        compound_statement -> `BEGIN` statement_list `END`
        """

        if PRINT_EAT_STACK: print("Calling eat() from line", getframeinfo(currentframe()).lineno)
        self.eat(type.BEGIN)
        nodes = self.statement_list()
        if PRINT_EAT_STACK: print("Calling eat() from line", getframeinfo(currentframe()).lineno)
        self.eat(type.END)

        root = Compound()
        for node in nodes:
            root.children.append(node)

        return root

    def statement_list(self):
        """
        statement_list -> statement | statement `SEMI` statement_list
        """

        node = self.statement()

        results = [node]

        while self.current_token.type == type.SEMI:
            if PRINT_EAT_STACK: print("Calling eat() from line", getframeinfo(currentframe()).lineno)
            self.eat(type.SEMI)
            results.append(self.statement())

        if self.current_token.type == type.IDENTIFIER:
            self.error()

        return results

    def statement(self):
        """
        statement -> compound_statement
                   | assignment_statement
                   | empty
        """

        if self.current_token.type == type.BEGIN:
            node = self.compound_statement()
        elif self.current_token.type == type.IDENTIFIER:
            node = self.assignment_statement()
        else:
            node = self.empty()

        return node

    def assignment_statement(self):
        """
        assignment_statement -> variable ASSIGN expr
        """

        left = self.variable()
        token = self.current_token
        if PRINT_EAT_STACK: print("Calling eat() from line", getframeinfo(currentframe()).lineno)
        self.eat(type.ASSIGN)
        right = self.expr()
        node = AssignOp(left, token, right)

        return node

    def variable(self):
        """
        variable -> `IDENTIFIER`
        """
        node = Var(self.current_token)
        if PRINT_EAT_STACK: print("Calling eat() from line", getframeinfo(currentframe()).lineno)
        self.eat(type.IDENTIFIER)
        return node

    def empty(self):
        return NoOp()

    def program(self):
        """
        program -> statement_list | compound_statement <`EOF`>
        """

        if self.current_token.type == type.BEGIN:
            node = self.compound_statement()
        else:
            # Some hackery to get the code to run without surrounding brackets
            node = Compound()
            node.children = self.statement_list()
        
        if PRINT_EAT_STACK: print("Calling eat() from line", getframeinfo(currentframe()).lineno)
        self.eat(type.EOF)


        #node = self.compound_statement()
        #if PRINT_EAT_STACK: print("Calling eat() from line", getframeinfo(currentframe()).lineno)
        #self.eat(type.EOF)

        return node

    def parse(self) -> Node:
        """Main Parser method

        Here is our program grammar:

        program -> <`SOF`> (statement_list | compound_statement) <`EOF`>

        compound_statement -> `BEGIN` statement_list `END`

        statement_list -> statement | statement `SEMI` statement_list

        statement -> compound_statement
                   | assignment_statement
                   | empty

        assignment_statement -> variable ASSIGN expr

        empty ->
        // What did you expect cuh

        expr -> term ((`PLUS`|`MINUS`) term)*

        term -> factor ((`MUL`|`DIV`) factor)*

        factor -> `PLUS` factor
                | `MINUS` factor
                | `INTEGER` 
                | `LPAREN` expr `RPAREN`
                | variable

        variable -> `IDENTIFIER`
        """
        node = self.program()
        if self.current_token.type != type.EOF:
            self.error()
        
        return node

def cmdline_input():
    while 1:
        text = input(">>> ")
        if not text:
            continue

        interpreter = Interpreter(text)
        result = interpreter.interpret()

        print("Globals:")
        print(interpreter.global_scope)
        print()
        print("Result: ")
        print(result)

def file_input():

    file = open("./hello.sap", "r")
    text = file.read()

    if not text:
        return

    interpreter = Interpreter(text)
    result = interpreter.interpret()

    print("Globals:")
    print(interpreter.global_scope)
    print()
    print("Result: ")
    print(result)
    

def main():
    if MODE == "cmdline":
        cmdline_input()
    elif MODE == "file":
        file_input()
        

# driver code
if __name__ == '__main__':
    main()
        