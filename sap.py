from __future__ import annotations

from enum import Enum
from inspect import currentframe, getframeinfo

# Constants

# Modes:
# "cmdline"
# "file"

MODE = "file"

PRINT_TOKENS = True
PRINT_EAT_STACK = False
PRINT_TREE = True

# Since the `type()` function is overwritten,
# this code allows us to still use the original `type()` function by calling `typeof()`
typeof = type


###########################################
#                                         #
#   Data classes                          #
#                                         #
###########################################

class type(Enum):
    """
    Enum class to hold all the data types for SAP language
    """
    INTEGER = object()
    REAL = object()
    INTEGER_CONST = object()
    REAL_CONST = object()

    MULT = object()
    INTEGER_DIV = object()
    FLOAT_DIV = object
    PLUS = object()
    MINUS = object()

    LPAREN = object()
    RPAREN = object()
    IDENTIFIER = object()
    ASSIGN = object()
    COMMENT = object()

    SEMI = object()
    COLON = object()

    BEGIN = object()
    END = object()

    EOF = object()


class global_scope(dict):
    def __init_subclass__(cls):
        return super().__init_subclass__()
    
    def __str__(self):
        text = []

        for key, value in sorted(self.items(), key=lambda x: x[1][0], reverse=True):
            text.append("  <" + str(value[0]) + "> " + str(key) +  " = " + str(value[1]))
        
        return "\n".join(text)


class Token:
    """
    Token data class

    Simple data class to hold information about a token
    """
    def __init__(self, start_pos: int, datatype: type, id: str | int | None):
        self.start_pos: int = start_pos
        self.type: type = datatype
        self.id: str | int | None = id

    def __str__(self) -> str:
        return f"Token[type = {self.type}, id = '{self.id}', start_pos = {self.start_pos}]"

    def __repr__(self) -> str:
        return repr(self.__str__())


###########################################
#                                         #
#   Lexer code                            #
#                                         #
###########################################

class Lexer:
    """
    Main lexer class

    The lexer is responsible for the tokenization of the code. (Side note: I think that is the british spelling)
    In other words, it splits the code up into its individual components.

    For example given the code:
    `2 + 2`

    The lexer will generate:
    ```
    Token[type = type.INTEGER_CONST, id = '2', startPos = 0]
    Token[type = type.PLUS, id = '+', startPos = 1]
    Token[type = type.INTEGER_CONST, id = '2', startPos = 2]
    ```
    """
    def __init__(self, text):
        self.text: str = text
        self.pos: int = 0
        self.lineno: int = 1
        self.linepos: int = 0
        self.current_char: str = self.text[self.pos]

        self.RESERVED_KEYWORDS: dict = {
            'int': type.INTEGER,
            'real': type.REAL
        }

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

    def peek(self) -> None | str:
        """Peeks at the next character in the code and returns it"""
        peek_pos = self.pos + 1
        if peek_pos > len(self.text) - 1:
            return None
        else:
            return self.text[peek_pos]

    def _identifier(self) -> Token:
        """Creates and returns an identifier token"""
        result = ""
        start_position = self.pos
        while self.current_char is not None and (self.current_char.isalnum() or self.current_char == "_"):
            result += self.current_char
            self.advance()

        # Checks if `result` is a keyword or not and returns the appropiate type.
        # Gets the type associated with `result if applicable, else default to `type.IDENTIFIER`
        token_type = self.RESERVED_KEYWORDS.get(result, type.IDENTIFIER)

        token = Token(start_position, token_type, result)

        return token

    def skip_whitespace(self):
        """Advances `self.pos` until a non-whitespace character has been reached"""
        while self.current_char is not None and self.current_char == " ":
            self.advance()

    def skip_comment(self):
        """Advances `self.pos` until a comment terminator (*/) has been reached"""
        while not (self.current_char == "*" and self.peek() == "/"):
            self.advance()
        # Advance twice more to skip over the final "*/"
        self.advance()
        self.advance()

    def number(self) -> int:
        """Consumes a number from the input code and returns it"""
        start_pos = self.pos
        number = ''

        while self.current_char is not None and self.current_char.isdigit():
            number += self.current_char
            self.advance()
        
        if self.current_char == ".":
            number += self.current_char
            self.advance()

            while self.current_char is not None and self.current_char.isdigit():
                number += self.current_char
                self.advance()

            token = Token(start_pos, type.REAL_CONST, float(number))

        else:
            token = Token(start_pos, type.INTEGER_CONST, int(number))

        return token

    def get_next_token(self) -> Token:
        """
        Responsible for breaking down and extracting 
        tokens out of code.
        """
        while self.current_char is not None:

            # Ignored characters

            if self.current_char == "\n":
                self.lineno += 1
                self.linepos = 0
                self.advance()
                continue

            elif self.current_char == " ":
                self.skip_whitespace()
                continue

            elif self.current_char == "/" and self.peek() == "*":
                self.skip_comment()
                continue

            # Terminals

            if self.current_char.isalpha() or self.current_char == "_":
                return self._identifier()

            elif self.current_char.isdigit():
                return self.number()

            # Operators

            elif self.current_char == "=":
                token = Token(self.pos, type.ASSIGN, self.current_char)
                self.advance()
                return token

            elif self.current_char == '*':
                token = Token(self.pos, type.MULT, self.current_char)
                self.advance()
                return token

            elif self.current_char == '/':

                if self.peek() != "/":
                    token = Token(self.pos, type.FLOAT_DIV, self.current_char)
                else:
                    token = Token(self.pos, type.INTEGER_DIV, self.current_char)
                    self.advance()

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

            # Symbols

            elif self.current_char == ";":
                token = Token(self.pos, type.SEMI, self.current_char)
                self.advance()
                return token

            elif self.current_char == ":":
                token = Token(self.pos, type.COLON, self.current_char)
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

            elif self.current_char == "{":
                token = Token(self.pos, type.BEGIN, self.current_char)
                self.advance()
                return token

            elif self.current_char == "}":
                token = Token(self.pos, type.END, self.current_char)
                self.advance()
                return token

            self.error()

        return Token(self.pos, type.EOF, None)


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
    """

    def __init__(self, text):
        self.text: str = text
        self.lexer: Lexer = Lexer(self.text)
        self.current_token: Token = self.lexer.get_next_token()

    def error(self):
        raise Exception(f"Error parsing input on line {self.lexer.lineno}, pos {self.lexer.linepos}\n"
                        f"Unexpected type <'{self.current_token.type.name}'>")

    def eat(self, expected_type: type):
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
            self.error()

    # Grammar definitions

    def program(self) -> Compound:
        """
        program -> (statement_list | compound_statement) <`EOF`>
        """
        if self.current_token.type == type.BEGIN:
            node = self.compound_statement()
        else:
            # Some hackery to get the code to run without surrounding brackets
            node = Compound()
            node.children = self.statement_list()

        if PRINT_EAT_STACK:
            print("Calling eat() from line", getframeinfo(currentframe()).lineno)
        self.eat(type.EOF)

        return node

    def type_spec(self) -> TypeNode:
        """
        type_spec -> `INTEGER` | `REAL`
        """
        token = self.current_token
        if token.type == type.INTEGER:
            if PRINT_EAT_STACK:
                print("Calling eat() from line", getframeinfo(currentframe()).lineno)
            self.eat(type.INTEGER)
        else:
            if PRINT_EAT_STACK:
                print("Calling eat() from line", getframeinfo(currentframe()).lineno)
            self.eat(type.REAL)
        node = TypeNode(token)
        return node


    def compound_statement(self) -> Compound:
        """
        compound_statement -> `BEGIN` statement_list `END`
        """
        if PRINT_EAT_STACK:
            print("Calling eat() from line", getframeinfo(currentframe()).lineno)
        self.eat(type.BEGIN)
        nodes = self.statement_list()
        if PRINT_EAT_STACK:
            print("Calling eat() from line", getframeinfo(currentframe()).lineno)
        self.eat(type.END)

        root = Compound()
        for node in nodes:
            root.children.append(node)

        return root

    def statement_list(self) -> list[Node]:
        """
        statement_list -> statement | statement `SEMI` statement_list
        """
        node = self.statement()

        results = [node]

        while self.current_token.type == type.SEMI:
            if PRINT_EAT_STACK:
                print("Calling eat() from line", getframeinfo(currentframe()).lineno)
            self.eat(type.SEMI)
            results.append(self.statement())

        if self.current_token.type == type.IDENTIFIER:
            self.error()

        return results
        
    def statement(self) -> Node:
        """
        statement -> compound_statement
                   | variable_change
                   | empty
        """
        if self.current_token.type == type.BEGIN:
            node = self.compound_statement()
        elif self.current_token.type == type.IDENTIFIER:
            node = self.variable_change()
        else:
            node = self.empty()

        return node

    def variable_change(self) -> VarDecl | AssignOp:
        """
        variable_change -> variable (`COLON` | `ASSIGN`)
        """
        var_node = self.variable()
        op_node = self.current_token
        if op_node.type == type.COLON:
            node = self.declaration_statement(var_node)
        else:
            node = self.assignment_statement(var_node)

        return node

    def declaration_statement(self, var_node):
        """
        declaration_statement -> variable `COLON` type_spec `ASSIGN` expr | variable `COLON` type_spec
        """
        colon_op = self.current_token
        if PRINT_EAT_STACK:
            print("Calling eat() from line", getframeinfo(currentframe()).lineno)
        self.eat(type.COLON)

        type_node = self.type_spec()
        
        # variable `COLON` type_spec `ASSIGN` expr
        if self.current_token.type == type.ASSIGN:
            assign_op = self.current_token
            if PRINT_EAT_STACK:
                print("Calling eat() from line", getframeinfo(currentframe()).lineno)
            self.eat(type.ASSIGN)
            expr_node = self.expr()

            node = VarDecl(var_node, colon_op, type_node, assign_op, expr_node)
        
        # variable `COLON` type_spec
        else:
            node = VarDecl(var_node, colon_op, type_node)

        return node

    def assignment_statement(self, var_node) -> AssignOp:
        """
        assignment_statement -> variable `ASSIGN` expr
        """
        token = self.current_token
        if PRINT_EAT_STACK:
            print("Calling eat() from line", getframeinfo(currentframe()).lineno)
        self.eat(type.ASSIGN)
        right = self.expr()
        node = AssignOp(var_node, token, right)

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
        while self.current_token.type in (type.PLUS, type.MINUS):
            token = self.current_token

            if token.type == type.PLUS:
                if PRINT_EAT_STACK:
                    print("Calling eat() from line", getframeinfo(currentframe()).lineno)
                self.eat(type.PLUS)

            elif token.type == type.MINUS:
                if PRINT_EAT_STACK:
                    print("Calling eat() from line", getframeinfo(currentframe()).lineno)
                self.eat(type.MINUS)

            node = BinOp(left=node, op=token, right=self.term())

        return node

    def term(self) -> Node:
        """
        term -> factor ((`MUL`|`INTEGER_DIV`|`FLOAT_DIV`) factor)*
        """
        node = self.factor()

        # factor ( (`MUL`|`DIV`) factor)*
        while self.current_token.type in (type.MULT, type.INTEGER_DIV, type.FLOAT_DIV):
            token = self.current_token

            if token.type == type.MULT:
                if PRINT_EAT_STACK:
                    print("Calling eat() from line", getframeinfo(currentframe()).lineno)
                self.eat(type.MULT)

            elif token.type == type.INTEGER_DIV:
                if PRINT_EAT_STACK:
                    print("Calling eat() from line", getframeinfo(currentframe()).lineno)
                self.eat(type.INTEGER_DIV)
            
            elif token.type == type.FLOAT_DIV:
                if PRINT_EAT_STACK:
                    print("Calling eat() from line", getframeinfo(currentframe()).lineno)
                self.eat(type.FLOAT_DIV)

            node = BinOp(left=node, op=token, right=self.factor())

        return node

    def factor(self) -> Node:
        """
        factor -> `PLUS` factor
                | `MINUS` factor
                | `INTEGER_CONST`
                | `REAL_CONST` 
                | `LPAREN` expr `RPAREN`
                | variable
        """
        token = self.current_token

        # `PLUS` factor
        if token.type == type.PLUS:
            if PRINT_EAT_STACK:
                print("Calling eat() from line", getframeinfo(currentframe()).lineno)
            self.eat(type.PLUS)
            node = UnaryOp(token, self.factor())
            return node

        # `MINUS` factor
        elif token.type == type.MINUS:
            if PRINT_EAT_STACK:
                print("Calling eat() from line", getframeinfo(currentframe()).lineno)
            self.eat(type.MINUS)
            node = UnaryOp(token, self.factor())
            return node

        # `INTEGER_CONST`
        elif token.type == type.INTEGER_CONST:
            if PRINT_EAT_STACK:
                print("Calling eat() from line", getframeinfo(currentframe()).lineno)
            self.eat(type.INTEGER_CONST)
            return Num(token)

        # `REAL_CONST`
        elif token.type == type.REAL_CONST:
            if PRINT_EAT_STACK:
                print("Calling eat() from line", getframeinfo(currentframe()).lineno)
            self.eat(type.REAL_CONST)
            return Num(token)

        # `LPAREN` expr `RPAREN`
        elif token.type == type.LPAREN:
            if PRINT_EAT_STACK:
                print("Calling eat() from line", getframeinfo(currentframe()).lineno)
            self.eat(type.LPAREN)
            node = self.expr()
            if PRINT_EAT_STACK:
                print("Calling eat() from line", getframeinfo(currentframe()).lineno)
            self.eat(type.RPAREN)
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
            print("Calling eat() from line", getframeinfo(currentframe()).lineno)
        self.eat(type.IDENTIFIER)
        return node

    def parse(self) -> Node:
        """Main Parser method

        Here is our program grammar:

        program -> (statement_list | compound_statement) <`EOF`>

        type_spec -> `INTEGER` | `REAL`

        compound_statement -> `BEGIN` statement_list `END`

        statement_list -> statement
                        | statement `SEMI` statement_list

        statement -> compound_statement
                   | variable_change
                   | empty

        variable_change -> variable (`COLON` | `ASSIGN`)

        declaration_statement -> variable `COLON` type_spec `ASSIGN` expr
                               | variable `COLON` type_spec

        assignment_statement -> variable `ASSIGN` expr

        empty ->
        // What did you expect cuh

        expr -> term ((`PLUS`|`MINUS`) term)*

        term -> factor ((`MUL`|`INTEGER_DIV`|`FLOAT_DIV`) factor)*

        factor -> `PLUS` factor
                | `MINUS` factor
                | `INTEGER_CONST`
                | `REAL_CONST` 
                | `LPAREN` expr `RPAREN`
                | variable

        variable -> `IDENTIFIER`
        """
        node = self.program()
        if self.current_token.type != type.EOF:
            self.error()

        return node


###########################################
#                                         #
#   Interpreter code                      #
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
        """
        text = ""

        if depth == 1:
            text += self.__class__.__name__ + "(\n"

        for key, value in tree_dict.items():

            if isinstance(value, Node):
                text += "   " * depth + str(key) + ": " + str(value.__class__.__name__) + "(\n"
                text += self._print_children(value.__dict__, depth + 1)
                text += "   " * depth + "),\n"

            elif isinstance(value, list):
                text += "   " * depth + str(key) + ": [\n"
                for node in value:
                    text += "   " * (depth + 1) + node.__class__.__name__ + "(\n"
                    text += self._print_children(node.__dict__, depth + 2)
                    text += "   " * (depth + 1) + "),\n"
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


class VarDecl(Node):
    def __init__(self, var_node, colon_op, type_node, assign_op=None, expr_node=None):
        self.var_node: Var = var_node
        self.colon_op: Token = colon_op
        self.type_node: TypeNode = type_node
        self.assign_op: Token = assign_op
        self.expr_node: Node = expr_node


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

class TypeNode(Node):
    def __init__(self, token):
        self.token: Token = token
        self.id = self.token.id


class Var(Node):
    def __init__(self, token):
        self.token: Token = token
        self.id = self.token.id


class Num(Node):
    def __init__(self, token):
        self.token: Token = token
        self.id: int = self.token.id


class NoOp(Node):
    pass


class Interpreter:
    """
    Main interpreter class

    The interpreter is responsible for processing abstract syntax trees
    and compiling (not machine code) them into a final result.
    It works by 'visiting' each node in the tree and processing it based on its attributes and surrounding nodes.
    """

    global_scope = global_scope()

    def __init__(self, text):
        self.parser: Parser = Parser(text)

    def interpret(self):
        """
        Initiates the recursive descent algorithm,
        generates a syntax tree,
        and executes the code.
        """
        tree = self.parser.parse()
        if PRINT_TREE:
            print(tree)
            #exit()
        return self.visit(tree)

    def visit(self, node: Node):
        method_name = "visit_" + typeof(node).__name__
        visitor = getattr(self, method_name, self.default_visit)
        return visitor(node)

    def default_visit(self, node: Node):
        """
        Code gets executed when there is no `visit_(...)` function associated with a given `Node` object.
        """
        raise Exception(f"No visit_{typeof(node).__name__} method")

    def visit_Compound(self, node: Compound):
        for child in node.children:
            self.visit(child)

    def visit_VarDecl(self, node: VarDecl):
        variable_id = node.var_node.id
        variable_type = node.type_node.id

        if variable_id in self.global_scope:
            raise ValueError("Attempted to intialise variable with same name!")

        if node.expr_node is not None:
            self.global_scope[variable_id] = [variable_type, self.visit(node.expr_node)]
        else:
            self.global_scope[variable_id] = [variable_type, None]

    def visit_AssignOp(self, node: AssignOp):
        variable_id = node.left.id
        if variable_id in self.global_scope:
            self.global_scope[variable_id][1] = self.visit(node.right)
        else:
            raise ValueError("Attempted to assign value to uninitialised varaible!")

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
        elif node.op.type == type.INTEGER_DIV:
            return int(self.visit(node.left) // self.visit(node.right))
        elif node.op.type == type.FLOAT_DIV:
            return self.visit(node.left) / self.visit(node.right)

    def visit_TypeNode(self, node: TypeNode):
        # Not utilised yet
        pass

    def visit_Var(self, node: Var):
        variable_id = node.id
        val = self.global_scope.get(variable_id)[1]
        if val is None:
            raise NameError(repr(variable_id))
        else:
            return val

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
    def __init__(self,):
        self.filename = None

    def execute(self, mode):
        """
        Calls the relevant function for the given mode
        """
        if mode == "cmdline":
            self.cmdline_input()
        elif mode == "file":
            if self.filename is not None:
                self.file_input(self.filename)
            else:
                self.file_input()
        else:
            raise ValueError(f"mode '{repr(mode)}' is not a valid mode.")

    @staticmethod
    def cmdline_input():
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

            interpreter = Interpreter(text)
            result = interpreter.interpret()

            print("Globals:")
            print(interpreter.global_scope)
            print()
            print("Result: ")
            print(result)

    @staticmethod
    def file_input(filename: str = "./hello.sap"):
        """
        Run interpreter in file mode
        """
        file = open(filename, "r")
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


if __name__ == '__main__':
    driver = Driver()
    driver.filename = "./hello.sap"
    driver.execute(MODE)
