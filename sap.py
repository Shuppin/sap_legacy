from __future__ import annotations

from inspect import currentframe, getframeinfo
from collections import defaultdict
from enum import Enum

# Constants

# Modes:
# "cmdline"
# "file"

MODE = "file"

PRINT_TOKENS = True
PRINT_EAT_STACK = False
PRINT_TREE = True
PRINT_SCOPE = True

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
    INTEGER         = object()
    REAL            = object()
    INTEGER_CONST   = object()
    REAL_CONST      = object()
    MULT            = object()
    INTEGER_DIV     = object()
    FLOAT_DIV       = object()
    PLUS            = object()
    MINUS           = object()
    RETURNS_OP      = object()
    LPAREN          = object()
    RPAREN          = object()
    IDENTIFIER      = object()
    DEFINITION      = object()
    ASSIGN          = object()
    COMMENT         = object()
    SEMI            = object()
    COLON           = object()
    COMMA           = object()
    BEGIN           = object()
    END             = object()
    EOF             = object()


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
        self.current_char: str | None = self.text[self.pos]

        self.RESERVED_KEYWORDS: dict = {
            'int': type.INTEGER,
            'real': type.REAL,
            'def': type.DEFINITION
        }

    def error(self):
        raise Exception(f'Lexer :: Error parsing input on line {self.lineno}, pos {self.linepos}\n')

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

    def skip_multi_comment(self):
        """Advances `self.pos` until a comment terminator (*/) has been reached"""
        while not (self.current_char == "*" and self.peek() == "/"):
            self.advance()
        # Advance twice more to skip over the final "*/"
        self.advance()
        self.advance()

    def skip_comment(self):
        while self.current_char is not None and not self.current_char == "\n":
            self.advance()
        self.advance()

    def number(self) -> Token:
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
                self.skip_multi_comment()
                continue

            elif self.current_char == "/" and self.peek() == "/":
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
                # Disabled in place of commas
                #else:
                #    token = Token(self.pos, type.INTEGER_DIV, self.current_char)
                #    self.advance()

                self.advance()

                return token

            elif self.current_char == '+':
                token = Token(self.pos, type.PLUS, self.current_char)
                self.advance()
                return token

            elif self.current_char == '-':
                if self.peek() == ">":
                    token = Token(self.pos, type.RETURNS_OP, self.current_char)
                    self.advance()
                    self.advance()
                else:
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

            elif self.current_char == ",":
                token = Token(self.pos, type.COMMA, self.current_char)
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
        raise Exception(f"Parser :: Error parsing input on line {self.lexer.lineno}, pos {self.lexer.linepos}\n"
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

    # Could be a function native to `Token`
    def is_type(self) -> bool:
        """
        Check if the current token is a datatype
        """
        if self.current_token.type in [
            type.INTEGER,
            type.REAL
        ]:
            return True
        else:
            return False

    # Grammar definitions

    def program(self) -> Program:
        """
        program -> (statement_list | compound_statement) <`EOF`>
        """
        node = Program()

        if self.current_token.type == type.BEGIN:
            node.statements.append(self.compound_statement())
        else:
            node.statements = self.statement_list()

        if PRINT_EAT_STACK:
            print("Calling eat() from line", getframeinfo(currentframe()).lineno)
        self.eat(type.EOF)

        return node

    def type_spec(self) -> TypeNode:
        """
        type_spec -> `INTEGER` | `REAL`
        """
        token = self.current_token
        if self.is_type():
            if PRINT_EAT_STACK:
                print("Calling eat() from line", getframeinfo(currentframe()).lineno)
            self.eat(token.type)
        else:
            self.error()

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
                   | procedure_declaration
                   | variable_declaration
                   | variable_assingment
                   | empty
        """
        if self.current_token.type == type.BEGIN:
            node = self.compound_statement()
        elif self.current_token.type == type.DEFINITION:
            node = self.procedure_declaration()
        elif self.is_type():
            node = self.variable_declaration()
        elif self.current_token.type == type.IDENTIFIER:
            node = self.variable_assignment()
        else:
            node = self.empty()

        return node

    def formal_parameter_list(self) -> list[Param]:
        """
        formal_parameter_list -> formal_parameter | empty | formal_parameter `COMMA` formal_parameter_list
        """
        if self.current_token.type == type.RPAREN:
            results = []

        else:
            node = self.formal_parameter()

            results = [node]

            while self.current_token.type == type.COMMA:
                if PRINT_EAT_STACK:
                    print("Calling eat() from line", getframeinfo(currentframe()).lineno)
                self.eat(type.COMMA)
                results.append(self.formal_parameter())

            if self.current_token.type == type.IDENTIFIER:
                self.error()

        return results

    def formal_parameter(self) -> Param:
        """
        formal_parameter -> type_spec variable
        """
        type_node = self.type_spec()
        var_node = self.variable()

        param_node = Param(var_node, type_node)

        return param_node

    def procedure_declaration(self) -> ProcedureDecl:
        """
        procedure_declaration -> `DEFINITION` variable `LPAREN` formal_parameter_list `RPAREN` compound_statement
                               | `DEFINITION` variable `LPAREN` formal_parameter_list `RPAREN` `RETURNS_OP` type_spec compound_statement  
        """
        if PRINT_EAT_STACK:
            print("Calling eat() from line", getframeinfo(currentframe()).lineno)
        self.eat(type.DEFINITION)

        procedure_var = self.variable()

        if PRINT_EAT_STACK:
            print("Calling eat() from line", getframeinfo(currentframe()).lineno)
        self.eat(type.LPAREN)

        params = self.formal_parameter_list()

        if PRINT_EAT_STACK:
            print("Calling eat() from line", getframeinfo(currentframe()).lineno)
        self.eat(type.RPAREN)

        if self.current_token.type == type.BEGIN:

            body = self.compound_statement()

            proc_decl = ProcedureDecl(procedure_var, params, body)

        elif self.current_token.type == type.RETURNS_OP:
            
            if PRINT_EAT_STACK:
                print("Calling eat() from line", getframeinfo(currentframe()).lineno)
            self.eat(type.RETURNS_OP)

            return_type = self.type_spec()

            body = self.compound_statement()

            proc_decl = ProcedureDecl(procedure_var, params, body, return_type=return_type)

        else:
            self.error()

        return proc_decl

    def variable_declaration(self) -> VarDecl | Compound:
        """
        variable_declaration -> type_spec variable `ASSIGN` expr
                              | type_spec variable (`COMMA` variable)*

        """
        type_node = self.type_spec()

        var_node = self.variable()

        # type_spec variable `ASSIGN` expr
        if self.current_token.type == type.ASSIGN:
            assign_op = self.current_token
            if PRINT_EAT_STACK:
                print("Calling eat() from line", getframeinfo(currentframe()).lineno)
            self.eat(type.ASSIGN)
            expr_node = self.expr()

            node = VarDecl(type_node, var_node, assign_op, expr_node)
        
        # type_spec variable (`COMMA` variable)*
        else:
            node = Compound()
            node.children.append(VarDecl(type_node, var_node))
            while self.current_token.type == type.COMMA:
                self.eat(type.COMMA)
                var_node = self.variable()
                node.children.append(VarDecl(type_node, var_node))

        return node


    def variable_assignment(self) -> AssignOp:
        """
        variable_assignment -> variable `ASSIGN` expr
        """
        var_node = self.current_token
        if PRINT_EAT_STACK:
            print("Calling eat() from line", getframeinfo(currentframe()).lineno)
        self.eat(type.IDENTIFIER)

        assign_op = self.current_token
        if PRINT_EAT_STACK:
            print("Calling eat() from line", getframeinfo(currentframe()).lineno)
        self.eat(type.ASSIGN)

        right = self.expr()
        node = AssignOp(var_node, assign_op, right)

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
                   | procedure_declaration
                   | variable_declaration
                   | variable_assingment
                   | empty

        formal_parameter_list -> formal_parameter
                               | empty
                               | formal_parameter `COMMA` formal_parameter_list

        formal_parameter -> type_spec variable

        procedure_declaration -> `DEFINITION` variable `LPAREN` formal_parameter_list `RPAREN` compound_statement

        variable_declaration -> type_spec variable `ASSIGN` expr
                              | type_spec variable (`COMMA` variable)*

        variable_assignment -> variable `ASSIGN` expr

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
#   Node visitor code                     #
#                                         #
###########################################

class NodeVisitor:
    """
    NodeVisitor base class

    Base class for all classes which visit/walk through a syntax tree
    """

    def visit(self, node: Node):
        method_name = "visit_" + typeof(node).__name__
        visitor = getattr(self, method_name, self.default_visit)
        return visitor(node)


    def default_visit(self, node: Node):
        """
        Code gets executed when there is no `visit_(...)` function associated with a given `Node` object.
        """
        raise Exception(f"{self.__class__.__name__} :: No visit_{typeof(node).__name__} method")


###########################################
#                                         #
#  Node defintions                        #
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
                    if isinstance(node, Node):
                        text += "   " * (depth + 1) + node.__class__.__name__ + "(\n"
                        text += self._print_children(node.__dict__, depth + 2)
                        text += "   " * (depth + 1) + "),\n"
                    else:
                        raise TypeError(f"Cannot print type '{typeof(node)}'")
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
    Symbol which represents user-defined types
    """
    def __init__(self, name, datatype):
        super().__init__(name, datatype)
    
    def __str__(self):
        return f"<variable> (id: '{self.name}', type: '{self.type.name}')"


class ProdcedureSymbol(Symbol):
    def __init__(self, name, params=[]):
        super().__init__(name)
        self.params: list[Param] = params

    def __str__(self):
        if len(self.params) == 0:
            return f"<procedure> (id: '{self.name}', parameters: <no params>)"
        else:        
            return f"<procedure> (id: '{self.name}', parameters: {', '.join(list(map(lambda param: f'({param.var_node.id}, <{param.type_node.id}>)', self.params)))})"


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
            self.define(BuiltinSymbol("REAL"))

    def __str__(self):
        text = "\nSCOPE (SCOPED SYMBOL TABLE):\n"
        text += f"Scope name    : {self.scope_name}\n"
        text += f"Scope level   : {self.scope_level}\n"
        text += f"Parent scope  : {self.parent_scope.scope_name if self.parent_scope else '<none>'}\n\n"
        text += "Scope symbol table contents\n"
        text += "---------------------------\n\n"
        symbols = defaultdict(list)

        for _, val in sorted(self._symbols.items()):
            symbols[val.__class__.__name__].append(val)

        symbols = dict(symbols)
        builtin_types = symbols.get(BuiltinSymbol.__name__)

        if builtin_types != None:
            for builtin_type in builtin_types:
                text += "  " + str(builtin_type) + "\n"
            text += "\n"

            del symbols[BuiltinSymbol.__name__]
            
        for _, symbols in symbols.items():
            for symbol in symbols:
                text += "  " + str(symbol) + "\n"
            text += "\n"

        # Simple code to add bars around the top and bottom of the string,
        # according to the longest line in the string.
        text = text.split("\n")
        del text[-1]
        longest_string_length = len(max(text, key=len))
        text.insert(2, "="*(longest_string_length+1))
        text.append("="*(longest_string_length+1) + "\n")
        text = "\n".join(text)

        return text

    def define(self, symbol: Symbol):
        self._symbols[symbol.name] = symbol

    def lookup(self, name: str, search_parent_scopes: bool=True) -> Symbol | None:
        symbol = self._symbols.get(name)
        if symbol is not None:
            return symbol

        # Recursively search up the scopes to find symbols
        if self.parent_scope is not None and search_parent_scopes:
            return self.parent_scope.lookup(name)
        else:
            return None


class SemanticAnalyser(NodeVisitor):
    """
    Constructs the symbol table and performs type-checks before runtime
    """
    def __init__(self):
        self.current_scope: SymbolTable | None = None

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
            raise NameError("TypeChecker :: Attempted to intialise variable with same name!")
        
        var_symbol = VarSymbol(var_id, type_symbol)
        self.current_scope.define(var_symbol)

        if node.expr_node is not None:
            self.visit(node.expr_node)

    def visit_ProcedureDecl(self, node: ProcedureDecl):
        proc_name = node.procedure_var.id
        proc_params: list[Param] = node.params

        if self.current_scope.lookup(proc_name) is not None:
            raise NameError("TypeChecker :: Attempted to declare procedure with same name!")

        proc_symbol = ProdcedureSymbol(proc_name, proc_params)
        self.current_scope.define(proc_symbol)

        proc_scope = SymbolTable(scope_name=proc_name, scope_level=self.current_scope.scope_level+1, parent_scope=self.current_scope)
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
            raise NameError(f"TypeChecker :: Attempted to assign value to uninitialised varaible {repr(var_id)}!")

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
            raise NameError(f"TypeChecker :: Unrecognised type {repr(type_id)}")
        else:
            return type_symbol

    def visit_Var(self, node: Var):
        var_id = node.id
        var_symbol = self.current_scope.lookup(var_id)

        if var_symbol is None:
            raise NameError(f"TypeChecker :: Attempted to use uninitialised value {repr(var_id)}")
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

class Interpreter(NodeVisitor):
    """
    Main interpreter class

    The interpreter is responsible for processing abstract syntax trees
    and compiling (not machine code) them into a final result.
    It works by 'visiting' each node in the tree and processing it based on its attributes and surrounding nodes.

    It also handles type-checking at runtime
    """

    global_scope = global_scope()

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
            raise ValueError("Interpreter :: Attempted to assign value to uninitialised varaible!")

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
    def __init__(self,):
        self.filename: str | None = None

    def run_program(self, mode):
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

    def process(self, code: str):

        parser = Parser(code)
        symbol_table = SemanticAnalyser()
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

            self.process(text)

    def file_input(self, filename: str = "./hello.sap"):
        """
        Run interpreter in file mode
        """
        file = open(filename, "r")
        text = file.read()

        if not text:
            return

        self.process(text)

        
if __name__ == '__main__':
    driver = Driver()
    driver.filename = "./syntax_showcase.sap"
    driver.run_program(MODE)
