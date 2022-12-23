from inspect import currentframe, getframeinfo

from sap.errors import ErrorCode, ParserError
from sap.tokens import Token, TokenType
from sap.lexer import Lexer
from sap.nodes import *

from sap import PRINT_TOKENS, PRINT_EAT_STACK


class Parser:
    """
    Main parser class

    The class is responsible for parsing the tokens and turning them into syntax trees.
    These trees make it easier to process the code and understand the relationships between tokens.

    For example give the set of tokens (equivalent to `1 + 1`):
    ```
    Token[type = type.INTEGER_CONST, id = '2', start_pos = 0]
    Token[type = type.PLUS, id = '+', start_pos = 2]
    Token[type = type.INTEGER_CONST, id = '2', start_pos = 4]
    ```

    The parser will generate:
    ```
    Program(
        statements: [
            BinOp(
                left: Num(
                    token: Token[type = type.INTEGER_CONST, id = '1', start_pos = 0],
                    id: 1
                ),
                op: Token[type = type.PLUS, id = '+', start_pos = 2],
                right: Num(
                    token: Token[type = type.INTEGER_CONST, id = '2', start_pos = 4],
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

    def error(self, error_code: ErrorCode, token: Token, message):
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
                        | empty
        """
        node = self.statement()

        results = [node]

        if not isinstance(node, NoOp):

            while self.current_token.type == TokenType.SEMI:
                if PRINT_EAT_STACK:
                    print("(Parser) Calling eat() from line", getframeinfo(currentframe()).lineno)
                self.eat(TokenType.SEMI)
                statement = self.statement()
                if isinstance(statement, NoOp):
                    self.eat(TokenType.EOF)
                else:
                    results.append(statement)

            #if not isinstance(results[-1], NoOp):
            #    if PRINT_EAT_STACK:
            #        print("(Parser) Calling eat() from line", getframeinfo(currentframe()).lineno)
            #    self.eat(TokenType.SEMI)

            # Commented out due to unknown behaviour
            #if self.current_token.type == TokenType.IDENTIFIER:
            #    self.error()

        return results

    def statement(self) -> Node:
        """
        statement -> compound_statement
                   | procedure_declaration
                   | variable_declaration
                   | variable_assignment
        """
        if self.current_token.type == TokenType.BEGIN:
            print("Compund here!")
            print(self.current_token.lineno)
            node = self.compound_statement()
        elif self.current_token.type == TokenType.DEFINITION:
            node = self.procedure_declaration()
        elif self.is_type():
            node = self.variable_declaration()
        elif self.current_token.type == TokenType.IDENTIFIER:
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
                message=f"'{self.current_token.id}' is not a valid type!"
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
        factor -> `PLUS` factor
                | `MINUS` factor
                | `INTEGER_CONST`
                | `FLOAT_CONST` 
                | `LPAREN` expr `RPAREN`
                | variable
        """
        token = self.current_token

        # `PLUS` factor
        if token.type == TokenType.PLUS:
            if PRINT_EAT_STACK:
                print("(Parser) Calling eat() from line", getframeinfo(currentframe()).lineno)
            self.eat(TokenType.PLUS)
            node = UnaryOp(token, self.factor())
            return node

        # `MINUS` factor
        elif token.type == TokenType.MINUS:
            if PRINT_EAT_STACK:
                print("(Parser) Calling eat() from line", getframeinfo(currentframe()).lineno)
            self.eat(TokenType.MINUS)
            node = UnaryOp(token, self.factor())
            return node

        # `INTEGER_CONST`
        elif token.type == TokenType.INTEGER_CONST:
            if PRINT_EAT_STACK:
                print("(Parser) Calling eat() from line", getframeinfo(currentframe()).lineno)
            self.eat(TokenType.INTEGER_CONST)
            return Num(token)

        # `FLOAT_CONST`
        elif token.type == TokenType.FLOAT_CONST:
            if PRINT_EAT_STACK:
                print("(Parser) Calling eat() from line", getframeinfo(currentframe()).lineno)
            self.eat(TokenType.FLOAT_CONST)
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
                        | empty

        statement -> compound_statement
                   | procedure_declaration
                   | variable_declaration
                   | variable_assignment

        compound_statement -> `BEGIN` statement_list `END`

        procedure_declaration -> `DEFINITION` variable `LPAREN` formal_parameter_list `RPAREN` compound_statement
                               | `DEFINITION` variable `LPAREN` formal_parameter_list `RPAREN` `RETURNS_OP` type_spec compound_statement

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

        factor -> `PLUS` factor
                | `MINUS` factor
                | `INTEGER_CONST`
                | `FLOAT_CONST` 
                | `LPAREN` expr `RPAREN`
                | variable

        variable -> `IDENTIFIER`
        ```
        """
        node = self.program()
        if self.current_token.type != TokenType.EOF:
            self.error(
                error_code=ErrorCode.SYNTAX_ERROR,
                token=self.current_token,
                message=f"Program terminated with <{self.current_token.type.value}>, not <{TokenType.EOF}>"
            )

        return node
