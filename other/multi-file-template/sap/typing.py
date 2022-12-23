from sap.errors import ErrorCode, SemanticAnalyserError
from sap.symbols import VarSymbol, ProcedureSymbol
from sap.nodevstr import NodeVisitor
from sap.symtable import SymbolTable
from sap.nodes import *

from sap import PRINT_SCOPE


class SemanticAnalyser(NodeVisitor):
    """
    Constructs the symbol table and performs type-checks before runtime
    """
    def __init__(self, text):
        self.text_lines: list[str] = text.split('\n')
        self.current_scope: SymbolTable | None = None

    def error(self, error_code: ErrorCode, token: Token, message):
        error = SemanticAnalyserError(error_code, message, token, surrounding_lines=self.text_lines)
        error.trigger()

    def analyse(self, tree: Node):
        return self.visit(tree)

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
            self.error(
                error_code=ErrorCode.NAME_ERROR,
                token=node.var_node.token,
                message="Cannot initialise variable with same name"
            )

        var_symbol = VarSymbol(var_id, type_symbol)
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

        proc_symbol = ProcedureSymbol(proc_name, proc_params)
        self.current_scope.define(proc_symbol)

        proc_scope = SymbolTable(scope_name=proc_name, scope_level=self.current_scope.scope_level + 1,
                                 parent_scope=self.current_scope)
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
            self.error(
                error_code=ErrorCode.NAME_ERROR,
                token=node.left,
                message=f"Variable {repr(var_id)} does not exist"
            )

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
            self.error(
                error_code=ErrorCode.NAME_ERROR,
                token=node.token,
                message=f"Unrecognised type {repr(type_id)}"
            )
        else:
            return type_symbol

    def visit_Var(self, node: Var):
        var_id = node.id
        var_symbol = self.current_scope.lookup(var_id)

        if var_symbol is None:
            self.error(
                error_code=ErrorCode.NAME_ERROR,
                token=node.token,
                message=f"Variable {repr(var_id)} does not exist"
            )
        else:
            return var_symbol

    def visit_Num(self, node):
        pass

    def visit_NoOp(self, node):
        pass
