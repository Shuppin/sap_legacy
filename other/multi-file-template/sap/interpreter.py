from sap.nodevstr import NodeVisitor
from sap.tokens import TokenType
from sap.nodes import *


# Temporary
class GlobalScope(dict):
    def __init_subclass__(cls):
        return super().__init_subclass__()

    def __str__(self):
        text = []

        for key, value in sorted(self.items(), key=lambda x: x[1][0], reverse=True):
            text.append("  <" + str(value[0]) + "> " + str(key) + " = " + str(value[1]))

        return "\n".join(text)


# Currently some unloved garbárge
class Interpreter(NodeVisitor):
    """
    Main interpreter class

    The interpreter is responsible for processing abstract syntax trees
    and compiling (not machine code) them into a final result.
    It works by 'visiting' each node in the tree and processing it based on its attributes and surrounding nodes.

    It also handles type-checking at runtime
    """

    global_scope = GlobalScope()

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
            raise ValueError("Interpreter :: Attempted to assign value to uninitialised variable!")

    def visit_UnaryOp(self, node: UnaryOp):
        if node.op.type == TokenType.PLUS:
            return +self.visit(node.expr)
        elif node.op.type == TokenType.MINUS:
            return -self.visit(node.expr)

    def visit_BinOp(self, node: BinOp):
        if node.op.type == TokenType.PLUS:
            return self.visit(node.left) + self.visit(node.right)
        elif node.op.type == TokenType.MINUS:
            return self.visit(node.left) - self.visit(node.right)
        elif node.op.type == TokenType.MULT:
            return self.visit(node.left) * self.visit(node.right)
        elif node.op.type == TokenType.INTEGER_DIV:
            return int(self.visit(node.left) // self.visit(node.right))
        elif node.op.type == TokenType.FLOAT_DIV:
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
