from sap.tokens import Token


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

        Nodes look like:
        ```
        BinOp(
            left: Num(...),
            ...
        ),
        ```

        Lists look like:
        ```
        list_name: [
            ...
        ]
        ```

        Everything else looks like:
        ```
        object_name: object_value,
        ```

        """
        text = ""

        if depth == 1:
            text += self.__class__.__name__ + "(\n"

        # Looks ugly, will always look ugly, but the output looks great!
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
                        raise TypeError(f"Cannot print type '{type(node)}'")
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
