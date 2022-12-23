from typing import Any

from sap.nodes import Node


class NodeVisitor:
    """
    NodeVisitor base class

    Base class for all classes which visit/walk through a syntax tree
    """

    def visit(self, node: Node) -> Any:
        """
        Executes the visit function associated with the given node
        """
        method_name = "visit_" + type(node).__name__
        visitor = getattr(self, method_name, self.default_visit)
        return visitor(node)

    def default_visit(self, node: Node):
        """
        Code gets executed when there is no `visit_(...)` function associated with a given `Node` object.
        """
        raise Exception(f"{self.__class__.__name__} :: No visit_{type(node).__name__} method")
