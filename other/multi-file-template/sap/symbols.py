from sap.nodes import Param


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
    Symbol which represents user-defined variables
    """
    def __init__(self, name, datatype):
        super().__init__(name, datatype)

    def __str__(self):
        return f"<variable> (id: '{self.name}', type: '{self.type.name}')"


class ProcedureSymbol(Symbol):
    """
    Symbol which represents procedure declarations
    """
    def __init__(self, name, params=[]):
        super().__init__(name)
        self.params: list[Param] = params

    def __str__(self):
        if len(self.params) == 0:
            return f"<procedure> (id: '{self.name}', parameters: <no params>)"
        else:
            # Okay, yes this is horrendous don't @me
            return (f"<procedure> (id: '{self.name}', parameters: "
                    f"{', '.join(list(map(lambda param: f'({param.var_node.id}, <{param.type_node.id}>)', self.params)))})")
