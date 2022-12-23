from collections import defaultdict

from sap.symbols import Symbol, BuiltinSymbol


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
            self.define(BuiltinSymbol("FLOAT"))

    def __str__(self):
        # Add header information
        text = "\nSCOPE (SCOPED SYMBOL TABLE):\n"
        text += f"Scope name    : {self.scope_name}\n"
        text += f"Scope level   : {self.scope_level}\n"
        text += f"Parent scope  : {self.parent_scope.scope_name if self.parent_scope else '<none>'}\n\n"
        text += "Scope symbol table contents\n"
        text += "---------------------------\n\n"

        # Organise contents of symbol table by symbol type.
        # This excludes the built-in symbol type, which is always printed first.
        symbols = defaultdict(list)
        for _, val in sorted(self._symbols.items()):
            symbols[val.__class__.__name__].append(val)

        symbols: dict[str, list] = dict(symbols)

        # At this point `symbols` is a dictionary (dict[str, list])
        # containing the name of each (present) symbol type and
        # a list of all the objects which are that type.

        # Show the built-in symbols first
        builtin_types: list[BuiltinSymbol] = symbols.get(BuiltinSymbol.__name__)
        if builtin_types is not None:
            for builtin_type in builtin_types:
                text += "  " + str(builtin_type) + "\n"
            text += "\n"
            # Remove it from `symbols` so it is not shown again
            del symbols[BuiltinSymbol.__name__]

        # Now show the remaining symbols
        for _, symbols in symbols.items():
            for symbol in symbols:
                text += "  " + str(symbol) + "\n"
            text += "\n"

        # Simple code to add bars around the top and bottom of the table output string.
        # The width of the bar is dependent on the longest line in the string.
        text = text.split("\n")
        del text[-1]
        longest_string_length = len(max(text, key=len))
        text.insert(2, "=" * (longest_string_length + 1))
        text.append("=" * (longest_string_length + 1) + "\n")
        text = "\n".join(text)

        return text

    def define(self, symbol: Symbol):
        """
        Adds a symbol to the symbol table
        """
        self._symbols[symbol.name] = symbol

    def lookup(self, symbol_name: str, search_parent_scopes: bool = True) -> Symbol | None:
        """
        Will search for the given symbol name in `self._symbols` and
        then it will search its parent scopes.

        `search_parent_scopes` (bool): Determines whether the function will search in parent scopes. 
        """
        symbol = self._symbols.get(symbol_name)
        if symbol is not None:
            return symbol

        # Recursively search up the scopes to find symbols
        if self.parent_scope is not None and search_parent_scopes:
            return self.parent_scope.lookup(symbol_name)
        else:
            return None
