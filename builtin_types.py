"""
Module containing all the built-in types
"""
class Type:
    def __init__(self, literal_value: str) -> None:
        self.literal_value: str = str(literal_value)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.literal_value})"

    __repr__ = __str__


class Bool(Type):
    def __init__(self, value: int) -> None:
        self.value: int = value
        literal_value = "True" if value else "False"
        super().__init__(literal_value)


class Int(Type):
    def __init__(self, literal_value: str) -> None:
        super().__init__(literal_value)


class Float(Type):
    def __init__(self, literal_value: str) -> None:
        super().__init__(literal_value)