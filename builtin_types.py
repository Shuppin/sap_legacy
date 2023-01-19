"""
Module containing all the built-in types
"""
from __future__ import annotations

class Type:
    def __init__(self, literal_value: str) -> None:
        self.literal_value: str = str(literal_value)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.literal_value})"
    
    __repr__ = __str__
    
    def to(self, to_type: Type):
        matching_function = getattr(self, "to" + to_type.__name__, self._no_matching_function)
        return matching_function()
        
    def _no_matching_function(self):
        return None
        

class NoneType(Type):
    def __init__(self) -> None:
        super().__init__("")
        
    def toFloat(self):
        return Float("0.0")
    
    def toInt(self):
        return Int("0")
    
    def toBool(self):
        return Bool(0)
    

class Bool(Type):
    def __init__(self, value: int) -> None:
        self.value: int = 1 if value > 0 else 0
        literal_value = "True" if value > 0 else "False"
        super().__init__(literal_value)


class Int(Type):
    def __init__(self, literal_value: str) -> None:
        super().__init__(literal_value)

    def toFloat(self):
        return Float(f"{self.literal_value}.0")


class Float(Type):
    def __init__(self, literal_value: str) -> None:
        super().__init__(literal_value)
        
    def toInt(self):
        truncated = ""
        # Add all numbers before decimal
        for char in self.literal_value:
            if char == ".":
                break
            truncated += char
        return Int(truncated)
    