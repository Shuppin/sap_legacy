"""
Module containing all the built-in types
"""
from __future__ import annotations

class Type:
    def __init__(self, value) -> None:
        self.value = value

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.value})"
    
    __repr__ = __str__
    
    def to(self, to_type: Type):
        if to_type == type(self):
            return self
        matching_function = getattr(self, "to" + to_type.__name__, self._no_matching_function)
        return matching_function()
        
    def _no_matching_function(self):
        return None
        

class NoneType(Type):
    def __init__(self) -> None:
        super().__init__(None)
            
    def toFloat(self):
        return Float(0.0)
    
    def toInt(self):
        return Int(0)
    
    def toBool(self):
        return Bool(0)
    

class Bool(Type):
    def __init__(self, value: int) -> None:
        value: int = 0 if (int(value) == 0) else 1
        super().__init__(value)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({'False' if (self.value == 0) else 'True'})"

    def toInt(self):
        return Int(self.value)

    def toFloat(self):
        return Float(float(self.value))


class Int(Type):
    def __init__(self, value: int) -> None:
        super().__init__(int(value))

    def toFloat(self):
        return Float(float(self.value))
    
    def toBool(self):
        return Bool(int(self.value))


class Float(Type):
    def __init__(self, value: str) -> None:
        super().__init__(float(value))
        
    def toInt(self):
        return Int(int(self.value))
    
    def toBool(self):
        return Bool(int(self.value))

# We don't want to import this into other files
del annotations
    