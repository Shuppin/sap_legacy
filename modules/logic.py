"""
Module which is responsible for all logic operations (NOT, AND, OR).

Essentially it converts SAP types into native Python types
and performs native python logic on them
"""
if __name__ == '__main__':
    from overloading    import create_overload
    from builtin_types  import Bool
else:
    from modules.overloading    import create_overload
    from modules.builtin_types  import Bool
    
__all__ = [
    "bool_not",
    "bool_and",
    "bool_or"
]
    
@create_overload
def bool_not() -> Bool: ...

@bool_not.overload()
def bool_not(bool1: Bool):
    return Bool(int(not bool1.value))


@create_overload
def bool_and() -> Bool: ...

@bool_and.overload()
def bool_and(bool1: Bool, bool2: Bool) -> Bool:
    return Bool(int(bool1.value and bool2.value))


@create_overload
def bool_or() -> Bool: ...

@bool_or.overload()
def bool_or(bool1: Bool, bool2: Bool) -> Bool:
    return Bool(int(bool1.value or bool2.value))