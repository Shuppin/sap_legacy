"""
Module which is responsible for all operations involving the manipulation of data. 

Essentially it converts SAP types into native Python types
and performs native python operands on them
"""
if not __name__.startswith("modules."):
    from builtin_types import *
else:
    from modules.builtin_types import *

def operand(op_str: str, obj1: Type, obj2: Type = None) -> Type | None:
    """
    Perform an operation using a given opertor and given data types

    op_str (str): The symbol representing the operation you want to perform

    obj1 (Type): Left hand value

    obj2 (Type): Right hand value (optional)

    Example usage:
    ```
    >> op('+', Int(4), Int(6))
    Int(10)
    >> op('-', Int(2))
    Int(-2)
    ```
    """
    operator_mapping = {
        # Arithmetic
        '+': lambda x, y: x + y,
        '-': lambda x, y: x - y,
        '*': lambda x, y: x * y,
        '/': lambda x, y: x / y,
        '//': lambda x, y: x // y,

        # Logic
        'and': lambda x, y: x and y,
        'or': lambda x, y: x or y,
        # 'not' exluded because it is a unary operator which requires special handling

        # Comparison
        '==': lambda x, y: x == y,
        '~=': lambda x, y: x != y,
        '>': lambda x, y: x > y,
        '<': lambda x, y: x < y,
        '>=': lambda x, y: x >= y,
        '<=': lambda x, y: x <= y
    }

    # Get the operation associated with op_str
    operation = operator_mapping.get(op_str)

    # If no operation was found, `op_str` is not valid
    if operation is None:
        return None

    # Create a list contiang the type of each object
    types_list = [type(obj) for obj in [obj1, obj2]]

    # Making this operation a function reduces syntax significantly
    def matches(valid_ops, *given_types):
        """
        Checks if every type in `given_types` is a class
        or subclass of every type in `types_list`
        """
        return op_str in valid_ops and all([issubclass(types_list[i], given_types[i]) for i in range(len(types_list))])

    # Match code

    # Arithmetic
    if matches(['+','-','/','*','//'], Int|Float, Int|Float):
        return Float(operation(obj1.value, obj2.value))
    elif matches('-', Int|Float, type(None)):
        return Float(-obj1.value)

    # Logic
    elif matches(['and', 'or'], Int|Float|Bool, Int|Float|Bool):
        return Bool(0 if (operation(obj1.value, obj2.value) == 0) else 1)
    elif matches('not', Int|Float|Bool, type(None)):
        return Bool(1 if (obj1.value == 0) else 0)

    # Comparsion
    elif matches(['==', '~=', '>', '<', '>=', '<='], Int|Float|Bool, Int|Float|Bool):
        return Bool(int(operation(obj1.value, obj2.value)))
    
    else:
        return None
