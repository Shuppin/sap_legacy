"""
Module which is responsible for all comparison operations (EQUAL, LESS THAN, etc...).

Essentially it converts SAP types into native Python types
and performs native python comparison on them
"""
if __name__ == '__main__':
    from overloading    import create_overload
    from builtin_types  import *
else:
    from modules.overloading    import create_overload
    from modules.builtin_types  import *
    
@create_overload
def is_equal() -> Bool: ...

@is_equal.overload()
def is_equal(bool1: Bool, bool2: Bool) -> Bool:
    return Bool(int(bool1.value == bool2.value))

@is_equal.overload()
def is_equal(int1: Int, int2: Int) -> Bool:
    return Bool( int( int(int1.literal_value) == int(int2.literal_value) ) )

@is_equal.overload()
def is_equal(float1: Int, float2: Int) -> Bool:
    return Bool( int( float(float1.literal_value) == float(float2.literal_value) ) )


@create_overload
def is_not_equal() -> Bool: ...

@is_not_equal.overload()
def is_not_equal(bool1: Bool, bool2: Bool):
    return Bool(int(bool1.value != bool2.value))

@is_not_equal.overload()
def is_not_equal(int1: Int, int2: Int) -> Bool:
    return Bool( int( int(int1.literal_value) != int(int2.literal_value) ) )

@is_not_equal.overload()
def is_not_equal(float1: Int, float2: Int) -> Bool:
    return Bool( int( float(float1.literal_value) != float(float2.literal_value) ) )


@create_overload
def is_less_than() -> Bool: ...

@is_less_than.overload()
def is_less_than(bool1: Bool, bool2: Bool) -> Bool:
    ...
