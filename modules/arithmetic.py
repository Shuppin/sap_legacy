"""
Module which is responsible for all arithmetic operations.

Essentially it converts SAP types into native Python types
and performs native python arithmetic on them
"""
# Not used
# from logging import Formatter
# from logging import FileHandler
# from logging import getLogger
from math import floor

if __name__ == '__main__':
    # Not used
    # from config         import ConfigParser
    from overloading    import create_overload

    from builtin_types  import *
else:
    # Not used
    # from modules.config       import ConfigParser
    from modules.overloading    import create_overload
    from modules.builtin_types  import *

__all__ = [
    "add",
    "sub",
    "mult",
    "truediv",
    "floordiv",
    "negate"
]

# Currently unused
"""
# Load config information
config = ConfigParser()

# Create formatter object using value(s) from config file
# Formatter defines how each line will look in the config file
formatter = Formatter(config.getstr("logging.format"), datefmt=config.getstr("logging.datefmt"))

# Handler defines which file to write to and how to write to it
handler = FileHandler(config.getstr("logging.destination"), mode="a")
handler.setFormatter(formatter)

# Get logging level from config file
log_level = config.getint(f"logging.levels.{config.getstr('logging.level')}")

# Create and setup logging object
logger = getLogger("arithmetic")
logger.setLevel(log_level)
logger.addHandler(handler)

LOG_ALL = config.getint("logging.levels.ALL")
"""
@create_overload
def add() -> Type: ...

@add.overload()
def add(int1: Int, int2: Int):
    return Int(str(int(int1.literal_value)+int(int2.literal_value)))

@add.overload()
def add(int1: Int, float1: Float):
    return Float(str(int(int1.literal_value)+float(float1.literal_value)))

@add.overload()
def add(float1: Float, float2: Float):
    return Float(str(float(float1.literal_value)+float(float2.literal_value)))


@create_overload
def sub() -> Type: ...

@sub.overload()
def sub(int1: Int, int2: Int):
    return Int(str(int(int1.literal_value)-int(int2.literal_value)))

@sub.overload()
def sub(int1: Int, float1: Float):
    return Float(str(int(int1.literal_value)-float(float1.literal_value)))

@sub.overload()
def sub(float1: Float, float2: Float):
    return Float(str(float(float1.literal_value)-float(float2.literal_value)))


@create_overload
def mult() -> Type: ...

@mult.overload()
def mult(int1: Int, int2: Int):
    return Int(str(int(int1.literal_value)*int(int2.literal_value)))

@mult.overload()
def mult(int1: Int, float1: Float):
    return Float(str(int(int1.literal_value)*float(float1.literal_value)))

@mult.overload()
def mult(float1: Float, float2: Float):
    return Float(str(float(float1.literal_value)*float(float2.literal_value)))


@create_overload
def truediv() -> Type: ...

@truediv.overload()
def truediv(int1: Int, int2: Int):
    return Float(str(int(int1.literal_value)/int(int2.literal_value)))

@truediv.overload()
def truediv(int1: Int, float1: Float):
    return Float(str(int(int1.literal_value)/float(float1.literal_value)))

@truediv.overload()
def truediv(float1: Float, float2: Float):
    return Float(str(float(float1.literal_value)-float(float2.literal_value)))


@create_overload
def floordiv() -> Type: ...

@floordiv.overload()
def floordiv(int1: Int, int2: Int):
    return Int(str(floor(int(int1.literal_value)/int(int2.literal_value))))

@floordiv.overload()
def floordiv(int1: Int, float1: Float):
    return Int(str(floor(int(int1.literal_value)/float(float1.literal_value))))

@floordiv.overload()
def floordiv(float1: Float, float2: Float):
    return Int(str(floor(float(float1.literal_value)/float(float2.literal_value))))


@create_overload
def negate() -> Type: ...

@negate.overload()
def negate(int1: Int):
    return Int(-int(int1.literal_value))

@negate.overload()
def negate(float1: Float):
    return Float(-float(float1.literal_value))
    

if __name__ == '__main__':
    # I'm so good at writing tests
    print("Adding")
    print(add(Int(1), Int(1)))
    print(add(Int(1), Float(1.34)))
    print(add(Float(2), Float(3.14)))
    print(add(Bool(3), Int(3)))
    print("Subtracting")
    print(sub(Int(10), Int(1)))
    print(sub(Int(1), Float(1.34)))
    print(sub(Float(2), Float(3.14)))
    print(sub(Bool(3), Int(3)))
    print("Multiplying")
    print(mult(Int(10), Int(1)))
    print(mult(Int(1), Float(1.34)))
    print(mult(Float(2), Float(3.14)))
    print(mult(Bool(3), Int(3)))
    print("True Divide")
    print(truediv(Int(10), Int(1)))
    print(truediv(Int(1), Float(1.34)))
    print(truediv(Float(2), Float(3.14)))
    print(truediv(Bool(3), Int(3)))
    print("Floor Divide")
    print(floordiv(Int(10), Int(2)))
    print(floordiv(Int(2), Float(1.34)))
    print(floordiv(Float(22), Float(3.14)))
    print(floordiv(Bool(3), Int(3)))