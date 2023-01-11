from math           import floor
from inspect        import getfullargspec
from collections    import Counter

from builtin_types  import *

__all__ = [
    "add",
    "sub",
    "mult",
    "truediv",
    "floordiv",
    "negate"
]

class _overload:
    def __init__(self, f):
        self.cases = {}

    def overload(self):
        def store_function(f):
            type_annotations = getfullargspec(f).annotations
            if type_annotations.get("return"):
                del type_annotations["return"]
            if self.cases.get(tuple(type_annotations.values())):
                raise Exception("Function " + repr(f.__name__) + " has duplicate overloads!")
            self.cases[tuple(type_annotations.values())] = f
            return self
        return store_function

    def _rearrange_values(self, spec_list, types_list):

        if not len(spec_list) == len(types_list):
            print("Returning none due to size difference")
            return None

        rearranged_values = [None]*len(types_list)
        spec_occurance_counter = Counter(spec_list)
        value_occurance_counter = Counter(types_list)

        if not spec_occurance_counter == value_occurance_counter:
            print("Returning none due to different lists")
            print(types_list)
            print(spec_occurance_counter)
            print(value_occurance_counter)
            return None

        for value in types_list:
            if value_occurance_counter.get(type(value)) is None:
                value_occurance_counter[type(value)] = 1
            else:
                value_occurance_counter[type(value)] += 1
            try:
                rearranged_values[[i for i, spec in enumerate(spec_list) if issubclass(spec, type(value))][value_occurance_counter[type(value)]-1]] = value
            except IndexError:
                return None

        return tuple(rearranged_values)

    def __call__(self, *args):
        for existing_arguments in self.cases.keys():
            sorted_values = None
            sorted_values = self._rearrange_values(existing_arguments, [type(arg) for arg in args])
            if sorted_values is not None:
                break

        if sorted_values is None:
            return None
            
        print("Overloading args: ", [type(arg) for arg in sorted_values])
        function = self.cases.get(tuple([type(arg) for arg in sorted_values]))
        if function is None:
            return None
        return function(*sorted_values)


@_overload
def add() -> Type:...

@add.overload()
def add(int1: Int, int2: Int):
    print("ints")
    return Int(str(int(int1.literal_value)+int(int2.literal_value)))

@add.overload()
def add(int1: Int, float1: Float):
    print("int, float")
    return Float(str(int(int1.literal_value)+float(float1.literal_value)))

@add.overload()
def add(float1: Float, float2: Float):
    print("floats")
    return Float(str(float(float1.literal_value)+float(float2.literal_value)))


@_overload
def sub() -> Type:...

@sub.overload()
def sub(int1: Int, int2: Int):
    return Int(str(int(int1.literal_value)-int(int2.literal_value)))

@sub.overload()
def sub(int1: Int, float1: Float):
    return Float(str(int(int1.literal_value)-float(float1.literal_value)))

@sub.overload()
def sub(float1: Float, float2: Float):
    return Float(str(float(float1.literal_value)-float(float2.literal_value)))


@_overload
def mult() -> Type:...

@mult.overload()
def mult(int1: Int, int2: Int):
    return Int(str(int(int1.literal_value)*int(int2.literal_value)))

@mult.overload()
def mult(int1: Int, float1: Float):
    return Float(str(int(int1.literal_value)*float(float1.literal_value)))

@mult.overload()
def mult(float1: Float, float2: Float):
    return Float(str(float(float1.literal_value)*float(float2.literal_value)))


@_overload
def truediv() -> Type:...

@truediv.overload()
def truediv(int1: Int, int2: Int):
    result = int(int1.literal_value)/int(int2.literal_value)
    if result == int(result):
        return Int(str(result))
    else:
        return Float(str(result))

@truediv.overload()
def truediv(int1: Int, float1: Float):
    return Float(str(int(int1.literal_value)/float(float1.literal_value)))

@truediv.overload()
def truediv(float1: Float, float2: Float):
    return Float(str(float(float1.literal_value)-float(float2.literal_value)))


@_overload
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


@_overload
def negate() -> Type: ...

@negate.overload()
def negate(int1: Int):
    return Int(-int(int1.literal_value))

@negate.overload()
def negate(int1: Float):
    return Float(-int(int1.literal_value))

if __name__ == '__main__':
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
