# Taken from [overloading.py] create_overload._rearrange_objects
"""
Takes in a list of objects and a list of types and rearranges
the list of objects so that they match the order of the types
list.

Example:
```
objects = [3, 4, "hello"]
types = [int, str, int]
result = _rearrange_objects(objects, types)
print(result)
>>> [3, "hello", 4]
```
"""
objects = [1, "hello", 2.5, "world", 3, 4.5, 7]
types = [int, int, str, int | float, str, float]

new_objects = []

for t in types:
    compatible_types = [obj for obj in objects if issubclass(type(obj), t)]
    if len(compatible_types) == 0:
        print("Error could not find an object which is compatible with", t)
        break
    else:
        print("The compatible types for", t, "are", compatible_types)
        first_object = compatible_types[0]
        first_object_index = objects.index(first_object)
        new_objects.append(objects.pop(first_object_index))

print(new_objects)
