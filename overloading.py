"""
Simple module which implements function overloading using wrapper functions.

Worth noting that it does not account for the order of arguments upon 
function a call, it just looks for any match (including subclasses!)
"""
# spent ages trying to implement the lack of ordering as
# a feature, but it sounds like a lack of trying here :(
from inspect import getfullargspec

# Yes I know the copious  amounts of printing is horrible, i'll fix it later

# Might upload as a separate repo on github at some
# point since it's a pretty neat little module

class create_overload:
    """
    Converts the wrapped function into an overload-able object
    """
    def __init__(self, _):
        self.cases: dict[tuple] = {}

    def __call__(self, *arguments):

        # Initialising sorted_values just in case self.cases is empty
        sorted_values = None

        for existing_arguments in self.cases.keys():
            # Attempt to rearrange args so that it matches the order of the existing arguments
            sorted_values = self._rearrange_objects(list(arguments), list(existing_arguments))
            # If it found a match, break out of the loop and continue
            if sorted_values is not None:
                break
        
        # If the loop completes and sorted values is still no, we found no matches
        if sorted_values is None:
            return None

        # Get the function associated with the arguments
        function = self.cases.get(existing_arguments)

        # If the function is none then existing_arguments shouldn't be in self.cases
        if function is None:
            raise Exception("Oopsie we shouldn't be here!")

        # Call the function and return the value of it
        return function(*sorted_values)

    def _rearrange_objects(self, objects: list, types: list) -> tuple | None:
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
        # Ensure that the objects and types are of the same size
        if len(objects) != len(types):
            print("Returned None due to list size mismatch")
            return None

        rearranged_objects = []

        # Iterate over all the types
        for t in types:
            # Filter out all the objects which are compatible with the current type
            compatible_types = [obj for obj in objects if issubclass(type(obj), t)]
            # If the length of this is 0, that means there are no compatiple types left
            if len(compatible_types) == 0:
                print("Could not find an object which is compatible with", t)
                return None
            else:
                # Gets first object from the compatible types, finds
                # it's position and removes it from the object list
                # and adds it to the new object list 
                print("The compatible types for", t, "are", compatible_types)
                first_object = compatible_types[0]
                first_object_index = objects.index(first_object)
                rearranged_objects.append(objects.pop(first_object_index))

        return tuple(rearranged_objects)

    def overload(self):
        def store_function(function):
            # getfullargspec retrieves various information about a function
            # We only care about the type annotations on the arguments
            type_annotations = getfullargspec(function).annotations

            # Remove the return type since we don't need it
            if type_annotations.get("return"):
                del type_annotations["return"]

            # If the these specific annotations already exist
            # then there are duplicate overloads, which aren't
            # allowed
            if self.cases.get(tuple(type_annotations.values())):
                raise Exception("Function " + repr(function.__name__) + " has duplicate overloads!")

            # Store these annotations with the function they're from
            self.cases[tuple(type_annotations.values())] = function

            return self

        return store_function
      