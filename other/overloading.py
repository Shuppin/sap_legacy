"""
Simple module which implements function overloading using wrapper functions.

Worth noting that it does not account for the order of arguments upon 
function a call, it just looks for any match (including subclasses!)
"""
# spent ages trying to implement the lack of ordering as
# a feature, but it sounds like a lack of trying here :(
from inspect import getfullargspec
from logging import Formatter
from logging import FileHandler
from logging import getLogger

if not __name__.startswith("modules."):
    from config import ConfigParser
else:
    from modules.config import ConfigParser
# Might upload as a separate repo on github at some
# point since it's a pretty neat little module

# Load config information
config = ConfigParser()

# Setup logging

# Create formatter object using value(s) from config file
# Formatter defines how each line will look in the config file
formatter = Formatter(config.getstr("logging.format"), datefmt=config.getstr("logging.datefmt"))

# Handler defines which file to write to and how to write to it
handler = FileHandler("logs/" + config.getstr("logging.destination"), mode="a")
handler.setFormatter(formatter)

# Get logging level from config file
log_level = config.getint(f"logging.levels.{config.getstr('logging.level')}")

# Create and setup logging object
logger = getLogger("overloading")
logger.setLevel(log_level)
logger.addHandler(handler)

# Store the value of the logging.levels.ALL level
LOG_ALL = config.getint("logging.levels.ALL")

class create_overload:
    """
    Converts the wrapped function into an overload-able object

    Example syntax:
    ```
    @create_overload
    def add():
        pass

    @add.overload()
    def add(num1: int, num2: int):
        ...
    ```
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
            logger.critical("Overload module could not find function asociated with given args, invalid state")
            if config.getbool("dev.raise_error_stack"):
                raise Exception("Oopsie we shouldn't be here!")
            else:
                print("(Internal) Overload module could not find function asociated with given args, invalid state")
                exit()

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
            logger.log(f"{type(self).__name__}._rearrange_objects(): Returned None due to list size mismatch", level=LOG_ALL)
            return None

        rearranged_objects = []

        # Iterate over all the types
        for t in types:
            # Filter out all the objects which are compatible with the current type
            compatible_types = [obj for obj in objects if issubclass(type(obj), t)]
            # If the length of this is 0, that means there are no compatiple types left
            if len(compatible_types) == 0:
                logger.log(msg=f"{type(self).__name__}._rearrange_objects(): Could not find an object which is compatible with {t}", level=LOG_ALL)
                return None
            else:
                # Gets first object from the compatible types, finds
                # it's position and removes it from the object list
                # and adds it to the new object list 
                logger.log(msg=f"{type(self).__name__}._rearrange_objects(): The compatible types for {t} are {compatible_types}", level=LOG_ALL)
                first_object = compatible_types[0]
                first_object_index = objects.index(first_object)
                rearranged_objects.append(objects.pop(first_object_index))

        return tuple(rearranged_objects)

    def overload(self):
        """
        Example syntax:
        ```
        @create_overload
        def add():
            pass

        @add.overload()
        def add(num1: int, num2: int):
            ...
        ```
        """
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
      