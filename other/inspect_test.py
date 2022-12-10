#python3
from inspect import currentframe, getframeinfo

print("Calling eat() from line", getframeinfo(currentframe()).lineno)