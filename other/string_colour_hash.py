# Short script to convert an arbitrary sting into a hex colour
# Lazy, hacky way to perform syntax highlighting,
# would be suitible for priting the syntax tree

def string_to_color(s):
    hash = 0
    for c in s:
        hash = ord(c) + ((hash << 5) - hash)
    color = '#'
    for i in range(3):
        value = (hash >> (i * 8)) & 0xFF
        color += '{:02x}'.format(value)
    return color

print(string_to_color('Token'))
print(string_to_color('Token'))
print(string_to_color('token'))
