# Short script to neatly print dictionary
# Implmented by the Node class in main

example_dict = {
    1: {
        2: {
            3: 4,
            5: 6
        },
        3: {
            4: 5,
            6: 7
        }
    },
    2: {
        3: {
            4: 5
        },
        4: {
            6: [1,2,3,4]
        }
    }
}

# Recursive function to neatly print a dictionary
def iterdict(d, depth=1):
    text = ""
    if depth == 1:
        text += "{\n"
    for key, value in d.items():
        if isinstance(value, dict):
            text += ("   " * depth + str(key) + ": {\n")
            text += iterdict(value, depth+1)
            text += ("   " * depth + "},\n")
        elif isinstance(value, list):
                text += "   " * depth + str(key) + ": [\n"
                for node in value:
                    text += "   " * (depth + 2) + str(node) + ",\n"
                text += "   " * depth + "],\n"
        else:            
            text += ("   " * (depth+1) + str(key) + ": " + str(value) + ",\n")
    if depth == 1:
        text += "}"

    return text

text = iterdict(example_dict)
print(text)