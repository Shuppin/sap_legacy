

my_dict = {
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

def iterdict(d, depth=1):
    text = ""
    if depth == 1:
        text += "{\n"
    for key, value in d.items():
        if isinstance(value, dict):
            text += ("   " * depth + str(key) + ": {\n")
            text += iterdict(value, depth+1)
            text += ("   " * depth + "},\n")
        else:            
            text += ("   " * (depth+1) + str(key) + ": " + str(value) + ",\n")
    if depth == 1:
        text += "}"

    return text

text = iterdict(my_dict)
print(text)