expression = "(2+(1-1))+(3+3)"
tokens = []

currentPos = 0
while currentPos < len(expression):
    letter = expression[currentPos]
    if letter == ' ':
        # ignore whitespace
        currentPos += 1
    elif letter.isdigit():
        text = ""
        startingPos = currentPos
        while currentPos < len(expression) and expression[currentPos].isdigit():
            text += expression[currentPos]
            currentPos += 1
        tokens.append(int(text))
    elif letter in ['+', '-', '(', ')']:
        tokens.append(letter)
        currentPos += 1


def parse_expression(tokens, whole=False):

    if tokens[0] in ['-','+']:
        print("Inserted leading 0")
        tokens.insert(0,0)

    print("Evaluating:", tokens)

    i = 0
    while '(' in tokens:
        print(i, tokens[i])
        if tokens[i] == '(':
            print("Found bracket!")
            start = i
            open_count = 1
            close_count = 0
            i += 1
            print("Seeking close bracket")
            while open_count != close_count:
                #print(i, tokens[i], open_count, close_count)
                print("\t", tokens[i])
                if tokens[i] == '(':
                    open_count += 1
                if tokens[i] == ')':
                    close_count += 1
                i += 1
            print("Found!")
            sub_tokens = tokens[start+1:i-1]
            value = parse_expression(sub_tokens)
            print("Evaluated value:", value)
            del tokens[start:i]
            i -= i-start
            tokens.insert(start, value)
        else:
            i += 1

    print("Bracketless expression:", tokens)
    i = 0
    while i < len(tokens):
        if tokens[i] == '+':
            print(tokens[i-1], "+", tokens[i+1], "=", end=" ")
            print(tokens[i-1] + tokens[i+1])
            tokens[i] = tokens[i-1] + tokens[i+1]
            del tokens[i-1]
            del tokens[i]
        elif tokens[i] == '-':
            print(tokens[i-1], "-", tokens[i+1], "=", end=" ")
            print(tokens[i-1] - tokens[i+1])
            tokens[i] = tokens[i-1] - tokens[i+1]
            del tokens[i-1]
            del tokens[i]
        else:
            i += 1

    if whole:
        return tokens
    else:
        return tokens[0]
    

x = parse_expression(tokens)
print(x)
