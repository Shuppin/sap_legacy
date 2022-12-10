from enum import Enum


class Type(Enum):
    NUM = object()
    STR = object()
    OP = object()
    IDENTIFIER = object()
    VARIABLE = object()
    BOOL = object()
    ASSIGN = object()
    COMMENT = object()
    SEMI = object()
    EOF = object()
    NEWLINE = object()

    
class Token():
    def __init__(self, type, id, startPos):
        self.type: Type = type
        self.id: str = id
        self.startPos: int = startPos

    def __str__(self):
        return f"Token[type = {self.type}, id = '{self.id}', startPos = {self.startPos}]"


class Lexer():
    def __init__(self, input: str):
        self.input = input
        #print(*self.lex(), sep="\n")

    def lex(self):
        linescount = 1
        tokens = []
        currentPos = 0
        while currentPos < len(self.input):
            letter = self.input[currentPos]
            if letter == ' ':
                # ignore whitespace
                currentPos += 1
            elif letter in ['+', '-', '/', '*']:
                tokens.append(Token(Type.OP, letter, currentPos))
                currentPos += 1
            elif letter == "=":
                tokens.append(Token(Type.ASSIGN, '=', currentPos))
                currentPos += 1
            elif letter == "#":
                text = ""
                startingPos = currentPos
                while currentPos < len(self.input) and self.input[currentPos] != '\n':
                    text += self.input[currentPos]
                    currentPos += 1
                tokens.append(Token(Type.COMMENT, text, startingPos))
            elif letter.isdigit():
                text = ""
                startingPos = currentPos
                while currentPos < len(self.input) and self.input[currentPos].isdigit():
                    text += self.input[currentPos]
                    currentPos += 1
                tokens.append(Token(Type.NUM, text, startingPos))
            elif letter.isalpha():
                text = ""
                startingPos = currentPos
                while currentPos < len(self.input) and self.input[currentPos].isalnum():
                    text += self.input[currentPos]
                    currentPos += 1
                if text == 'true' or text == 'false':
                    tokens.append(Token(Type.BOOL, text, startingPos))
                elif text == 'var':
                    startingPos = currentPos
                    currentPos += 1
                    var_id = ""
                    while currentPos < len(self.input) and self.input[currentPos].isalnum():
                        var_id += self.input[currentPos]
                        currentPos += 1
                    tokens.append(Token(Type.VARIABLE, var_id, startingPos))
                else:
                    tokens.append(Token(Type.IDENTIFIER, text, startingPos))
            elif letter == '\n':
                tokens.append(Token(Type.NEWLINE, "<newline>", currentPos))
                linescount += 1
                currentPos += 1
            elif letter == ';':
                tokens.append(Token(Type.SEMI, ";", currentPos))
                currentPos += 1

            else:
                print(f"Unkown Literal {repr(letter)} at position {currentPos} on line {linescount}")
                exit()
                
        tokens.append(Token(Type.EOF, "<EOF>", currentPos))
        return tokens


if __name__ == '__main__':
    input_string = """
    # hello world! how are you?
    var myvar1 = 5;
    120 + 54 + myvar1;
    """
    lexer = Lexer(input_string)
    print(*lexer.lex(), sep="\n")