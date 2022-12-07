class Solution:
    def calculate(self, s: str) -> int:
        tokens = []

        currentPos = 0
        while currentPos < len(s):
            letter = s[currentPos]
            if letter == ' ':
                # ignore whitespace
                currentPos += 1
            elif letter.isdigit():
                text = ""
                while currentPos < len(s) and s[currentPos].isdigit():
                    text += s[currentPos]
                    currentPos += 1
                tokens.append(int(text))
            elif letter in ['+', '-', '(', ')']:
                tokens.append(letter)
                currentPos += 1

        return self.parse_expression(tokens)

    def parse_expression(self, tokens):

        if tokens[0] in ['-','+']:
            tokens.insert(0,0)

        i = 0
        while '(' in tokens:
            if tokens[i] == '(':
                start = i
                open_count = 1
                close_count = 0
                i += 1
                while open_count != close_count:
                    if tokens[i] == '(':
                        open_count += 1
                    if tokens[i] == ')':
                        close_count += 1
                    i += 1
                sub_tokens = tokens[start+1:i-1]
                value = self.parse_expression(sub_tokens)
                del tokens[start:i]
                i -= i-start
                tokens.insert(start, value)
            else:
                i += 1

        i = 0
        while i < len(tokens):
            if tokens[i] == '+':
                tokens[i] = tokens[i-1] + tokens[i+1]
                del tokens[i-1]
                del tokens[i]
            elif tokens[i] == '-':
                tokens[i] = tokens[i-1] - tokens[i+1]
                del tokens[i-1]
                del tokens[i]
            else:
                i += 1
        
        return tokens[0]


x = Solution()
print(x.calculate('1+1'))