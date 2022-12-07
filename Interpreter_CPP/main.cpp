#include <iostream>
#include <list>

using namespace std;

struct Token {
    string type;
    string id;
    int position;
};

void tokenize(string code) {
    int current_position = 0;
    list<Token> tokens;

    while (current_position < code.length()) {
        char letter = code[current_position];

        if (letter == ' ') 
        {   // ignore whitespace
            current_position++;
        }
        else if (letter == '=')
        {
            Token token = Token();
            token.type = "assign";
            token.id = "=";
            token.position = current_position;
            tokens.push_back(token);
        }
        // ...
        else if 
    }
}

int main() {

    string sample_code = """int x = 5;""";

    return 0;
}