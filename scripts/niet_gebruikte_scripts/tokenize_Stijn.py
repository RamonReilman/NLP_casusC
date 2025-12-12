import re
import sys

class Tokens:
    def __init__(self, token_dict={}):
        self.token_dict = token_dict
        self.token_counts = {}
        self.n_tokens = 0

    def generate_tokens(self, input_string):
        """
        generates self variables based on input_string
        :param input_string: string with text to generate tokens from
        """
        # set self values to default
        self.token_dict = {}
        self.token_counts = {}
        self.n_tokens = 0

        # characters to skip
        between_chars = [" ", ".", ",", "!", "?", "\n", "\t"]

        # put all characters in the token dict
        for char in input_string:
            if char in between_chars:
                continue
            if char in self.token_dict.values():
                self.token_counts[char] += 1
            else:
                self.token_dict[self.n_tokens] = char
                self.token_counts[char] = 1
                self.n_tokens += 1

        paired_tokens = []
        paired_token_counts = []

        while True:
            # NOTE: the double for-loop generates all possible token combinations, making len(paired_tokens) increase exponentially.
            # This should be changed with something that checks the string itself for possible combinations, not previously found token
            for i in self.token_dict.keys():
                for j in self.token_dict.keys():
                    token_byte_pair = self.token_dict[i] + self.token_dict[j]
                    if token_byte_pair in paired_tokens or token_byte_pair in self.token_dict.values():
                        continue
                    paired_tokens.append(token_byte_pair)
                    paired_token_counts.append(len(re.findall(token_byte_pair, input_string)))

            # break if there are no new token combinations in the string
            if max(paired_token_counts) < 1:
                break

            highest_index = paired_token_counts.index(max(paired_token_counts))

            # add new token combination to the dictionaries
            self.token_counts[paired_tokens[highest_index]] = max(paired_token_counts)
            self.token_dict[self.n_tokens] = paired_tokens[highest_index]
            self.n_tokens += 1

            # remove added tokens from the list
            paired_tokens.pop(highest_index)
            paired_token_counts.pop(highest_index)

    def tokens_to_string(self, token_list):
        """
        translates tokens of this class to string
        :param token_list: list of integers of the tokens to be translated
        :return: string of the translated tokens
        """
        output = ""
        for token in token_list:
            output += self.token_dict[token]
        return output

    def string_to_tokens(self, input_string):
        """
        generates token list based on this class' tokens from given string
        :param input_string: string to be converted to tokens
        :return: list of integers holding the tokens
        """
        remaining_characters = input_string
        current_token = len(self.token_dict.values()) - 1
        used_locations = []
        token_output = []
        while remaining_characters:
            # regex to check if only in-between characters remain in the string
            if not bool(re.compile(r'[^\n\t .,!?]').search(remaining_characters)):
                break
            found = remaining_characters.find(self.token_dict[current_token])
            if found == -1:
                current_token -= 1
            else:
                # append the found token to the right location and replace the found token in the string with blank space
                location = 0
                while True:
                    if location == len(used_locations) or found < used_locations[location]:
                        used_locations.insert(location, found)
                        token_output.insert(location, current_token)
                        remaining_characters = (remaining_characters[:found] + " " * len(self.token_dict[current_token]) +
                                                remaining_characters[found + len(self.token_dict[current_token]):])
                        break
                    location += 1
        return token_output



    def __str__(self):
        return f"tokens: {self.token_dict}, token counts: {self.token_counts}, number of tokens: {self.n_tokens}"



def read_txt(file):
    output_string = ""
    with open(file, "r") as txt_file:
        for line in txt_file:
            output_string += line
        txt_file.close()
    return output_string

def read_enc(file):
    enc_string = []
    with open(file, "r") as enc_file:
        for line in enc_file:
            encoded_line = line.split(",")
            for code in encoded_line:
                enc_string.append(code)
        enc_file.close()
    return enc_string

def read_tok(file):
    tokens = {}
    with open(file, "r") as tok_file:
        for line in tok_file:
            code, token = line.split(",")
            tokens[token] = code
        tok_file.close()
    return tokens

def write_txt(string, file):
    with open(file, "w") as txt_file:
        txt_file.write(string)
        txt_file.close()

def write_tok(token_dict, file):
    with open(file, "w") as tok_file:
        for key in token_dict.keys():
            tok_file.write(key + "," + token_dict[key])
        tok_file.close()

def write_enc(tokens, file):
    with open(file) as enc_file:
        for token in tokens:
            enc_file.write(token + ",")
        enc_file.close()



def main():
    input1, input2, output = sys.argv[2], sys.argv[3], sys.argv[4]
    if input1.endswith(".txt") and input2 == "None":
        string_input = read_txt(input1)
        tokeniser = Tokens()
        tokeniser.generate_tokens(string_input)
        
    elif input1.endswith(".txt") and input2.endswith(".enc"):

    elif input1.endswith(".tok") and input2.endswith(".enc"):
        tokens = read_tok(input1)
        encoded = read_enc(input2)
        tokeniser = Tokens(tokens)
        string_output = tokeniser.tokens_to_string(encoded)
        write_txt(string_output, output)
    input_string = "Three of the three pigs were pigs. The big bad wolf wanted all three up in trees"
    tokeniser = Tokens()
    tokeniser.generate_tokens(input_string)
    print(tokeniser)
    print(tokeniser.string_to_tokens("Threre of thre"))



if __name__ == "__main__":
    main()
