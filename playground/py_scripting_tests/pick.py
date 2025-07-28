
def encode(strs):
    compressed = ''
    for string in strs:
        length = len(string)
        compressed += '#'+str(length)+string
    return compressed


def decode(s: str):
    strings = []
    i = 0
    while i < len(s):
        if s[i] == '#':
            j = i+1
            start = j + 1
            end = int(s[j]) + 1
            substring = s[start:j+end]
            print(substring)
            strings.append(substring)
            i = j + end
        else:
            return strings
    return strings


strs = ["we", "say", ":", "yes", "!@#$%^&*()"]

compressed = encode(strs)
print(compressed)

strings = decode(compressed)
