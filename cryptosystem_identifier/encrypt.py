# Letters that we care for
LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

def simple_subs(key, message):
    translated = ''
    charsA = LETTERS
    charsB = key
    # loop through each symbol in the message
    for symbol in message:
        if symbol.upper() in charsA:
            # encrypt/decrypt the symbol
            symIndex = charsA.find(symbol.upper())
            if symbol.isupper():
#                 print(symbol)
                translated += charsB[symIndex].upper()
            else:
                translated += charsB[symIndex].lower()
        else:
            # symbol is not in LETTERS, just add it
            translated += symbol
    return translated

def vigenere(key, message):
    translated = []  # stores the encrypted/decrypted message string
    keyIndex = 0
    key = key.upper()
    for symbol in message:  # loop through each character in message
        num = LETTERS.find(symbol.upper())
        if num != -1:  # -1 means symbol.upper() was not found in LETTERS
            num += LETTERS.find(key[keyIndex])  # add if encrypting
            num += 1 # a should have shift of 1
            num %= len(LETTERS)  # handle the potential wrap-around
            # add the encrypted/decrypted symbol to the end of translated.
            if symbol.isupper():
                translated.append(LETTERS[num])
            elif symbol.islower():
                translated.append(LETTERS[num].lower())
            keyIndex += 1  # move to the next letter in the key
            if keyIndex == len(key):
                keyIndex = 0
        else:
            # The symbol was not in LETTERS, so add it to translated as is.
            translated.append(symbol)
    return ''.join(translated)


# key = "KEY"
# message = "IF YOU WOULD LIKE TO SOLVE AN IMPORTANT SOCIAL PROBLEM WHILE WORKING WITH THE LATEST TECHNOLOGY AND LEARNING FROM INCREDIBLE MENTORS WE HAVE A PROJECT FOR YOU THIS PROJECT CALLED CELESTINI PROJECT HAS"

# # encrypt = subs_cipher(key, message)
# encrypt = vigenere_cipher(key, message)

# print(encrypt)
