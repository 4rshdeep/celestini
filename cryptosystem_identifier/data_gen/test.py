def getFrequencyOfText(inputText):
    frequency = {}
    for letter in inputText:
        if letter in frequency:
            frequency[letter] += 1
        else:
            frequency[letter] = 1

    return frequency


a = input("Enter Text ")
a = getFrequencyOfText(a)
a.pop(' ', 1)
print(a)
b = sorted(list(a.values()) , reverse=True)
b.extend([0] * 26)
print(b)
