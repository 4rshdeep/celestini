import sys
import collections, numpy

def calculateIC(cipher_text):
	cipher_flat = "".join(
		[x.upper() for x in cipher_text.split()
		if x.isalpha()]
	)
	# Tag em
	N = len(cipher_flat)
	freqs = collections.Counter(cipher_flat)
	alphabet = map(chr, range(ord('A'), ord('Z')+1))
	freqsum = 0.0
	# Do the math
	for letter in alphabet:
		freqsum += freqs[letter] * (freqs[letter] - 1)
	IC = freqsum / (N*(N-1))
	return IC

englishExpectedFrequencies = {
    'a': 0.08167,
    'b': 0.01492,
    'c': 0.02782,
    'd': 0.04253,
    'e': 0.12702,
    'f': 0.02228,
    'g': 0.02015,
    'h': 0.06094,
    'i': 0.06966,
    'j': 0.00153,
    'k': 0.00772,
    'l': 0.04025,
    'm': 0.02406,
    'n': 0.06749,
    'o': 0.07507,
    'p': 0.01929,
    'q': 0.00095,
    'r': 0.05987,
    's': 0.06327,
    't': 0.09056,
    'u': 0.02758,
    'v': 0.00978,
    'w': 0.02361,
    'x': 0.0015,
    'y': 0.01974,
    'z': 0.00074
}

import numpy as np

def getFrequencyOfText(inputText):
	d = collections.Counter(inputText)
	d.pop(' ', -1) # remove space
	b = sorted(list(d.values()), reverse=True)
	n = float(sum(b))
	b.extend([0] * 26)
	b = np.array(b, dtype=float)
	b = np.divide(b, n)
	return b
	# print(b)
	# print(b)
    # frequency = {}
    # for letter in inputText:
    #     if letter in frequency:
    #         frequency[letter] += 1
    #     else:
    #         frequency[letter] = 1
    # return frequency

# def get_ndarray(dict):
	# for i in dict
def frequency_index(text):
	eng = np.array(sorted(englishExpectedFrequencies.values()))
	freq = getFrequencyOfText(text)
	result = 0
	for i, val in zip(freq, eng):
# 		if i == 0:
# 			continue
		result += (val-i)
	return result

# result = 0
# for i, val in zip(freq, eng):
# 	if i == 0:
# 		continue
# 	result += (val-i)
# a = input("text ")
# print(frequency_index(a))
