import sys
import collections, numpy

# Bag em
# cipher_file = open(sys.argv[1], 'rb')
# cipher_text = cipher_file.read()
text = "QPWKA LVRXC QZIKG RBPFA EOMFL  JMSDZ VDHXC XJYEB IMTRQ WNMEA IZRVK CVKVL XNEIC FZPZC ZZHKM  LVZVZ IZRRQ WDKEC HOSNY XXLSP MYKVQ XJTDC IOMEE XDQVS RXLRL  KZHOV"

# remove all non alpha and whitespace and force uppercase
# SOTHATCIPHERTEXTLOOKSLIKETHIS
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

	print(IC)
# print("%.3f" % IC, "({})".format(IC))


def getFrequencyOfText(inputText):
	d = collections.Counter(inputText)
	d.pop(' ')
	# for k, v in d.items():
		# print(k, v)
	for i in sorted(d):
		print(i)
	return d
    # frequency = {}
    # for letter in inputText:
    #     if letter in frequency:
    #         frequency[letter] += 1
    #     else:
    #         frequency[letter] = 1
    # return frequency

# def get_ndarray(dict):
	# for i in dict

print(getFrequencyOfText(text))
# calculateIC(text)
