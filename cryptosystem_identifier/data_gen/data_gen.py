from encrypt import *

# PATH =
f = open("./data/plain_text.txt", "r")
vig = open("./data/train_vig.txt", "w")
ss = open("./data/train_simple_sub.txt", "w")

SS_KEY = "HKJXUBFSONMYTECZWPLDAVIQRG"
VIG_KEY = "KEY"
for line in f.readlines():
	v = vigenere(VIG_KEY, line.strip())
	vig.write(v+"\n")
	s = simple_subs(SS_KEY, line.strip())
	ss.write(s+"\n")
