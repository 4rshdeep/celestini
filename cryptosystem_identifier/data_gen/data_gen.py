from encrypt import *

# PATH =
f = open("./data/train/plain_text_long.txt", "r")
vig = open("./data/train/vig_long.txt", "w")
ss = open("./data/train/simple_sub_long.txt", "w")

SS_KEY = "HKJXUBFSONMYTECZWPLDAVIQRG"
VIG_KEY = "KEY"
for line in f.readlines():
	v = vigenere(VIG_KEY, line.strip())
	vig.write(v+"\n")
	s = simple_subs(SS_KEY, line.strip())
	ss.write(s+"\n")
