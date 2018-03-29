from encrypt import *

f = open("test_plain.txt", "r")
vig = open("test_vig.txt", "w")
ss = open("test_simple_sub.txt", "w")

SS_KEY = "HKJXUBFSONMYTECZWPLDAVIQRG"
VIG_KEY = "KEY"
for line in f.readlines():
	v = vigenere(VIG_KEY, line.strip())
	vig.write(v+"\n")
	s = simple_subs(SS_KEY, line.strip())
	ss.write(s+"\n")