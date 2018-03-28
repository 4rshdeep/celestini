N = 50 # Number of Training examples needed

import re

rf = open("got.txt")
text = rf.read()
clean_text = ''.join(re.findall('[A-Za-z ]', text))
clean_text = re.sub(' +', ' ', clean_text)
clean_text = clean_text.upper()

# print(clean_text[:100])
wf = open("corpus.txt", "w")
wf.write(clean_text)

wf = open("plain_text.txt", "w")
vig = open("vig.txt", "w")
ss = open("simple_sub.txt", "w")

for i in range(50):
	start = 200*i
	end   = (200*(i+1))
	a = clean_text[start : end]
	wf.write(a)
	wf.write("\n")



