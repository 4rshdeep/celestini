N = 5000 # Number of Training examples needed
import re
# from encrypt import *

# rf = open("got.txt")
# text = rf.read()
# clean_text = ''.join(re.findall('[A-Za-z ]', text))
# clean_text = re.sub(' +', ' ', clean_text)
# clean_text = clean_text.upper()

# print(clean_text[:100])
wf = open("./data/text/corpus.txt", "r")
# wf.write(clean_text)
clean_text = wf.read()

wf = open("./data/train/plain_text_long.txt", "w")


for i in range(N):
	start = 200*i
	end   = (200*(i+1))
	a = clean_text[start : end]
	wf.write(a)
	wf.write("\n")



