lis=[
"THJU DRPUYY HEX SOL KHEEUPTUE TOFSD NCOE AL HL IUYY DSUR THR SHVU LICPE DSUOP LICPXL DC LDHEEOL BCP DSU TCTUED RUD DSUR JHEECD YCVU DSU THE CP DSUR ICAYX SHVU KUUE SOL BPCT DSU LDHPDOL DSUOP YCVU BCP",
"O QDDXNYYGP XZXJ KPXRZS XZZ SLZFSY GTX RZS GPW ACTSSJQ NTTWI AP XSFGAZWM LX QTADC WNNP VSJM SNR AWHOJ VLX SZZBSJC MZS YJHEMDC TE EMDX BZD QHVJKJ YN QTQRJS STV DJQ UFHXJ GLI BFY DORTCJR STRE YN MQNZIX",
"ANDNJR EMD WFRE YHXJ GP MZO TEQJQPI ALYSWJ VP MZGJ MZYGTSF ET FLNM LSC PADCDSSNMR YN WTRP GX XJDENMR QNCI SJBHY NM EMD QNDWI BLYDWDM DFHO YZNYEFQKJYGP DZCI HD SNE YGP UKLHD ET CTXBFXR XD ALYSWJ OWFMDF",
"D DNF BHWQ VSJQP XGLQK HJ FZMDC GQZYGPWR QFBP IZCPDYJC QTQ L RNXJME XGP YGZZFSY GP BZD FAZZS ET KZXD SNR EJLAJQ HNSS MDC GTE KHYFKWD GP XMLUOPI SSJ FZIRHTNO NE JTT HNKW NMDNREXGP KNWQNHJC SNL LQNYL Z",
"RFKWJQJ YN EMD RTCDBNZI FLYD PILFWDD FMRJQ SFC LQVLDR MJDY F RFQJJ XTWQDY YGTSF NFSPQXY BZD XNCWX DMD SFC HTTYIDO MHX GTE YGP RZEYDC BZD YNZ NLATQEFME KNC MDC YN NTMNJQY MDCXDWK VTYG SNR AWHOJ VSJM EM",
"PD VPWD LQNYJ APSDLYG EMD EWDPX DORTCJ SFWMPI SZ KZNJ GPWXZZ CZ SNE MZGJ SSJ REWDYLSS YN XJDE YGP QZYSHDYDCX HY YGP KHPQC DMD DFHO GKFSSWDVSJM LQK XD REWDYLSS NR XFQDMZWJC T XGZZKO MZGJ DTLGE YGZZRLSC",
"QTNE FMO YGCJD EMNFXZYI GZWRP JCXZQP XZTIVSNBS RDLSR WTQO YXHNM HNKW MZGJ MPFQ EBHNJ XZZQ YZLMJQDWNMGR HTM SNR MFSEQDD FFLNMDY VZWRP TCOX DORTCJ QPUKTJC LSC T MZGJ Z AQZY DNFAD QTQRTSEJM CTNDJ AZQSZS",
"WTQO YXHNM OJEPFSPI GTR NY YGP LQPJM QTQV GTE KZTQDO YN AZQDZD HMDY QNCI SJBHY BDYY SZ MZCWDYMZW GNWYNY YNZP SSJ QFGX QTQO FMO YGP HQZXRCTZOX GP MZD YDY YGZZRLSC XJM TAD DJME BNCI SZ MDWRZY YZWQGLWS",
]


# lis is for vigenere ciphers
# lis2 is for substitution ciphers
answer=[]
ct=0
for i in lis: # change lis to lis2 for substitution ciphers
	ct+=1
	answer.append([])
	countarr=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
	for j in i:
		if j==' ':
			continue
		else:
			countarr[ord(j)-65]=countarr[ord(j)-65]+1
	total = sum(countarr)+0.0
	countarr.sort(reverse=True)

	for kk in range(26):
		answer[-1].append(countarr[kk]/total*100)
print ct	
	# print int(countarr[0]/total*100),
	# print int(countarr[1]/total*100),
	# print int(countarr[2]/total*100),
	# print int(countarr[3]/total*100),
	# print int(countarr[4]/total*100),
	# print int(countarr[5]/total*100),
	# print int(countarr[6]/total*100),
	# print int(countarr[7]/total*100),
	# print int(countarr[8]/total*100),
	# print int(countarr[9]/total*100),
	# print int(countarr[10]/total*100),
	# print int(countarr[11]/total*100),
	# print int(countarr[12]/total*100),
	# print int(countarr[13]/total*100),
	# print int(countarr[14]/total*100),
	# print int(countarr[15]/total*100),
	# print int(countarr[16]/total*100),
	# print int(countarr[17]/total*100),
	# print int(countarr[18]/total*100),
	# print int(countarr[19]/total*100),
	# print int(countarr[20]/total*100),
	# print int(countarr[21]/total*100),
	# print int(countarr[22]/total*100),
	# print int(countarr[23]/total*100),
	# print int(countarr[24]/total*100),
	# print int(countarr[25]/total*100)
print answer

