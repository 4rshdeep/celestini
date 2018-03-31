# The matrix is assumed to be an m * n matrix (m being the rows)
m = input("enter m: ")
n = input("enter n: ")

lis=[]
for i in range(m):
	print "Enter elements of row ",i,"(space seperated)"
	tmplis=raw_input().split()
	for j in range(n):
		tmplis[j] = int(tmplis[j])
	lis.append(tmplis)
row = m-1
col = 0

#input taken

s = input("enter search value: ")

# we begin at the bottom left element, if it is equal to the search element, then we must return true here,
# else if search element is less that that element then all the elements in that row are more than search element, hence we only need to check the upper matrix
# similarly if search element is more than that element then all the elements in that column above that element are less than the search element, hence we only need to check the right matrix
while(row>=0 and col<n):
	if s==lis[row][col]:
		print "true"
		break
	elif s<lis[row][col]:
		row=row-1
	else:
		col=col+1
else:
	print "false"


# what is the best time complexity
# ans = O(m+n)
# the maximum number of comparisions that would be done are m+n(in the worst case)
# and that would be when we go up then right then up then right ...

# what is the best space complexity
# O(1)
# as the only extra variables we used are row and col