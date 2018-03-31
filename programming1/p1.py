# a function that converts a matrix into a new representation where only its non zero elements are stored
# like 3 0 2
#	   0 0 0
#      1 0 0
# would get converted to [[(0,3),(2,2)],[],[(0,1)]]
def mat_to_sparse(mat):
	ans = []
	for i in range(len(mat)):
		ans.append([])
		for j in range(len(mat[0])):
			if mat[i][j]!=0:
				ans[i].append((j,mat[i][j]))
	return ans


#these are the matrices to multiply
n = input("Enter the value of N: ")
print "Input for first",n,"*",n,"matrix"
mat1=[]
for i in range(n):
	print "Enter row ",i," (space seperated values)"
	mat1.append(raw_input().split())
	for j in range(n):
		mat1[-1][j] = int(mat1[-1][j])
print "Input for second",n,"*",n,"matrix"
mat2=[]
for i in range(n):
	print "Enter row ",i," (space seperated values)"
	mat2.append(raw_input().split())
	for j in range(n):
		mat2[-1][j] = int(mat2[-1][j])
mat2org=mat2

#get the transpose of mat2
mat2t=[]
for i in range(len(mat2[0])):
	mat2t.append([])
	for j in range(len(mat2)):
		mat2t[i].append(mat2[j][i])

mat1 = mat_to_sparse(mat1)
mat2 = mat_to_sparse(mat2t)

# print mat1
# print mat2

# Multiplication
mulMat = []
for i in range(len(mat1)):
	mulMat.append([])
	for j in range(len(mat2)):
		a=0
		b=0
		sum=0
		# to get the element at (i,j) we need to multiply each element of row i in mat1 with appropriate element (one which has the same column number) of row j in mat2 and them sum them
		while(a<len(mat1[i]) and b<(len(mat2[j]))):
			if(mat1[i][a][0]==mat2[j][b][0]): # if they share the same column, then we multiply and add the result to sum
				sum=sum+mat1[i][a][1]*mat2[j][b][1]
				a=a+1
				b=b+1
			elif(mat1[i][a][0]<mat2[j][b][0]):# we increment a or b, whichever matrix has the lesser column number
				a=a+1
			else:
				b=b+1
		if sum!=0: # we only store if the value isn't 0
			mulMat[i].append((j,sum))
print "Sparse multiplication: ",mulMat

# Convolution
mat2org = mat_to_sparse(mat2org)
# print mat2org
sum=0
for i in range(len(mat1)):
	a=0
	b=0
	while(a<len(mat1[i]) and b<(len(mat2org[i]))): # for each row we need to multiply the ith elment of mat1 with the ith element of mat2 and add them in the sum variable
		if(mat1[i][a][0]==mat2org[i][b][0]):
			sum=sum+mat1[i][a][1]*mat2org[i][b][1]
			a=a+1
			b=b+1
		elif(mat1[i][a][0]<mat2org[i][b][0]):
			a=a+1
		else:
			b=b+1
print "Convolution: ",[sum]

# what is the time complexity?
# O(N*max(m1,m2))
# what we do is that for every i,j in N*N, we multiply ith row of matrix1 with jth row of matrix2
# so order is N*N*no of elements in each row
# no of elements in each row are roughly m1/N and m2/N respectively
# To calculate the convolution, we went to each nonzero number and multiplied them if they shared the same row and column
# which again had the complexity max(m1,m2)

# what is the space complexity?
# we generated a matrix to store that non zero elements so the space complexity is max(m1,m2)