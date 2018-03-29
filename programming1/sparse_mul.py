# a function that converts a matrix into a new representation where only its non zero elements are stored
def mat_to_sparse(mat):
	ans = []
	for i in range(len(mat)):
		ans.append([])
		for j in range(len(mat[0])):
			if mat[i][j]!=0:
				ans[i].append((j,mat[i][j]))
	return ans


#these are the matrices to multiply
mat1 = [[1,0,2],[2,1,2]]
mat2 = [[1,1],[0,2],[3,1]]

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

mulMat = []
for i in range(len(mat1)):
	mulMat.append([])
	for j in range(len(mat2)):
		a=0
		b=0
		sum=0
		while(a<len(mat1[i]) and b<(len(mat2[j]))):
			if(mat1[i][a][0]==mat2[j][b][0]):
				sum=sum+mat1[i][a][1]*mat2[j][b][1]
				a=a+1
				b=b+1
			elif(mat1[i][a][0]<mat2[j][b][0]):
				a=a+1
			else:
				b=b+1
		if sum!=0:
			mulMat[i].append((j,sum))
print mulMat

# what is the time complexity?
# O(N*max(m1,m2))
# what we do is that for every i,j in N*N, we multiply ith row of matrix1 with jth row of matrix2
# so order is N*N*no of elements in each row
# no of elements in each row are roughly m1/N and m2/N respectively

# what is the space complexity?
# we generated a matrix to store that non zero elements so the space complexity is max(m1,m2)