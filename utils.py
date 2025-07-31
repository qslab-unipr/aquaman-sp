import numpy as np

#Sorting the 2^n possibilities
def bubble_sort(polars):
	"Order the phase in clockwise verse"
	n=len(polars)
	for i in range(n):
		already_sorted=True
		for j in range(n - i - 1):
			if polars[j][1]>polars[j + 1][1]:
				polars[j], polars[j + 1] = polars[j + 1], polars[j]
				already_sorted=False
		#If there were no swaps during the last iteration,
		#the array is already sorted, and you can terminate
		if already_sorted:
			break
	return polars

def format_matrix2(matrix):
	"""
	Better representation of the matrix
	"""
	M=('\n'.join(''.join('{:21}'.format(np.round(item, 20)) for item in row) for row in matrix))
	return str(M)

def format_matrix(matrix):  
	"""
	Better representation of the matrix
	"""                                               
	M=('\n\n'.join(''.join('{:21}'.format(np.round(item, 20)) for item in row) for row in matrix))
	return print(M)
