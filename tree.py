import numpy as np

class TreeNode:
	"""
	Tree used to save the different amplitude of the state
	"""
	def __init__(self, value=0, left=None, right=None):
		"""
		Initialize the tree with the root node

		Args:
			value (float): Amplitude of the corresponding basis state
			left (TreeNode): left child
			right (TreeNode): right child
		"""
		self.value = value
		self.left = left
		self.right = right

def build_sum_tree(arr, start=0, end=None):
	"""
	Add the different amplitude to the tree

	Args:
		arr (np.array): array which contains the amplitudes
		start (int): starting node
		end (int): last node

	Returns:

		TreeNode (TreeNode)
	"""
	if end is None:
		end = len(arr)
	
	if end - start == 1:
		return TreeNode(np.sqrt(arr[start].real**2 + arr[start].imag**2))
	
	mid = (start + end) // 2
	left_subtree = build_sum_tree(arr, start, mid)
	right_subtree = build_sum_tree(arr, mid, end)
	
	return TreeNode(np.sqrt(left_subtree.value.real**2 + left_subtree.value.imag**2 + right_subtree.value.real**2 + right_subtree.value.imag**2), left_subtree, right_subtree)

def explore_tree(node, path="", result_dict=None):
	"""
	Function to explore the tree which contains the ampltiude
	"""
	if result_dict is None:
		result_dict = {}
	
	if node is None:
		return result_dict
	
	# Store the value in the dictionary with the path as key
	result_dict[path] = node.value
	
	# Traverse left subtree, appending '0' to the path
	explore_tree(node.left, path + "0", result_dict)
	
	# Traverse right subtree, appending '1' to the path
	explore_tree(node.right, path + "1", result_dict)
	
	return result_dict

def print_tree(node, level=0):
	"""
	Print the tree
	"""
	if node is not None:
		print_tree(node.right, level + 1)
		print(' ' * 4 * level + '->', node.value)
		print_tree(node.left, level + 1)