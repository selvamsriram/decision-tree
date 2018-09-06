import copy
import math
import numpy as np

#1 = enabled 0 = disabled
debug = 1

class Attribute:
	pass

class Data:
	
	def __init__(self, *, fpath = "", data = None):
		
		if not fpath and data is None:
			raise Exception("Must pass either a path to a data file or a numpy array object")

		self.raw_data, self.attributes, self.index_column_dict, \
		self.column_index_dict = self._load_data(fpath, data)

	def _load_data(self, fpath = "", data = None):
		
		if data is None:
			data = np.loadtxt(fpath, delimiter=',', dtype = str)

		header = data[0]
		index_column_dict = dict(enumerate(header))

		#Python 2.7.x
		# column_index_dict = {v: k for k, v in index_column_dict.items()}

		#Python 3+
		column_index_dict = {v: k for k, v in index_column_dict.items()}

		data = np.delete(data, 0, 0)

		attributes = self._set_attributes_info(index_column_dict, data)

		return data, attributes, index_column_dict, column_index_dict
	
	def _set_attributes_info(self, index_column_dict, data):
		attributes = dict()

		for index in index_column_dict:
			column_name = index_column_dict[index]
			if column_name == 'label':
				continue
			attribute = Attribute()
			attribute.name = column_name
			attribute.index = index - 1
			attribute.possible_vals = np.unique(data[:, index])
			attributes[column_name] = attribute

		return attributes

	def get_attribute_possible_vals(self, attribute_name):
		"""

		Given an attribute name returns the all of the possible values it can take on.
		
		Args:
		    attribute_name (str)
		
		Returns:
		    TYPE: numpy.ndarray
		"""
		return self.attributes[attribute_name].possible_vals

	def get_row_subset(self, attribute_name, attribute_value, data = None):
		"""

		Given an attribute name and attribute value returns a row-wise subset of the data,
		where all of the rows contain the value for the given attribute.
		
		Args:
		    attribute_name (str): 
		    attribute_value (str): 
		    data (numpy.ndarray, optional):
		
		Returns:
		    TYPE: numpy.ndarray
		"""
		if not data:
			data = self.raw_data

		column_index = self.get_column_index(attribute_name)
		new_data = copy.deepcopy(self)
		new_data.raw_data = data[data[:, column_index] == attribute_value]
		return new_data

	def get_column(self, attribute_names, data = None):
		"""

		Given an attribute name returns the corresponding column in the dataset.
		
		Args:
		    attribute_names (str or list)
		    data (numpy.ndarray, optional)
		
		Returns:
		    TYPE: numpy.ndarray
		"""
		if not data:
			data = self.raw_data

		if type(attribute_names) is str:
			column_index = self.get_column_index(attribute_names)
			return data[:, column_index]

		column_indicies = []
		for attribute_name in attribute_names:
			column_indicies.append(self.get_column_index(attribute_name))

		return data[:, column_indicies]


	def get_column_index(self, attribute_name):
		"""

		Given an attribute name returns the integer index that corresponds to it.
		
		Args:
		    attribute_name (str)
		
		Returns:
		    TYPE: int
		"""
		return self.column_index_dict[attribute_name]

	def __len__(self):
		return len(self.raw_data)

def information_gain_per_column (subset_data, col):
    # Compute the entropy of the labels first
    unique_labels = np.unique (subset_data[:, 0])
    if unique_labels.shape[0] == 1:
        label_entropy = 0
    else:
        count = np.zeros ((unique_labels.shape[0], 1))
        for i in range (0, unique_labels.shape[0]):                     # For each unique label
            for j in range (0, subset_data[:, 0].shape[0]):             # For each row in the label column
                if subset_data[j, 0] == unique_labels[i]:
                    count [i] += 1                                      # Count of +ve or -ve

    row_size = subset_data[:, 0].shape[0]

    i = 0
    label_entropy = 0
    for i in range (0, unique_labels.shape[0]):
        division_result = (count[i]/row_size)
        label_entropy -= (division_result) * math.log (division_result, 2)

    print ("Label Entropy : ", label_entropy)
	# See how many different inputs are there in the column
	# For each different type of input in a column, compute the entropy
	# Find the the +ve and -ve outcomes for each different type of input
	# Compute expected entropy
	# Compute Information Gain = H (Label) - Expected entropy (Column)
	# Returns information gain per column
    return subset_data

def information_gain_all_columns (subset_data, attributes):
    # Find how many different columns are there and loop for all of them
    no_of_col = subset_data.shape[1]

    # Create a single dimensional array to save all the information gain computed
    info_gain_per_column = np.zeros ((no_of_col, 1))

    # Call information_gain_per_column (subset_data, col)
    for col in info_gain_per_column:
        information_gain_per_column (subset_data, col)

    # Return the information_gain_per_column array
    return subset_data

def get_next_attribute (subset_data, attributes):
    # Call information_gain_all_columns (subset_data, attributes)
    # information_gain_all_columns (subset_data, attributes)
    # Select the attribute that has the maximum value in information_gain_per_column []
    # Return the Attribute Name and Column index
    information_gain_all_columns (subset_data, attributes)
    return subset_data

def add_node (subset_data, attributes):
    # If the every label is either +ve or -ve then the tree is complete at this branch, add the decision and return """

    # Take the label column and check if everything is same """
    unique_labels = np.unique (mango.raw_data[:, 0]).shape[0]
    if unique_labels == 1:
        # Take action now, and assign this as the decision for this branch
        if debug == 1:
            print ("Labels are the same")
    elif debug == 1:
            print ("unique labels", unique_labels)

    #The tree is not complete yet. Because the labels differ, branching possible """

    #Check which can be the new attribute that is added to the tree by information gain """
    selected_attribute = get_next_attribute (subset_data, attributes)


mango = Data (fpath = "Mango_train.csv")

add_node (mango.raw_data, mango.attributes)


#Create a new node, named by the attribute """
#Delete the selected attribute from the dataset (Delete the column completely) """
#For each different type of input this attribute can take, we must add a branch and leading decision """
#Call add_node (subset, attribute_data) and obtain the child node to be added as decision """
#Add the given child and the link in 2 dimensional array which has columns [Value of the parent, Leading Decision/child tree] """
#Return the newly added node """	

