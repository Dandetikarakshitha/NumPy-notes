'''
*NumPy - Numerical Python
*it is used for datascience,AI&ML,engineering
*Python is most popular language for AI&ML
*easy to read & write
*offers many Libraries
*integrates well with:APIs, databases, webapps
*Numpy is fatster tha puthon
'''
import numpy as np             #Importing numpy alias np

#print(np.__version__)
'''
my_list = [1, 2, 3, 4]
my_list = my_list * 2
print(my_list)
'''

'''
array = np.array([1, 2, 3, 4])
print(array)
print(type(array))
'''

#NumPy arrays
#np.arange()
#np.arange([start, ]stop, [step, ]dtype=None)
'''
array = np.arange(10)    #create an array from 0 to 9
print(array)

arr = np.arange(4, 10)
print(arr)
'''
'''
#create a 1d array
arr = np.array([1, 2, 3, 4, 5])
print(arr)
print(arr[0])

#create 2d array

arr2d = np.array([[1, 2, 3],
                 [4, 5, 6]])

print(arr2d)
'''

#Properties of nd arrays
'''
#1.shape
arr = np.array([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]])

print(arr.shape)

#2.Datatye(dtype)

arr = np.array([1.1, 2.2, 3.3], dtype=np.int32)
print(arr)
print(arr.dtype)

#3.size  the size of ndarray is the total number of elements in the array.it is equal to the product of the dimensional of the array

arr = np.zeros((2, 3, 4))  #generate all zeros in 3d array
print(arr)
print(arr.size)

#4.Number of dimensions(ndim) - the ndim attribute of an ndarray specifies the number of dimensions of axes of the array
arr_4d =np.ones((2, 3, 4, 5))
print(arr_4d)

print(arr_4d.ndim)

#5.itemsize - the itemsize attribute of an ndarray specifies the size of each element in bytes
array = np.array([1, 2, 3, 4])
print(array.dtype)
print(array.itemsize)
'''


#datatypes
#data types are important for controlling memory usage and ensuring data integrity in numerical computations
'''
Common NumPy datatype:
int:integer
float:Floating
bool:boolean(true or false)
complex:complex number with real and imaginary parts
uInt:Unsigned integer(no negative values)
'''

#precison
#impact of precision on memory usage:
'''
array1 = np.array([1.1, 1,2 ,1.3, 1.4], dtype=np.float32)
array2 = np.array([1.1, 1,2 ,1.3, 1.4], dtype=np.float64)
print(array1.itemsize * array1.size, "bytes")  #it take 3 elements for 4bytes
print(array2.itemsize * array2.size, "bytes")  #it take 3 elements for 8bytes
'''

#np.zeros,np.ones,np.full;
''''
#np.zeros - creates an array filled with zeros.It takes the shape of the desired array as input and returns an array of that shape filled with zeros 
#Syntax:  numpy.zeros(shape, dtype=float)
zero_array = np.zeros((2,3))
print(zero_array)

#np.ones - creates an array filled with ones
#Syntax:  numpy.ones(shape, dtype=None)
ones_array = np.ones((2,3))
print(ones_array)

#np.full -  creates an array filled with a specified constant value
#Syntax:  numpy.full(shape, fill_value, dtype=None)
full_array = np.full((2,3), 5)
print(full_array)
'''

#Multidimensional arrays
'''
array = np.array([[['A', 'B', 'C'],['D', 'E', 'F'],['G', 'H', 'I']],
                  [['J', 'K', 'L'],['M', 'N', 'O'],['P', 'Q', 'R']]])
print(array.ndim)
print(array.shape)
print(array[0][0][0])     #chain indexing
print(array[0][1][1])     
print(array[0,0,0])     
word = array[0,0,0]+ array[1,0,1]+array[1,0,1]+array[0,2,2]
print(word)
'''
#Indexing - it refers to accessing individual elements of an array using their position within the array
'''
arr = np.array([1, 2, 3, 4])
print(arr[0])
print(arr[-1])  #negative indexing
'''
#Slicing - it allows you to extract a subset of elements from an array by specifying a range of indices
#array[start:end:step]
'''
arr =np.array([1, 2, 3, 4, 5])
print(arr[1:3])
print(arr[0:4:2])
'''
'''
array = np.array([['1', '2', '3'],
                  ['4', '5', '6'],
                  ['7', '8', '9'],
                  ['10', '11', '12'],
                  ['13', '14', '15']])

print(array[1:4:2])
print(array[::2])
print(array[::-1])
#columns
print(array[:, -1])
print(array[:, 1])

print(array[:, 0:2])
print(array[:, 1:4])
'''

#Scalar arithmetic
'''
array = np.array([1, 2, 3])

print(array + 1)
print(array - 1)
print(array * 2)
print(array / 2)
print(array ** 2)
'''
#Vectorized math functions
'''
#array = np.array([1, 2, 3])
array = np.array([1.001, 2.234, 3.345])

print(np.sqrt(array))
print(np.round(array))
print(np.ceil(array))
print(np.pi)
'''
#Exercise
'''
radii = np.array([1, 2, 3])
print(np.pi * radii **2)
'''

#Arithmetic Operations
'''
array1 = np.array([1, 2, 3])
array2 = np.array([4, 5, 6])
print(array1 + array2)
print(array1 - array2)
print(array1 * array2)
print(array1 / array2)
print(array1 ** array2)
'''

#Comparison / Relational operations:

#marks = np.array([99, 51, 23, 100, 73])
'''
print(marks == 100)
print(marks >= 100)
print(marks <= 100)
print(marks > 100)
print(marks < 100)
print(marks != 100)
'''
#marks[marks < 50] = 0
#print(marks)


#Broadcasting - it allows numpy to performs operations on arrays with different shapes by virtually expanding dimensions
#   so they match the larger arrays shape

#the dimensions have the same size (or) one of the dimension has a size of 1
'''
array1 = np.array([[1, 2, 3, 4]])
array3 = np.array([[1, 2, 3, 4],[5, 6, 7, 8],[9,10, 11, 12],[13, 14, 15, 16]])
array2 = np.array([[1], [2], [3], [4]])
print(array1.shape)
print(array2.shape)
print(array3.shape)

print(array1 * array2)
print(array3 * array2)
'''



#AGGREGATE Functions - summarize data and typically return a singlevalue
'''
array = np.array([[1, 2, 3, 4,5],
                 [6, 7, 8, 9, 10]])

print(np.sum(array))      #sum
print(np.mean(array))     #mean
print(np.std(array))      #standard deviation
print(np.var(array))      #var
print(np.min(array))      #minimum
print(np.max(array))      #maximum
print(np.argmin(array))    #argumentminimum
print(np.argmax(array))    #argument maximum


print(np.sum(array, axis=0))  #sum of columns
print(np.sum(array, axis=1))  #sum of rows
'''


#FILTERING - refers to thr process of selecting elements from an array that match a given condition

#ages = np.array([[21, 17, 19, 20, 16, 30, 18, 65],
#                [39, 22, 15, 99, 18, 19, 20, 21]])
'''
teenagers = ages[ages < 18]
#adults = ages[ages >= 18]
#adults = ages[ages >= 18 & (ages < 65)]
adults = ages[ages >= 18 | (ages < 65)]
seniors = ages[ages >= 65]
evens = ages[ages % 2 ==0]
odds = ages[ages % 2 !=0]

print(teenagers)
print(adults)
print(seniors)
print(evens)
print(odds)
'''
#adults = np.where(ages >=18, ages, -1) # we can replace the 0 with -1 or np.NaN
#print(adults)


#RANDOM NUMBERS GENERATOR
'''
rng = np.random.default_rng()
#print(rng.integers(1, 7))
#print(rng.integers(low=1, high=101))
#print(rng.integers(low=1, high=101, size=3))
#print(rng.integers(low=1, high=101, size=(3, 2)))
'''
'''
np.random.seed(seed=1)
print(np.random.uniform(low=-1, high=1, size=3))
'''
'''
rng = np.random.default_rng()

array = np.array([1, 2, 3, 4, 5])
rng.shuffle(array)
print(array)
'''

#Mass level indexing and slicing

#Boolean indexing - allows you to select elements from an array based on a condition,You create a boolean mask indicating which elements satisfy the conditions , and then use the mask to extract the desired values
'''
arr1D = np.array([1, 2, 3, 4, 5])
mask = arr1D > 3
selected_elements = arr1D[mask]
print(selected_elements)

arr2D = np.array([[1, 2, 3], [4, 5, 6]])
mask = arr2D >= 5
selected_elements = arr2D[mask]
print(selected_elements)
'''

#Fancy Indexing -  allows you to select from an array using of indices.
'''
arr = np.array([1, 2, 3, 4, 5])
indices= [0, 2, 4]
selected_elements = arr[indices]
print(selected_elements)
'''

#Transposing arrays - exchanging its rows and columns.you can transpose an array using the T or the transpose() function
'''
arr2D = np.array([[1, 2, 3], [4, 5, 6]])
print(arr2D.shape)
transpose_arr  = arr2D.T
print(transpose_arr.shape)
print(transpose_arr)

#Swapping  Axes - rearranging the dimensions of an array.you can swap axes using swapaxes() function

arr2D = np.array([[1, 2, 3], [4, 5, 6]])
swapped_arr = arr2D.swapaxes(0, 1)
print(swapped_arr)

arr3D = np.array([[[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12]],
                  
                 [[13, 14, 15, 16],
                  [17, 18, 19, 20],
                  [21, 22, 23, 24]]])

swapped_arr = np.swapaxes(arr3D, 0, 2)         
print(swapped_arr )
'''


#Matrix Multiplication
'''
matrixa = np.array([[1, 2], [3,4]])
matrixb = np.array([[5, 6], [7,8]])
result = np.matmul(matrixa, matrixb)
print(result)
'''

#Reshaping(np.reshape()):
'''
arr = np.arange(1, 10)
reshaped_arr = arr.reshape([3, 3])
print(reshaped_arr)
'''

#Trignometric functions
'''
angles = np.array([0, np.pi, np.pi/2, 3*np.pi/4, np.pi])
result_sin = np.sin(angles)
print("sine:",result_sin)

result_cos = np.cos(angles)
print("cosine:",result_cos)

result_tan = np.tan(angles)
print("tan:",result_tan)
'''


#ravel - returns a flattened one-dimensional array
'''
arr = np.arange(1,13)
reshape = arr.reshape([3, 4])
print(reshape)
flat = reshape.ravel()
print(flat)
'''
#flatten - similar to ravel but flatten returns acopy instead of a view of the original data, thus not affecting the original array
'''
flat_copy = reshape.flatten()
print(flat_copy)
'''
#Difference between ravel and flatten
'''
a = np.array([[1, 2], [3,4]])
b = a.ravel()
b[0] = 100

c = a.flatten()
c[1] = 200

print("Original array after modifying raveled array:",a)
print("Original array after modifying raveled array does not change:",a)
'''

#Squeeze: it is used to remove axes of length one from an array
'''
c = np.array([[[1, 2, 3, 4]]])
print(c.shape)

squeezed = c.squeeze()
print(squeezed.shape)
'''

#Splitting and Joining
'''
#1.np.split
x = np.arange(9)
print(x)
x_split = np.split(x, 3)
print(x_split)
'''
#2.np.array.split: splitting into unequal subarrays
'''
x_array_split = np.array_split(x, 4)
print(x_array_split)
'''
#np.hsplit and np.vsplt
'''
y = np.array([[1, 2, 3], [4, 5, 6]])
res = np.hsplit(y, 3)
resu = np.vsplit(y, 2)
print(res)
print(resu)
'''