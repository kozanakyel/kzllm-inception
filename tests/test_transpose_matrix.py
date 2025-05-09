import pytest



x = [[1,2], [3,4], [5,6]]


def T(x):
    # get transpose of matrix
    shape = len(x), len(x[0])
    new_shape = shape[1], shape[0]
    
    new_matrix = [[0 for _ in range(new_shape[1])] for _ in range(new_shape[0])]
    print(new_matrix)
    
    # assert isinstance(new_matrix, str)
    
    for i in range(shape[0]):
        for j in range(shape[1]):
            new_matrix[j][i] = x[i][j]
            
    print(new_matrix)
    return new_matrix
    
    
def test_T():
    t_matrix = T(x)
    
    