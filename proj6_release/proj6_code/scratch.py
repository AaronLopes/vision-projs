import numpy as np

test_box = np.array([[0.,0.,0.],
                         [0.,0.,1.],
                         [0.,2.,0.],
                         [3.,0.,0.],
                         [0.,2.,1.],
                         [3.,0.,1.],
                         [3.,2.,0.],
                         [3.,2.,1.]])

test_hand1 = np.array([1.5,1.,.5])

print(np.amin(test_box, 0))
