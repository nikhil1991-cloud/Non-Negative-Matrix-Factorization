import numpy as np
from NNMF import NNMF_MU
from matplotlib import pyplot as plt

#Lets try a random array A_array of shape 6 by 4 and try to factorize into basis 3 such that W is 6 by 3 and H is 3 by 4
m,n,r=6,4,3
A_array = np.random.rand(m,n)
W,H,Norm = NNMF_MU(A_array,r,1000,initial_W=None,initial_H=None)
#Predicted value of A
A_predicted = np.matmul(W,H)
#Showing the difference

plt.subplot(2,2,1)
plt.imshow(A_array)
plt.colorbar()
plt.xlabel('A_array')

plt.subplot(2,2,2)
plt.imshow(A_predicted)
plt.colorbar()
plt.xlabel('A_predicted')

plt.subplot(2,2,3)
plt.plot(np.log10(Norm))
plt.xlabel('Iterations')
plt.ylabel('Log10(Cost Function)')
