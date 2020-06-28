
import  numpy as np

x = np.arange(9.).reshape(3, 3)
print(x)

indx = x[np.where((x>3) & (x<6))]


print(indx)