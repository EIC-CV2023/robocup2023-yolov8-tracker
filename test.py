import numpy as np

l  = np.array(range(12,24)).reshape((2,2,3))
print(l)

t = np.array(((False,True),(True,False)))

print(t)

print(l[t])