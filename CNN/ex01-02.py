import numpy as np
a = np.array([2, 3, 4])
b = np.array([(1.5, 2, 3),(1.5, 2, 3)])
c = np.arange(15)
d = np.arange(15).reshape(3, 5)

print(a)
print(b)
print(c)
print(d)


e = np.zeros((3, 4))
f = np.ones((2, 3, 4), dtype=np.int16)
g = np.eye(3)
h = np.random.rand(2,3)
i = h.max()
j = h.argmax()
k = a * 2
l = b.sum(axis=0)
m = np.exp(c)

print(e)
print(f)
print(g)
print(h)
print(i)
print(j)
print(k)
print(l)
print(m)