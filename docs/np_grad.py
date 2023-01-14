import numpy as np
np.random.seed(0)


# generte data
B, D = 3, 4

x = np.random.randn(B, D)
y = np.random.randn(B, D)
z = np.random.randn(B, D)
# forward
a = x * y
b = a + z
c = np.sum(b)
# backward
grad_c = 1.0
grad_b = grad_c * np.ones((B, D))
grad_a = grad_b.copy()
grad_z = grad_b.copy()
grad_x = grad_a * y
grad_y = grad_a * x
