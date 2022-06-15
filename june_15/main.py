import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

my_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]],
                         dtype=torch.float32,
                         device=device,
                         requires_grad=True)

# print(my_tensor)
# print(my_tensor.dtype)
# print(my_tensor.device)
# print(my_tensor.shape)
# print(my_tensor.requires_grad)

x = torch.empty(size=(3, 3))

# tensor with 0 as values
y = torch.zeros((3, 3))

# tensor with random values
z = torch.rand((3, 3))

# tensor with all values as 1
a = torch.ones((3, 3))

# tensor as identity matrix
b = torch.eye(4, 4)

# tensor created just like python range function
c = torch.arange(start=0, end=5, step=1)

#
d = torch.linspace(start=0.1, end=1, steps=10)

# empty tensor with given mean and standard deviation
e = torch.empty(size=(1, 5)).normal_(mean=0, std=1)

# same as torch.rand but with lower and upper bounds
f = torch.empty(size=(1, 5)).uniform_(0, 1)

# tensor as a diagonal matrix
g = torch.diag(torch.ones(3))

h = torch.arange(4)

# type conversion to bool type
print(h.bool())

# convert to int16 type
print(h.short())

# convert to int64 type
print(h.long())

# convert to float16
print(h.half())

# convert to float32
print(h.float())

# convert to float64
print(h.double())
