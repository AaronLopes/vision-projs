import torch
from proj0_code import student_code

X = torch.rand([591, 910]) * 200
val = 200

print(torch.le(X, val) * 1)


# g_t = student_code.vector_transpose(g)
# b_t = student_code.vector_transpose(b)
