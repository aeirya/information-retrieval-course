<!-- here is the error function and its gradient: -->

## Gradient
MSE: 

$\text{E} = \frac{1}{2m} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2$

Gradient with respect to weights:

$\nabla_{\text{}}E = -\frac{1}{m} \sum_{i=1}^{m} [(y_i - \hat{y}_i) \cdot \nabla y_i]$

Note that $y_{i,j} = y(q_i, p_j) = q_i \cdot p_j $

$\nabla_{p} E(i, j) = -2q_i (R_{i,j} - q_i\cdot p_j^T)$

Also adding regularization term.
