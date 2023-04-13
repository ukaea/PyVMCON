The VMCON Algorithm
===================

VMCON aims to solve a nonlinearly constrained problem (general nonlinear programming problem), 
specifically, minimising an objective function subject to several nonlinear constraints. 


The Problem
-----------
Also called the *general non-linear programming problem*, we want to minimise an objective function (figure of merit) :math:`f(\vec{x})`

subject to some non-linear constraints:

.. math:: 
    c_i(\vec{x}) = 0, \quad i = 1,...,k

    c_i(\vec{x}) \geq 0, \quad i = k+1,...,m

where the objective and constraint functions are all :math:`n`-dimensional.

Several facts are worth noting about problem formulation: 

1. To maximise some function :math:`g(\vec{x})` we would minimise the objective :math:`f(\vec{x}) = -g(\vec{x})`.
2. To constrain the solution such that :math:`h(\vec{x}) = a` we would apply the constraint :math:`h(\vec{x}) - a = 0`.
3. To constrain the solution such that :math:`h(\vec{x}) \geq a` we would apply the constraint :math:`h(\vec{x}) - a \geq 0`.
4. To constrain the solution such that :math:`h(\vec{x}) \leq 0` we would apply the constraint :math:`-h(\vec{x}) \geq 0`.
5. To constrain the solution such that :math:`h(\vec{x}) \leq a` we would apply the constraint :math:`a-h(\vec{x}) \geq 0`.

Initialisation of VMCON
-----------------------
VMCON is initialised with:

* The objective function to minimise, :math:`f(\vec{x})`, as described above.
* The constraints :math:`c_i(\vec{x}), i = 1,...,m`, as described above.
* An initial sample point :math:`\vec{x}_0`.
* :math:`\mathbf{B}`: the initial Hessian approximation matrix, usually the identity matrix.
* :math:`\epsilon`: the "user-supplied error tolerance".

It should be noted that :math:`\mathbf{B}` will need to be of dimension :math:`d \times d` where :math:`d = \mathrm{max}(n, m)`.

We also set the iteration number to 1, :math:`j=1`.


The Quadratic Programming Problem
---------------------------------
The Quadratic Programming Probelm (QPP) will also be known as the Quadratic Sub-Problem (QSP) because it forms only a part of the 
VMCON algorithm--with the other half being the Augmented Lagrangian. 

The QPP provides the search direction :math:`\delta_j` which is a vector within which :math:`\vec{x}_j` will lay. 
Solving the QPP also provides the Lagrange multipliers, :math:`\lambda_{j,i}`. 

The quadratic program to be minimised on iteration :math:`j` is:

.. math::
    Q(\delta) = f(\vec{x}_{j-1}) + \delta^T\nabla f(\vec{x}_{j-1}) + \frac{1}{2}\delta^TB\delta

subject to 

.. math::
    \nabla c_i(\vec{x}_{j-1})^T\delta + c_i(\vec{x}_{j-1}) = 0, \quad i=1,...,k

    \nabla c_i(\vec{x}_{j-1})^T\delta + c_i(\vec{x}_{j-1}) \ge 0, \quad i=k+1,...,m


The Convergence Test
--------------------
The convergence test is performed on the :math:`j`'th iteration after the QSP. The convergence test is the sum of two terms:

* The predicted change in magnitude of the objective function.
* The complimentary error; where the complimentary error being 0 would mean that a specific constraint is at equality or the Lagrange multipliers are 0.

This is encapsulated in the equation:

.. math::
    \lvert \nabla f(\vec{x}_{j-1})^T \cdot \delta_j \rvert + \sum^m_{i=1}\lvert \lambda_{j,i} c_i(\vec{x}_{j-1}) \rvert < \epsilon

