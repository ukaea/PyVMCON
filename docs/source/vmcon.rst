The VMCON Algorithm
===================

VMCON aims to solve a nonlinearly constrained problem (general nonlinear programming problem),
specifically, minimising an objective function subject to several nonlinear constraints.


The Problem
-----------
Also called the *general non-linear programming problem*, we want to minimise an objective function (figure of merit), :math:`f(\vec{x})`, subject to some non-linear constraints:

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

The Lagrangian
--------------
VMCON is an augmented Lagrangian solver which means it uses the Lagrange multipliers, :math:`\vec{\lambda}`, of some
solution to characterise the quality of the solution.

Of specific note is the Lagrangian function:

.. math::
    L(\vec{x}, \vec{\lambda}) = f(\vec{x}) - \sum_{i=1}^m \vec{\lambda}_ic_i(\vec{x})

and the derivative of the Lagrangian function, with respect to :math:`\vec{x}`:

.. math::
    \nabla_XL(\vec{x}, \vec{\lambda}) = \nabla f(\vec{x}) - \sum_{i=1}^m \vec{\lambda}_i \nabla c_i(\vec{x})
    :label: lagrangian-derivative

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

The QPP provides the search direction :math:`\delta_j` which is a vector upon which :math:`\vec{x}_j` will lay.
Solving the QPP also provides the Lagrange multipliers, :math:`\lambda_{j}`.

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


The Line Search
---------------
The line search helps to mitigate poor initial conditions. It does this by searching in a line along the 'search direction' :math:`\delta` such that:

.. math::
    \vec{x}_j = \vec{x}_{j-1} + \alpha_j\vec{\delta}_j

:math:`\alpha` is found via the minimisation of:

.. math::
    \phi(\alpha) = f(\vec{x}_j) + \sum_{i=1}^k \vec{\mu}_{j,i}|c_i(\vec{x}_j)| + \sum_{i=k+1}^m \vec{\mu}_{j,i}|min(c_i(0, \vec{x}_j))|


On the :math:`j` th iteration,  :math:`\vec{\mu}_{j,i}` is a 1D vector which contains :math:`i = 1,...,m` elements.
On the first iteration:

.. math::
    \vec{\mu}_1 = |\vec{\lambda}_1|

On subsequent iterations:

.. math::
    \vec{\mu}_j = max[|\vec{\lambda}_0|, \frac{1}{2}(\vec{\mu}_{j-1} + |\vec{\lambda}_j|)]

The line search iterates for a maximum of 10 steps and exits if the chosen value of :math:`\alpha` satisfies either the Armijo condition:

.. math::
    \phi(\alpha) \leq \phi(0) + 0.1\alpha(\phi(1) - \phi(0))

or the so-called Kovari condition, which was an ad-hoc break condition in the PROCESS implementation of VMCON, therefore does not appear in the paper:

.. math::
    \phi(\alpha) > \phi(0)

Once the line search exits, we have found our optimal value and :math:`\alpha_j = \alpha`.

Finally, on each iteration of the line search, we revise :math:`\alpha` using a quadratic approximation:

.. math::
    \alpha = min\left(0.1\alpha, \frac{-\alpha^2}{\phi(\alpha) - \phi(0) - \alpha(\phi(1) - \phi(0))}\right)


The Broyden-Fletcher-Goldfarb-Shanno (BFGS) Quasi-Newton Update
---------------------------------------------------------------
The final stage of an iteration of the VMCON optimiser is to update the Hessian approximation via a BFGS update.

For an unconstrained problem, we use the following differences to update :math:`\mathbf{B}`:

.. math::
    \vec{\xi} = \vec{x}_j - \vec{x}_{j-1}

.. math::
    \vec{\gamma} = \nabla_XL(\vec{x}_j, \vec{\lambda}_j) - \nabla_XL(\vec{x}_{j-1}, \vec{\lambda}_j)

which is calculated using :eq:`lagrangian-derivative`.

Since we have a constrained problem, we define a further quantity:

.. math::
    \vec{\eta} = \theta\vec{\gamma} + (1-\theta)\mathbf{B}\vec{\xi}

where

.. math::
    \theta = \begin{cases}
        1 ,& \text{if } \vec{\xi}^T\vec{\gamma} \geq 0.2\vec{\xi}^T\mathbf{B}\vec{\xi}\\
        \frac{0.8\vec{\xi}^T\mathbf{B}\vec{\xi}}{\vec{\xi}^T\mathbf{B}\vec{\xi} - \vec{\xi}^T\vec{\gamma}},& \text{otherwise}
    \end{cases}


The definition of :math:`\vec{\eta}` ensures :math:`\mathbf{B}` remains positive semi-definite, which is a prereqesite to solving the QSP.

We can then perform the BFGS update:

.. math::
    \mathbf{B_{NEW}} = \mathbf{B} - \frac{B\vec{\xi}\vec{\xi}^TB}{\vec{\xi}^TB\vec{\xi}} + \frac{ \vec{\eta} \vec{\eta}^T}{\vec{\xi}^T\vec{\eta}}


The VMCON Algorithm
-------------------
This page covers the mathematics and theory behind the VMCON algorithm. For completeness, the following flow diagram demonstrates
how the algorithm is implemented at a high level.

.. mermaid::

    flowchart
        setup("Initialisation of VMCON") --> j1("j = 1")
        j1 --> qsp("The Quadratic Programming Problem (Lagrange multipliers and search direction)")
        qsp --> convergence_test(["Convergence criterion met?"])
        convergence_test -- "Yes" --> exit[["Exit"]]
        convergence_test -- "No" --> linesearch("Line search (next evaluation point)")
        linesearch --> bfgs("BFGS update")
        bfgs --> incrementj("j = j + 1")
        incrementj --> qsp
