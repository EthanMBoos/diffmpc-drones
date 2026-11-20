# How DiffMPC works

Most control loops on a robot are PID. A PID controller looks at the error
between where a quantity is and where it should be, and pushes against that
error, its running sum, and its rate of change. It needs no model of the robot,
it has three gains to tune, and it works well for a large share of problems
(e.g., holding a motor speed or a drone's body rates).

PID has limits that show up when the motion gets more dynamic. It reacts to an
error only after the error exists, so it cannot prepare for something that is
about to happen (e.g., a turn ahead on the path). It handles one quantity per
loop, while the quantities of a real vehicle are coupled. It also has no notion
of what the vehicle can physically do, which matters most near the edge of that
ability (e.g., a car whose tires are sliding).

Model predictive control (MPC) is what people reach for in those cases. An MPC
controller holds a model of the vehicle. At every control step it uses the model
to plan the next short stretch of motion, picks the plan with the lowest cost,
applies the first control of that plan, and then plans again. Because it plans
with a model, it can anticipate, it can trade off all of the vehicle's
quantities together, and it can work with the vehicle's physics where PID can
only react to them.

MPC is harder to use than PID in three ways. It needs a model, and it is only as
good as that model. It has to solve an optimization problem at every control
step, fast enough to keep up with the vehicle. And it has many more numbers to
choose: a weight for every term of the cost, and the mass and other physical
constants in the model. These numbers interact, they are usually tuned by an
expert through repeated trials, and a controller tuned for one condition can
fail in another.

DiffMPC ([paper](https://arxiv.org/abs/2510.06179),
[code](https://github.com/ToyotaResearchInstitute/diffmpc)), from Toyota
Research Institute, addresses the third difficulty. It replaces hand tuning with
gradient ascent: it computes how the controller's output changes when those
numbers change, so the numbers can be adjusted in the direction that raises a
reward. Doing this over many simulated runs is expensive, so the paper's main
technical work is a solver that runs many of these problems at once on a GPU.

The sections below explain the paper's method step by step and then point to
where it lives in the code. Statements about this repository's example were
measured on it; statements about the car come from the paper.

## Three nested loops

Everything in the paper belongs to one of three loops, each inside the next.

```
 TRAINING LOOP        one pass = one update of θ           (this repo: 20)
 │
 │   FLIGHT LOOP      one pass = one control step          (this repo: 10)
 │   │
 │   │   SOLVER LOOP  one pass = one improvement of a plan (this repo: up to 50)
 │   │   │
 │   │   │   build a simpler problem around the current plan
 │   │   │   solve one linear system              <- the paper's GPU work
 │   │   │   step toward the answer
 │   │   │
 │   │   apply the first control of the plan to the simulated drone
 │   │   add this step's reward
 │   │
 │   gradient of the total reward with respect to θ
 │   θ <- θ + rate × gradient
```

The parameters θ are whatever is being tuned. In this repository θ is 30 cost
weights. In the paper's car experiment θ is 14 numbers: nine cost weights and
five physical parameters of the tires and wheels.

The paper's contribution sits in the innermost loop. It solves the linear
system in a way that suits a GPU, and it reuses that same solve to get the
gradient. The outer two loops are standard.

## The problem the solver is given

At each control step the solver looks for the best plan over a short horizon. A
plan is a sequence of states and controls,
$z = (x_0, u_0, x_1, u_1, \dots, x_T)$, and the best plan minimizes a cost while
obeying the model:

```math
\min_{z}\ \sum_{t=0}^{T} c^{x}_t(x_t) + \sum_{t=0}^{T-1} c^{u}_t(u_t)
\quad \text{subject to} \quad
f_t(x_{t+1}, x_t, u_t) = 0, \qquad x_0 = x_s .
```

The cost $c$ says what a good plan is (e.g., close to the goal, with little
control effort). The constraint $f_t = 0$ says that the plan must be physically
possible: the state at step $t+1$ must be the one the model predicts from the
state and control at step $t$. The last condition pins the first state of the
plan to the drone's current state $x_s$. The parameters θ can appear in the
cost, in the model, or in both.

Only the first control $u_0$ is applied. At the next control step the drone has
moved, and the solver plans again from the new state.

In this repository a state is 13 numbers (position, velocity, an attitude
quaternion, and body rates) and a control is 4 numbers (a thrust and three
torques). The plan covers 20 states at 0.03 s apart, which is 0.57 s. A plan is
therefore about 340 numbers, tied together by 260 constraint equations.

## The question training asks

Training needs the gradient of a reward with respect to θ, and the reward
depends on θ only through the solver's answer. The quantity to find is therefore
how the best plan changes when θ changes.

There are two ways to get it. The first is to differentiate through every
operation the solver performed, iteration by iteration. This is expensive, and
its cost grows with the number of iterations. The second uses the fact that the
answer satisfies an equation, and differentiates that equation. This is called
implicit differentiation, and its cost does not depend on how the answer was
found.

A problem small enough to do by hand shows the idea. Take one variable and two
parameters:

```math
z^\star = \arg\min_z \ \tfrac{1}{2} a z^2 - b z .
```

At the minimum the derivative with respect to $z$ is zero, so $z^\star$
satisfies the condition $F(z, a, b) = a z - b = 0$. Solving gives
$z^\star = b/a$, and differentiating that directly gives
$\partial z^\star / \partial a = -b/a^2$.

Implicit differentiation reaches the same result without the formula for
$z^\star$. The condition $F = 0$ holds for every value of $a$, so its total
derivative with respect to $a$ is zero:

```math
\frac{\partial F}{\partial z}\frac{\partial z^\star}{\partial a}
+ \frac{\partial F}{\partial a} = 0
\quad\Longrightarrow\quad
a \,\frac{\partial z^\star}{\partial a} + z^\star = 0
\quad\Longrightarrow\quad
\frac{\partial z^\star}{\partial a} = -\frac{z^\star}{a} = -\frac{b}{a^2} .
```

Only two things were needed: the answer $z^\star$ itself, and the derivatives of
the condition $F$ at that answer. The control problem is handled the same way,
with a matrix in place of the number $a$.

One assumption carries over as well. The derivation used $F = 0$, which is true
only at the minimum. If the solver stops before reaching the minimum, the result
is an approximation.

## The gradient of a constrained solution

The control problem has constraints, so its condition for a minimum has two
parts. Write the total cost of a plan as $c(z, \theta)$ and stack all of the
constraint equations into $g(z, \theta) = 0$. Introduce one multiplier per
constraint equation, collected in $\lambda$, and combine cost and constraints
into one function, the Lagrangian $L = c + \lambda^\top g$. At a minimum:

```math
F(z, \lambda, \theta) =
\begin{bmatrix}
\nabla_z L \\
g
\end{bmatrix}
=
\begin{bmatrix}
\nabla_z c + (\nabla_z g)^\top \lambda \\
g
\end{bmatrix}
= 0 .
```

The second row says the plan obeys the model. The first row says the cost cannot
be lowered by any change to the plan that keeps obeying the model. A multiplier
$\lambda_t$ measures how much the cost would fall if the model constraint at
step $t$ were loosened slightly. These are the Karush-Kuhn-Tucker (KKT)
conditions.

Let $w = (z, \lambda)$. Differentiating $F(w(\theta), \theta) = 0$ as in the toy
problem gives

```math
\frac{\partial w}{\partial \theta}
= -\left[\frac{\partial F}{\partial w}\right]^{-1}
\frac{\partial F}{\partial \theta},
\qquad
\frac{\partial F}{\partial w} =
\begin{bmatrix} \nabla_{zz} L & (\nabla_z g)^\top \\ \nabla_z g & 0 \end{bmatrix} .
```

The matrix $\partial F / \partial w$ is the KKT matrix. It plays the role that
$a$ played in the toy problem.

Computing all of $\partial w / \partial \theta$ would take one linear solve per
parameter. Training does not need all of it. It needs the gradient of one
number, a loss or reward $\ell$ that depends on the plan. The chain rule gives

```math
\frac{\partial \ell}{\partial \theta}
= \frac{\partial \ell}{\partial w}\frac{\partial w}{\partial \theta}
= -\underbrace{\frac{\partial \ell}{\partial w}
\left[\frac{\partial F}{\partial w}\right]^{-1}}_{\xi^\top}
\frac{\partial F}{\partial \theta} .
```

The marked factor does not involve θ. It is found by solving one linear system,
$(\partial F/\partial w) \xi = (\partial \ell/\partial z, 0)$. The KKT
matrix is symmetric, so this system uses the same matrix as the one above and
not its transpose. The gradient for every
parameter then follows from one multiplication:

```math
\frac{\partial \ell}{\partial \theta}
= -\left(\frac{\partial F}{\partial \theta}\right)^{\!\top} \xi .
```

**The gradient for every parameter therefore costs one extra linear solve, about
as much as one more iteration of the solver.** This is the result everything
else in this document rests on. It is the same reverse-mode idea that makes
backpropagation cheap, applied to a solver, and it means a controller can be
trained inside the same machinery as a neural network: large batches, a GPU, and
gradient ascent.

The result is older than the paper. It comes from adjoint methods in optimal
control and was brought into deep learning around 2017 (Amos and Kolter, 2017;
Amos et al., 2018). The paper's own contribution is what comes next: making that
linear solve fast on a GPU, so that the thousands of solves in one training
update are affordable.

## Solving the problem one quadratic at a time

The drone's model is nonlinear, so the control problem cannot be solved in one
step. Sequential quadratic programming (SQP) solves it as a sequence of simpler
problems. Each pass of the solver loop does three things.

It first builds a quadratic program around the current plan. The cost is
replaced by its second-order approximation, which gives matrices $Q_t$, $R_t$
and vectors $q_t$, $r_t$. The model is replaced by its first-order
approximation, which gives matrices $A^+_t$, $A_t$, $B_t$ and a vector $C_t$:

```math
\min_z \ \sum_t \tfrac12 x_t^\top Q_t x_t + q_t^\top x_t
+ \sum_t \tfrac12 u_t^\top R_t u_t + r_t^\top u_t
\quad \text{subject to} \quad
A^+_t x_{t+1} + A_t x_t + B_t u_t = C_t .
```

It then solves that quadratic program exactly. A quadratic cost with linear
constraints has linear KKT conditions, so this is one linear system.

It finally moves from the current plan toward the quadratic program's answer.
The quadratic program is accurate only near the current plan, so a full step can
make things worse. The line search tries the fractions 1, 0.7, 0.3, 0.1, and
0.01 of the step and keeps the largest one that sufficiently lowers a merit
function, which is the cost plus a penalty on violating the model. If none
qualifies, it takes the smallest.

The loop ends when the plan changes by less than a tolerance between passes, or
when it reaches its iteration limit.

One simplification matters later. The matrices $Q_t$ and $R_t$ use the curvature
of the cost only and ignore the curvature of the model. The paper states that
this is standard, that it avoids second derivatives of the model, and that it
can make the gradients less accurate.

## The shape of the linear system

Stack the plan in time order and the quadratic program's KKT conditions become

```math
\begin{bmatrix} G & H^\top \\ H & 0 \end{bmatrix}
\begin{bmatrix} z \\ \lambda \end{bmatrix}
=
\begin{bmatrix} -b \\ d \end{bmatrix},
```

where $G$ holds the cost matrices, $H$ holds the model matrices, $b$ holds the
vectors $q_t$ and $r_t$, and $d$ holds $x_s$ and the vectors $C_t$. Both
matrices are almost entirely zeros, and the pattern of the zeros is what the
method exploits. For a horizon of three steps:

```
 G: one block per variable, nothing off the diagonal

         x0   u0   x1   u1   x2   u2   x3
   x0  [ Q0                               ]
   u0  [      R0                          ]
   x1  [           Q1                     ]
   u1  [                R1                ]
   x2  [                     Q2           ]
   u2  [                          R2      ]
   x3  [                               Q3 ]

 H: one row of blocks per constraint, each touching one step and the next

            x0   u0   x1   u1   x2   u2   x3
   start  [ I                               ]
   step 0 [ A0   B0   A0+                   ]
   step 1 [           A1   B1   A1+         ]
   step 2 [                     A2   B2   A2+ ]
```

$G$ is block diagonal because the cost is a sum of terms that each involve one
state or one control. $H$ is a staircase because each model equation involves
only step $t$ and step $t+1$. Time gives the problem its structure.

## Why conjugate gradient

The system is solved in two stages. The top row gives the plan in terms of the
multipliers, $z = -G^{-1}(b + H^\top \lambda)$. Substituting this into the
bottom row leaves a smaller system in the multipliers alone:

```math
S \lambda = \gamma,
\qquad S = -H G^{-1} H^\top,
\qquad \gamma = d + H G^{-1} b .
```

This matrix $S$ is the Schur complement. Inverting $G$ is cheap, because each
block is small and all blocks can be inverted at the same time. $S$ inherits the
staircase of $H$: each row of blocks touches only its neighbors in time.

```
 S: block tridiagonal

   [ *  *             ]
   [ *  *  *          ]
   [    *  *  *       ]
   [       *  *  *    ]
   [          *  *  * ]
   [             *  * ]
```

The usual way to solve such a system is to eliminate one time step after
another (a Riccati recursion, as in iLQR). Each step needs the result of the
one before, so the work is a chain as long as the horizon, and a GPU cannot
shorten a chain.

Conjugate gradient (CG) never eliminates anything. It improves a guess for
$\lambda$ using only products of $S$ with a vector, and in such a product every
row of blocks can be computed at the same time.

```
 Riccati recursion:   step T -> step T-1 -> ... -> step 0     a chain of T steps

 Conjugate gradient:  [ all steps at once ]  repeated k times  k short passes
```

The trade is a chain of $T$ dependent steps for $k$ independent passes. It pays
off on a GPU when $k$ is modest, and it does not pay off on a CPU: the paper
reports that a Riccati-based solver (trajax) is faster than DiffMPC there.

The number of passes $k$ depends on the condition number of $S$, which is the
ratio between the most and the least that $S$ can stretch a vector. The
number of passes grows roughly with the square root of that ratio. The paper
therefore multiplies by a preconditioner $\Phi^{-1}$, a cheap approximation of
$S^{-1}$ that is also block tridiagonal, so applying it is parallel too. This is
the "preconditioned" in PCG.

CG also accepts a starting guess. Consecutive control steps solve nearly the
same problem, so the previous multipliers are a good start. The paper measures
this warm start as a 4 to 11% saving, depending on the tolerance.

## The backward pass

The gradient needs one linear solve with the KKT matrix of the control problem.
With the simplification from the SQP section, that matrix is the one the last
pass of the solver already built for its quadratic program. The backward pass
is therefore the last forward solve with a different right-hand side:

```
 forward :  [ G  Hᵀ ] [ z ]   [ -b ]
            [ H  0  ] [ λ ] = [  d ]

 backward:  [ G  Hᵀ ] [ z̃ ]   [ ∂ℓ/∂z ]      same matrix, same S, same Φ⁻¹
            [ H  0  ] [ λ̃ ] = [   0   ]

 then:      ∂ℓ/∂θ = -(∂F/∂θ)ᵀ [ z̃ ; λ̃ ]
```

The right-hand side $\partial \ell / \partial z$ says how much the downstream
reward cares about each entry of the plan. In a flight, only the first control
is used, so it is zero everywhere except at $u_0$.

The last line needs $\partial F / \partial \theta$, which is how the condition
for a minimum depends directly on the parameters. The code obtains it by
automatic differentiation of the cost and the constraints. For a cost weight the
result is simple: up to a constant factor, the gradient for the weight on one
state component is that component's tracking error multiplied by the matching
entry of $\tilde z$, summed over the plan.

In JAX this is a custom gradient rule attached to the solve. JAX does not
differentiate through the solver's iterations. It calls the rule.

## Training through a flight

The controller is a function from the current state to a control,
$\pi_\theta(x)$, defined as the first control of the solved plan. A flight
alternates the controller and a simulator, and the reward is summed over the
flight:

```math
J(\theta) = \sum_{t=1}^{N} R\big(x_t, \pi_\theta(x_{t-1})\big),
\qquad x_{t} = \mathrm{Sim}\big(x_{t-1}, \pi_\theta(x_{t-1})\big) .
```

```
   θ ────┬──────────────────┬──────────────────┐
         ▼                  ▼                  ▼
  x0 ─▶[solve]─ u0 ─▶[sim]─ x1 ─▶[solve]─ u1 ─▶[sim]─ x2 ─▶ ...
                       │                        │
                       ▼                        ▼
                      R1                       R2
```

The gradient of $J$ follows every arrow backward. The simulator is ordinary JAX
code, so JAX differentiates it. Each solve uses the rule from the previous
section. The parameters feed every solve, so θ receives a contribution from
every control step.

The paper calls this reinforcement learning, and it differs from the kind most
introductions describe. Nothing is learned by trial and error, and no noise is
added to explore. The gradient is computed by differentiating the simulator,
not estimated from noisy trials, which is why the simulator has to be written in
JAX.

One path in the diagram is missing in this repository. A changed state changes
the next plan, which is the arrow from $x_1$ into the second solve. DiffMPC
returns a gradient only for what is passed to it as `weights`, and the notebook
passes the current state another way, so the gradient treats each solve's start
state as fixed. The effect of this on training has not been measured.

Training for robustness adds one thing. Each flight in a batch gets its own
simulator, with its own randomly drawn physical parameters, and the gradients
are added. The controller then cannot fit one simulator and has to do well
across the range. This is domain randomization.

## What the paper measured

The paper reports speed on benchmark problems and one application.

On a GPU, DiffMPC computed rewards and gradients 4 to 7 times faster than the
fastest other solver tested (trajax), across linear problems and a nonlinear
spacecraft problem. On a CPU, trajax was faster. An imitation learning benchmark
on a cart-pole trained about 2 times faster.

The application is a controller that drifts a Toyota Supra. Training used 32
simulated cars per update, each with randomized tire parameters and a water
puddle at a random place, for flights of 200 steps and 1,000 updates, which took
14 hours on an RTX 4090. The authors report that fewer than 32 cars or fewer
than 100 steps did not train reliably.

Training lowered the rear tire friction the controller assumes by 13% and the
cost on sideslip error by 58%. The controller therefore plans as if it has less
grip than it does and tolerates more tracking error, which leaves margin when a
puddle removes grip. In simulation it completed 20 of 20 randomized runs,
against 14 of 20 for the hand-tuned controller. On the real car it drifted a
circle through a puddle where the hand-tuned controller spun out. A second test
on a figure 8 was run once per controller, and the authors draw no conclusion
from it.

Five choices in that experiment are worth knowing before designing one for a
drone.

- Limits on the controls were enforced in the simulator and left out of the
  controller's problem, because DiffMPC handles only equality constraints.
- The solver ran one SQP iteration per control step, in training and on the car,
  not to convergence.
- The cost weights were trained as logarithms, which keeps them positive and
  puts weights of very different sizes on one scale.
- The reward needed a term that rewards sideslip. Without it, training found a
  controller that follows the path without drifting.
- The simulator included a 20 ms control delay and a more accurate integrator
  than the controller's model.

## Reading the code

DiffMPC is about 2,400 lines of Python in `third_party/diffmpc/diffmpc`. Four
files hold nearly all of it.

| File | Holds |
| --- | --- |
| `problems/optimal_control_problem.py` | the cost, the constraints, and their approximations ($Q, q, R, r, A, B, C$) |
| `solvers/sqp.py` | the solver loop, the line search, and the gradient rule |
| `solvers/pcg.py` | $S$, $\Phi^{-1}$, the conjugate gradient loop, and the plan recovered from $\lambda$ |
| `dynamics/` | one `state_dot` per vehicle, and the integrators |

The code makes several choices that the paper does not describe and that shape
how it is used.

**θ is whatever is passed as `weights`.** `solver.solve(guess, params, weights)`
takes two dictionaries. Every entry of `weights` replaces the entry of the same
name in `params` and receives a gradient. Everything left in `params` is held
fixed, and so is the starting guess. A physical parameter is trained by moving
it from one dictionary to the other. Passing `mass` this way returns a gradient
for the mass the controller assumes; this was checked once and not validated
further.

**The gradient rule is required, not only faster.** The solver and PCG loops are
`jax.lax.while_loop`, which JAX cannot differentiate in reverse. The rule
attached to `solve` with `jax.custom_vjp` is the only way a gradient gets
through.

**Failure is a value.** Compiled JAX code cannot raise an error partway through.
A solve that does not settle returns a plan with `status` set to -100, and any
NaN in the plan is replaced by zero. Nothing stops the program, so the status is
the only sign that a solve did not settle. The last section says when that
matters.

**Parallel over time means `vmap` over the time index.** The matrix $S$ is never
assembled. It is stored as one small block per time step, and a product with $S$
is one batched matrix multiplication over those blocks. The approximations of
the cost and the model are computed the same way.

**A batch waits for its slowest member.** Under `vmap`, a `while_loop` keeps
running until every member of the batch has finished. One drone whose solve does
not settle costs every drone the full iteration limit.

**The backward pass carries its own warm start.** The solve stores the result of
its backward linear solve in a field named `dual_backward_guess` and passes it
to the next backward solve through the channel JAX uses for gradients. That
field is not a gradient. It is also what JAX 0.12.1 objects to: the field starts
out identical for every drone in a batch, and the newer JAX refuses to combine
it. This repository makes the field depend on each drone's state, by adding zero
times the state to it, and the notebook's form then runs on 0.12.1 unchanged.

**The parameter derivative is taken one parameter at a time.** The solve itself
is done once, but $\partial F / \partial \theta$ is computed with forward-mode
differentiation over every entry of `weights`. This is cheap for 30 numbers and
would not be for the weights of a neural network. A network should instead
output a few parameters, which are then passed as `weights`.

**The cost weights enter inverted.** $S$ is built from $G^{-1}$, so a weight of
0.0001 and a weight of 100,000 put entries of very different sizes into the same
matrix. This is the likely reason PCG needs many passes on the example.

**The problem class is specific.** The cost is a weighted squared distance from
a reference, with three weight vectors, and the model is integrated explicitly,
so $A^+_t$ is always $-I$. A different cost, or a different vehicle, is a new
subclass written outside `third_party/`.

A few smaller things are worth knowing before reading. Constructing a solver
runs one solve. Double precision has to be switched on, because the PCG
tolerance is below what single precision can resolve. The last row of
`solution.controls` is padding.

## Solver status and gradient accuracy

On this repository's example the solver reports an error at every control step
(`status` is -100), and the gradient differs from the exact one by up to a
factor of two. Neither is being fixed.

In most numerical work a solver runs until its answer stops changing, and an
answer from a solver that did not converge is not used. MPC in a control loop
works differently. The controller applies only the first control of each plan
and then replans from the drone's new state, so an error in one plan is
corrected by the next. A common scheme, the real-time iteration, runs a single
solver iteration per control step for this reason, and the paper's car used it
in training and on the road. The quantity being optimized is the flight, not any
one plan.

In most deep learning the gradient is exact for the computation that was run,
and a gradient that fails a finite-difference check indicates a bug. Training
does not depend on that exactness. Each update is a small step, so the gradient
only has to point uphill on average, and trial-and-error methods (e.g., PPO)
train on estimates that are typically much noisier than this one. DiffMPC gives
up exactness on purpose: it ignores the curvature of the drone model so that the
backward pass can reuse the linear system from the forward pass, which is what
makes the gradient cheap.

Both were measured here. The solver does not settle because its linear solver
stops when a scaled error falls below a tolerance, and the example's smallest
cost weights (0.0001) make the scaled error much smaller than the real one.
DiffMPC's notebook behaves the same when run unchanged. With the tolerance
lowered from 1e-12 to 1e-30 and the iteration limit raised from 50 to 400, the
solver settles at all ten control steps, and 20 updates then took 719 s against
200 s and ended at a reward of -139.30 against -130.90, in one run with one
drone. The notebook's settings are therefore the default, and the others are
behind a flag:

```sh
uv run diffmpc-drones --settle
```

The gradient was checked on one settled solve. DiffMPC's gradient matched a
direct calculation with the same simplification to four digits, so the code does
what the paper describes. A direct calculation that keeps the model's curvature
matched finite differences and differed from DiffMPC's by up to a factor of two.
No setting removes that difference.

```sh
uv run python -m diffmpc_drones.gradient_check
```

If training on a harder task stalls or moves the wrong way, `--settle` is the
first thing to try.

The method also has limits that no setting removes. The controller's problem can
hold only equality constraints, so a limit (e.g., maximum thrust) has to live in
the simulator or in the cost. The gradient passes through every step of the
simulated flight, so its error grows with the length of the flight, and grows
faster for a vehicle that is unstable. The paper adds a third as an open
problem: a poor starting guess can make the solver diverge.

The open work for this repository is in [TODO.md](TODO.md).
