# TODO

The goal is a drone controller, tuned by training, that survives a disturbance
that the hand-tuned controller does not. The repository currently reproduces
DiffMPC's quadrotor example, which trains a hover controller's cost weights.
None of the work below has been started.

Training comes first, in a simulator written in JAX. The simulated drone is
currently the model the controller plans with, so the first step is a separate
simulator with thrust limits and with mass, drag, and wind drawn separately for
each drone. DiffMPC has no inequality constraints, so the limits can exist only
in the simulator, and it is not known whether training learns to respect them.
Training then covers the mass and thrust that the controller assumes, in
addition to the cost weights. The trained controller is compared with the
starting one on a disturbance that neither was trained on. This needs a GPU;
everything so far has run on a laptop CPU.

A good verification step would be to run the trained controller in
[ripple](https://github.com/EthanMBoos/ripple), a second simulator with a
Crazyflie model. This is not decided, but ripple is probably the easiest
simulator to plug into. Which commands its multirotor accepts, and whether wind
applies to it, have not been checked.

If ripple is used, a further step is flight under ArduCopter in ripple, where
the controller sends setpoints over MAVLink as a companion computer does on a
real drone. ArduCopter accepts a thrust and body rates, not torques, so the
controller would need a second drone model with those controls. ripple's
ArduPilot bridge flies only ArduPlane today, so this requires copter support in
ripple first.

## Known limits

The solver does not settle on the quadrotor example. At each control step it
improves a plan over repeated iterations and should stop once the plan changes
by less than 0.00000001 between iterations. Here it uses all 50 allowed
iterations at every step, is still changing the plan by between 0.00001 and 0.3,
and reports an error status. The drone still flies, because an unsettled plan is
still a usable plan.

This comes from DiffMPC's quadrotor example, not from this repo's changes or
from DiffMPC in general. The notebook reports the same status when run unchanged
on JAX 0.7.2, and DiffMPC's own tests pass. The likely cause is the example's
cost weights, which range from 0.0001 to 100,000, but that has not been
confirmed.

It matters because DiffMPC's gradient assumes that the solver found the best
plan. When it has not, the gradient is approximate. Training still raises the
reward, but not at every update.
