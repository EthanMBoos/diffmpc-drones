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
