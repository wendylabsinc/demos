The G1 29-DOF model with two seven-motor Dex3 hands and its 50 referenced
meshes come from Unitree Robotics' unitree_ros repository, commit
5994d4faef0a9cadd3287f8de0199a67eeb2a259.

Source: https://github.com/unitreerobotics/unitree_ros/tree/5994d4faef0a9cadd3287f8de0199a67eeb2a259/robots/g1_description

The original model is retained as g1.xml. The original repository license is
retained as LICENSE. The task builds its supported-base scene in memory.
Shoulder mount convex-hull contact exclusions are explicit in sim.py.
They require checking against the physical robot during commissioning.
