Wundy User Manual
1. Introduction

Wundy is a 1-dimensional finite element solver for bars and beams under small-strain assumptions. It supports linear elastic behavior and a nonlinear 1D Neo-Hookean material and solves nonlinear problems using the Newton–Raphson method. All problems are defined entirely through YAML input files.

2. Problem Statement and Governing Equations

Wundy solves 1D, small-strain, quasi-static problems for bars and beams.

2.1 Small-Strain Kinematics

ε(x) = du/dx

2.2 Equilibrium (Strong Form)

Bar with distributed load b(x):
d(Aσ)/dx + b(x) = 0

Beam with distributed load q(x):
EI d⁴w/dx⁴ + q(x) = 0

2.3 Weak Form

r(u) = f_ext – f_int = 0
Linear case: Ku = f

3. Material Models
3.1 Linear Elastic

σ = E ε
Tangent modulus = E

3.2 1D Neo-Hookean

λ = 1 + ε
Provides nonlinear stress and tangent modulus.

4. Boundary Conditions and Loads
4.1 Dirichlet BCs

Prescribed displacement at a node (u = value).

4.2 Nodal Forces

Concentrated forces at nodes.

4.3 Distributed Loads

Converted to equivalent nodal forces using Gauss quadrature.

5. Newton–Raphson Method

Residual: r = f_ext – f_int
Tangent stiffness: K_T = ∂f_int/∂u

Iteration steps:

Initial guess

Compute residual

Compute tangent

Solve K_T Δu = r

Update u

Check convergence

6. YAML Input Structure
6.1 Top-Level Layout

wundy:
coords: [...]
connect: [...]
boundary: [...]
cload: [...]
material: [...]
element block: [...]
solver: {...}

6.2 Node Coordinates

coords: [0.0, 1.0, 2.0, 3.0]

6.3 Connectivity

connect:

[0,1]

[1,2]

[2,3]

6.4 Boundary Conditions

boundary:

node: 0
value: 0.0

6.5 Nodal Forces

cload:

node: 3
amplitude: 2.0

6.6 Materials

material:

type: ELASTIC
name: MAT_LINEAR
parameters:
E: 10.0

6.7 Element Blocks

element block:

name: bar-block
material: MAT_LINEAR
elements: all
element_type: t1d1

6.8 Solver Settings

solver:
type: nonlinear
max_iters: 25
tol_residual: 1e-8

7. Example Problems
7.1 Linear Bar Example

wundy:
coords: [0,1,2,3,4]
connect:
- [0,1]
- [1,2]
- [2,3]
- [3,4]
boundary:
- node: 0
value: 0.0
cload:
- node: 4
amplitude: 2.0
material:
- type: ELASTIC
name: MAT_LINEAR
parameters:
E: 10.0
element block:
- name: bar-block
material: MAT_LINEAR
elements: all
element_type: t1d1
solver:
type: linear

Expected solution: u(L) = PL/(EA)

7.2 Neo-Hookean Nonlinear Bar

wundy:
coords: [0,1,2,3]
connect:
- [0,1]
- [1,2]
- [2,3]
boundary:
- node: 0
value: 0.0
cload:
- node: 3
amplitude: 1.0
material:
- type: NEO_HOOKE_1D
name: MAT_NH
parameters:
E: 200.0
element block:
- name: bar-block
material: MAT_NH
elements: all
element_type: t1d1
solver:
type: nonlinear
max_iters: 25
tol_residual: 1e-8

7.3 Beam MMS Verification

Manufactured solution: w(x) = x²(1–x)²
q(x) computed from EI w'''' + q = 0
Used to verify beam element accuracy.

8. Verification and Validation
8.1 Verification

Linear bar matches analytical

Nonlinear bar converges

Beam MMS matches manufactured solution

8.2 Validation

Neo-Hookean stiffening captured

Linear and nonlinear match at small loads

Displacement decreases with larger E

9. Running a Simulation

Command line:
python -m wundy input.yaml
or
python bin/run_wundy.py input.yaml

10. Outputs

Nodal displacements

Reaction forces

Element stress/strain

Load–displacement curves

10.1 Sample Output (Linear Bar Example)

Run the linear bar example:

python -m wundy src/wundy/docs/linear_bar.yaml

--- Wundy Solver Output ---

Nodal Displacements:
u = [0.0000, 0.0500, 0.1000, 0.1500, 0.2000]

Reaction Forces:
Node 0: -2.0000

Element Stresses:
Element 0: 2.0
Element 1: 2.0
Element 2: 2.0
Element 3: 2.0

Solver Status: Converged

11. Summary

Wundy provides:


1D FEM (bars and beams)

Linear & Neo-Hookean materials

Newton–Raphson nonlinear solving

YAML-based input

Verified examples and MMS tests
This manual describes everything needed to run 1D FEM simulations for the final project.