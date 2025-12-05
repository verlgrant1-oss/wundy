# WUNDY — 1D Finite Element Framework
A modular educational FEA code for linear and nonlinear bar elements.

---

## Overview

WUNDY is a lightweight, modular 1D finite-element framework developed over several structured weekly milestones.  
The framework supports:

- Linear elastic bar elements  
- Nonlinear Neo-Hookean bar elements  
- Newton–Raphson nonlinear solving  
- YAML-driven input files  
- Full preprocess → assembly → solve → postprocess pipeline  
- Complete automated testing (pytest)

All components integrate into a general-purpose solver that loads a YAML model, preprocesses it, and solves either a linear or nonlinear FE system depending on the material definitions.

---

## Repository Structure
```
wundy/
│
├── bin/
│   └── run.py
│
├── src/wundy/
│   ├── first.py
│   ├── elements.py
│   ├── materials.py
│   ├── ui.py
│   ├── solvers.py
│   ├── nonlinear_bar.yaml
│   └── linear_bar.yaml
│
├── tests/
│   ├── first.py
│   ├── test_elements.py
│   ├── test_materials.py
│   ├── user_input.py
│
└── README.md
```

---

## Features Implemented

### 1. YAML Input System
Every simulation is controlled by a single YAML file with sections:

- nodes  
- elements  
- boundary conditions  
- materials  
- element blocks  
- concentrated & distributed loads  

The schema is validated and converted into internal FE arrays using `ui.preprocess`.

---

### 2. Element Library

Implemented element:
- **T1D1** — 2-node 1D bar element

Capabilities:
- Strain extraction  
- Linear stiffness  
- Nonlinear tangent stiffness  
- Internal force vector  

---

### 3. Material Models

#### Elastic
σ = E * ε

#### Neo-Hookean (1D)
Used for the nonlinear example.  
Provides stress and tangent modulus for Newton iterations.

---

### 4. Assembly System
WUNDY performs:

- Global stiffness matrix assembly  
- Global internal force assembly  
- External force vector assembly  
- Automatic DOF mapping  

DOFs follow:

```
global = node_index * dof_per_node + local_dof
```

---

### 5. Nonlinear Newton Solver

The `solvers.py` module performs:

```
loop:
    build tangent stiffness
    compute internal force
    residual = Fext - Rint
    solve Δu
    update u
    check convergence
```

Works for any 1D nonlinear material.

---

## Running Examples

### Linear Bar
```bash
python bin/run.py src/wundy/linear_bar.yaml
```

### Nonlinear Bar
```bash
python bin/run.py src/wundy/nonlinear_bar.yaml
```

---

## Tests

Run all tests:

```bash
pytest
```

Expected:

```
8 passed in 0.16s
```

Tests cover:

- materials  
- elements  
- preprocess  
- first-order FE solver  
- case-insensitive name lookups  

---

## Development Timeline

### Week 1
- Basic linear FE solver (`first.py`)

### Week 2
- Modular elements, materials, preprocess

### Week 3
- Neo-Hookean model and Newton solver

### Final Project
- Full YAML-driven solver  
- Linear + nonlinear integration  
- Example models  
- All tests pass  

---

## Requirements
```
numpy
pyyaml
schema
pytest
```

Install:

```bash
pip install -r requirements.txt
```

---

## Future Extensions

- Multiple DOFs per node  
- Truss & frame elements  
- Plasticity  
- Eigenvalue analysis  
- Visualization  

---

## Notes

This project was developed by Verl Grant as a complete instructional FE code base.  
All components are functional and verified by the automated test suite.
