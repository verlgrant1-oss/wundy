# Wundy User Manual (Draft – Week 1)

## 1) Problem Statement (Linear, Small-Strain 1D Bar)
This project implements a 1-D finite element solver for a linear elastic bar under small strain.  
It computes nodal displacements, reaction forces, and internal element forces using the relation:

- **Strain:**  ε = du/dx  
- **Stress:**  σ = E ε  
- **Element stiffness (2-node bar):**  k_e = (E A / L_e) * [[1, -1], [-1, 1]]  
- **Global system:**  K u = F  

**Boundary conditions supported (Week 1):**
- **Dirichlet (prescribed displacement)** — defined using the `boundary` key (u = 0 by default).
- **Neumann (nodal forces)** — defined using the `cload` key.

> Units must be consistent (e.g., inches & pounds, or meters & newtons).

---

## 2) YAML Input Format
All input data are provided through a single YAML file with a top-level key `wundy:`.

### Keys inside `wundy:`:
| Key | Description | Example |
|-----|--------------|----------|
| `coords` | List of nodal x-coordinates. | `[0, 1, 2, 3, 4]` |
| `connect` | 2-node element connectivity. | `[[0,1],[1,2],[2,3],[3,4]]` |
| `boundary` | Nodes with fixed displacement (u=0). | `- node: 0` |
| `cload` | Nodal concentrated loads (forces). | `- node: 4  amplitude: 2.0` |
| `material` | Material properties (linear elastic). | `- type: elastic  name: mat-1  parameters: {E: 10.0, nu: 0.3}` |
| `element block` | Groups elements with a material and element type. | `- material: mat-1  name: block-1  elements: all  element_type: t1d1` |

---

## 3) Example Inputs

### A) Point-Load Example
*(corresponds to `tests/first.py::test_first_1`)*
```yaml
wundy:
  coords: [0, 1, 2, 3, 4]
  connect: [[0,1],[1,2],[2,3],[3,4]]
  boundary:
    - node: 0
  cload:
    - node: 4
      amplitude: 2.0
  material:
    - type: elastic
      name: mat-1
      parameters: {E: 10.0, nu: 0.3}
  element block:
    - material: mat-1
      name: block-1
      elements: all
      element_type: t1d1
