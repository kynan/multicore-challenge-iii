# Generating high-performance multiplatform finite element solvers from high-level descriptions

!SLIDE left title

# Generating high-performance multiplatform finite element solvers from high-level descriptions

## Florian Rathgeber, Graham R. Markall, Nicolas Loriant, David A. Ham, Paul H. J. Kelly, Carlo Bertolli

### Imperial College London

## Lawrence Mitchell

### University of Edinburgh

## Mike B. Giles, Gihan R. Mudalige

### University of Oxford

## Istvan Z. Reguly

### Pazmany Peter Catholic University, Hungary

!SLIDE left

# FEM is a versatile tool for science and engineering

## Tsunami simulation of the Hokkaido-Nansei-Oki tsunami of 1993

<iframe width="853" height="480" src="http://www.youtube.com/embed/Y6mM_PCNhq0?rel=0" frameborder="0" allowfullscreen></iframe>

The simulation was carried out with [Fluidity CFD code](http://amcg.ese.ic.ac.uk/index.php?title=FLUIDITY) solving the non-hydrostatic Navier-Stokes equations, using a [free surface and wetting and drying algorithm](http://amcg.ese.ic.ac.uk/index.php?title=Wetting_and_Drying)

!SLIDE huge

# The challenge

> How do we get performance portability for the finite element method without sacrificing generality?

!SLIDE left

# The strategy

## Get the abstractions right
... to isolate numerical methods from their mapping to hardware

## Start at the top, work your way down
... as the greatest opportunities are at the highest abstraction level

## Harness the power of DSLs
... for generative, instead of transformative optimisations

!SLIDE left

# The tools

## Embedded domain-specific languages

... capture and *efficiently express characteristics* of the application/problem domain

## Active libraries

... encapsulate *specialist performance expertise* and deliver *domain-specific optimisations*

## In combination, they

* raise the level of abstraction and incorporate domain-specific knowledge
* decouple problem domains from their efficient implementation on different hardware
* capture design spaces and open optimisation spaces
* enable reuse of code generation and optimisation expertise and tool chains

!SLIDE huge

# The big picture

!SLIDE

}}} images/mapdes_abstraction_layers_helmholtz.svg

!SLIDE huge

# Higher level abstraction

## From the equation to the finite element implementation

!SLIDE left

# FFC takes equations in UFL

## Helmholtz equation
@@@ python
f = state.scalar_fields["Tracer"]

v = TestFunction(f)
u = TrialFunction(f)

lmbda = 1
a = (dot(grad(v), grad(u)) - lmbda * v * u) * dx

L = v*f*dx

solve(a == L, f)
@@@

!NOTES

## Fluidity extensions

* **`state.scalar_fields`** interfaces to Fluidity: read/write field of given name
* **`solve`** records equation to be solved and returns `Coefficient` for solution field

!SLIDE left

# ... and generates local assembly kernels

## Helmholtz OP2 kernel
@@@ clike
void kernel(double A[1][1], double *x[2],
            int j, int k) {
  // Kij - Jacobian determinant
  // FE0 - Shape functions
  // Dij - Shape function derivatives
  // W3  - Quadrature weights
  for (unsigned int ip = 0; ip < 3; ip++) {
    A[0][0] += (FE0[ip][j] * FE0[ip][k] * (-1.0)
      + (((K00 * D10[ip][j] + K10 * D01[ip][j]))
        *((K00 * D10[ip][k] + K10 * D01[ip][k]))
      + ((K01 * D10[ip][j] + K11 * D01[ip][j]))
        *((K01 * D10[ip][k] + K11 * D01[ip][k]))
      )) * W3[ip] * det;
  }
}
@@@

!SLIDE huge

# Lower level abstraction

## From the finite element implementation to its efficient parallel execution

!SLIDE left

# OP2 – an active library for unstructured mesh computations

## Abstractions for unstructured grids

* **Sets** of entities (e.g. nodes, edges, faces)
* **Mappings** between sets (e.g. from edges to nodes)
* **Datasets** holding data on a set (i.e. fields in finite-element terms)

## Mesh computations as parallel loops

* execute a *kernel* for all members of one set in arbitrary order
* datasets accessed through at most one level of indirection
* *access descriptors* specify which data is passed to the kernel and how it is addressed

## Multiple hardware backends via *source-to-source translation*

* partioning/colouring for efficient scheduling and execution on different hardware
* currently supports CUDA/OpenMP + MPI - OpenCL, AVX support planned

!SLIDE left

# OP2 for finite element computations

## Finite element local assembly
... means computing the *same kernel* for *every* mesh entity (cell, facet)

## OP2 abstracts away data marshaling and parallel execution
* controls whether/how/when a matrix is assembled
* OP2 has the choice: assemble a sparse (CSR) matrix, or keep the local assembly matrices (local matrix approach, LMA)
* local assembly kernel is *translated* for and *efficiently executed* on the target architecture

## Global asssembly and linear algebra operations
... implemented as a thin wrapper on top of backend-specific linear algebra packages:  
*PETSc* on the CPU, *Cusp* on the GPU

!SLIDE left

# Finite element assembly and solve in PyOP2

@@@ python
def solve(A, x, b):
    # Generate kernels for matrix and rhs assembly
    mat_code = ffc_interface.compile_form(A, "mat")
    rhs_code = ffc_interface.compile_form(b, "rhs")
    mat_kernel = op2.Kernel(mat_code, "mat_cell_integral_0_0")
    rhs_kernel = op2.Kernel(rhs_code, "rhs_cell_integral_0_0")

    # misc setup (skipped)

    # Construct OP2 matrix to assemble into
    sparsity = op2.Sparsity((elem_node, elem_node), sparsity_dim) 
    mat = op2.Mat(sparsity, numpy.float64)
    f = op2.Dat(nodes, 1, f_vals, numpy.float64)

    # Assemble and solve
    op2.par_loop(mass, elements(3,3),
             mat((elem_node[op2.i[0]], elem_node[op2.i[1]]), op2.INC),
             coords(elem_node, op2.READ))

    op2.par_loop(rhs, elements(3),
             b(elem_node[op2.i[0]], op2.INC),
             coords(elem_node, op2.READ),
             f(elem_node, op2.READ))

    op2.solve(mat, b, x)
@@@

!SLIDE huge

# Preliminary performance results

!SLIDE left

# Experimental setup

##Solver
CG with Jacobi preconditioning using PETSc 3.1 (PyOP2), 3.2 (DOLFIN)

##CPU
2 x 6 core Intel Xeon E5650 Westmere (HT off), 48GB RAM

##Mesh
2D unit square meshed with triangles (200 - 204800 elements)

##Dolfin
Revision 6906, Tensor representation, CPP optimisations on, form compiler optimisations off

!SLIDE

}}} images/runtime_linear.svg

!SLIDE

}}} images/speedup_linear.svg

!SLIDE left

# Resources

All the code mentioned is open source and available on *GitHub*. Try it!

## OP2 library
<https://github.com/OP2/OP2-Common>

## PyOP2
<https://github.com/OP2/PyOP2>

## FFC
<https://code.launchpad.net/~mapdes/ffc/pyop2>

## This talk
<https://kynan.github.com/multicore-challenge-iii>
