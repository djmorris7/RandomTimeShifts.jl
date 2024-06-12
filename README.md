# RandomTimeShifts

This Julia package provides functionality for solving stochastic density dependent population models according to the regime outlined in:
> Dylan Morris, John Maclean and Andrew J. Black., 2023. Computation of random time shift distributions for stochastic population models.

The code is generalised for multivariable models through a general progeny generating function (PGF) structure. This implemention only incorporates linear and quadratic branching dynamics . See the paper for details. The methods are easily extensible to more complex branching structures but we opted for typical instances arising when working with biological populations (e.g. epidemics, within-host processes). The package provides full support for computing these distributions under different initial conditions and facilitates both approximations (the PE and MM methods) outlined in the paper.

See the paper for details and the example repo [Computation_of_random_time-shifts](https://github.com/djmorris7/Computation_of_random_time-shifts) for an example of how to use this package.

## Installation

```julia
Pkg.add(url = "git@github.com:djmorris7/RandomTimeShifts.jl.git")
```

Following this we can simply call `using RandomTimeShifts` as per normal.

## Coding style

For demonstrative purposes we have used exact qualifiers whenever the packaged code is used. I.e. we use `RandomTimeShifts.func()` instead of simply writing `func()`. This coding style ensures that linters in editors (like VSCode or Sublime) can provide autocompletion and suggestions.
