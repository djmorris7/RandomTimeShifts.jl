# RandomTimeShifts

This Julia package provides functionality for solving stochastic density dependent population models according to the regime outlined in "Computation of random time shift distributions for stochastic population models". The code is generalised for multivariable models and there are two common PGF structures implemented.

## Installation

```julia
Pkg.add("git@github.com:djmorris7/RandomTimeShifts.jl.git")
```

Following this we can simply call `using RandomTimeShifts` as per normal.

## Coding style

For demonstrative purposes we have used exact qualifiers when using the packaged code. I.e. we use `RandomTimeShifts.func()` instead of simply writing `func()`. This coding style ensures that linters in editors (like VSCode or Sublime) can provide autocompletion and suggestions.

## To-dos

- As it currently stands, I have not implemented an autodiff framework for discrete-time models. This is a work in progress and is a feature I intend to bring in later on 