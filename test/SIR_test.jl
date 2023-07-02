using RandomTimeShifts
using Test
using OrdinaryDiffEq
using Random
using Distributions

"""
    diff_SIR_coeffs(β, lifetimes, λ, phis)
    
Function which differentiates the functional equation of the PGF and LST. 
        
Arguments: 
    β = effective transmission parameter 
    lifetimes = β + γ
    λ = the growth rate 
    phis = the previous moments used to calculate the nth moment
    
Outputs: 
    A[1] = coefficient of nth moment
    b = constant term
"""
function diff_SIR_coeffs(β, lifetimes, λ, phis)
    n_previous = length(phis)
    n = n_previous + 1
    A, b = RandomTimeShifts.diff_quadratic_1D(β, lifetimes, n, λ, phis)
    A += RandomTimeShifts.lhs_coeffs(1)
    return A[1], b
end

"""
    F_fixed_s(u, pars, t)

Evaluates the Ricatti ODE governing F(s, t) for fixed u0 = s.

Arguments: 
    u = current state
    pars = (R0, γ_inv)
    t = dummy variable for the current time 
    
Outputs: 
    du = value of ODE
"""
function F_fixed_s(u, pars, t)
    R0, γ_inv = pars
    γ = 1 / γ_inv
    β = R0 * γ
    du = -(β + γ) * u + γ + β * u^2
    return du
end

"""
    F_offspring_ode(s, t, pars)
    
Calculates the PGF for the imbedded process given a fixed s and integrating over (0, t
with pars = (R0, γ_inv).
"""
function F_offspring_ode(s, t, pars)
    u0 = s[1]
    prob = ODEProblem(F_fixed_s, u0, (0, t), pars)
    sol = solve(prob, Tsit5(); save_start = false)

    return sol.u[end]
end

"""
sir_deterministic!(du, u, pars, t)
    
Evaluate the system of ordinary differential equations for the SIR model with parameters 
pars = (R0, γ_inv).

Arguments: 
    du = required for inplace calculations when using OrdinaryDiffEq
    u = current state
    pars = model parameters in form (R0, γ_inv)
    t = dummy variable for the current time
    
Outputs: 
    None
"""
function sir_deterministic!(du, u, pars, t)
    R0, γ_inv = pars
    γ = 1 / γ_inv
    β = R0 * γ

    s, i, i_log = u

    du[1] = ds = -β * i * s
    du[2] = di = β * i * s - γ * i
    du[3] = di_log = β * s - γ

    return nothing
end

"""
    compute_W_lst(pars, Z0)
    
Computes the LST using our method and inverts it. 

Arguments: 
    pars = (R0, γ_inv)
    Z0 = initial condition for the SIR model
    
Outputs: 
    W_lst = a function that computes the LST of W
"""
function compute_W_lst(pars, Z0)
    R0, γ_inv = pars
    γ = 1 / γ_inv
    β = R0 * γ

    a = γ + β
    λ = β - γ

    num_moments = 31

    μ = 2 * β / (β + γ)

    diff_SIR_coeffs_(phis) = diff_SIR_coeffs(β, a, λ, phis)

    moments = RandomTimeShifts.calculate_moments_1D(diff_SIR_coeffs_)

    moments_err = moments[end]
    moments = moments[1:(end - 1)]

    ϵ_target = 1e-10
    L = RandomTimeShifts.error_bounds(ϵ_target, moments_err, num_moments - 1)

    u0 = 0.5
    h = 0.1

    prob = ODEProblem(F_fixed_s, u0, (0, h), pars)
    sol = solve(prob, Tsit5(); abstol = 1e-11, reltol = 1e-11)

    μ = exp(λ * h)
    F_offspring(s) = F_offspring_ode(s, h, pars)

    coeffs = RandomTimeShifts.moment_coeffs(moments)
    W_lst = RandomTimeShifts.construct_lst(coeffs, μ, F_offspring, L, Z0[2], λ, h)

    return W_lst
end

"""
    compute_W_distribution(I0, pars, W_lst)
    
Computes the LST using our method and inverts it. 

Arguments: 
    I0 = initial condition for the SIR BP model
    pars = (R0, γ_inv)
    W_lst = the lst for W 
    
Outputs: 
    W_cdf = a function that computes the CDF of W
"""
function compute_W_distribution(I0, pars, W_lst)
    R0, γ_inv = pars
    γ = 1 / γ_inv
    β = R0 * γ
    q = γ / β

    q_star = q^I0

    W_cdf = RandomTimeShifts.construct_W_cdf_ilst(W_lst, q_star)
    return W_cdf
end

"""
    SIR_lst_exact 
    
Arguments: 
    θ = the input for the LST 
    pars = parameters in form (R0, γ_inv)

Outputs:
    The exact value of the LST for W
"""
function SIR_lst_exact(θ, pars)
    R0, γ_inv = pars

    γ = 1 / γ_inv
    β = R0 * γ
    q = γ / β

    return q + (1 - q) / (1 + θ / (1 - q))
end

"""
    SIR_cdf_exact 
    
Arguments: 
    w = the input for the CDF
    pars = parameters in form (R0, γ_inv)

Outputs:
    The exact value of the CDF for W
"""
function SIR_cdf_exact(w, pars)
    R0, γ_inv = pars

    γ = 1 / γ_inv
    β = R0 * γ
    q = γ / β

    return 1 - (1 - q) * exp(-(1 - q) * w)
end

##

@testset "SIR LST" begin
    Random.seed!(12345)

    K = Int(10^6)
    Z0 = [K - 1, 1, 0]
    pars = [1.9, 2.0]
    W_lst = compute_W_lst(pars, Z0)

    n_test = 1000
    test_points = rand(Normal(0, 1), n_test) + rand(Normal(0, 1), n_test) * im

    # real_parts_approx_equal = true
    # imag_parts_approx_equal = true

    θ = test_points[1]
    y1 = W_lst(θ)
    y2 = SIR_lst_exact(θ, pars)
    # @test isapprox(real(y1), real(y2), atol = 1e-4)
    # @test isapprox(imag(y1), imag(y2), atol = 1e-4)
    # Check that the inversion is accurate to 4 decimal places.
    for θ in test_points
        y1 = W_lst(θ)
        y2 = SIR_lst_exact(θ, pars)
        @test isapprox(real(y1), real(y2), atol = 1e-3)
        @test isapprox(imag(y1), imag(y2), atol = 1e-3)
    end

    # @test real_parts_approx_equal
    # @test imag_parts_approx_equal
end

@testset "SIR CDF" begin
    Random.seed!(12345)

    K = Int(10^6)
    Z0 = [K - 1, 1, 0]
    pars = [1.9, 2.0]
    W_lst = compute_W_lst(pars, Z0)

    I0 = Z0[2]
    W_cdf = compute_W_distribution(I0, pars, W_lst)

    n_test = 100
    test_points = rand(Uniform(0, 10), n_test)
    w = test_points[1]
    # Check that the CDF values are the same up to 3 decimal places.
    # @test isapprox(W_cdf(w), SIR_cdf_exact(w, pars), atol = 1e-3)
    for w in test_points
        @test isapprox(W_cdf(w), SIR_cdf_exact(w, pars), atol = 1e-3)
    end
end