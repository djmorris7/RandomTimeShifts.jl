"""
    error_bounds(ϵ, μ, n)
    
Calculates the region about 0 to ensure the error is no more than ϵ given 
the (n + 1)th moment μ and n. 

Arguments: 
    ϵ = the tolerance for the neighbourhood about 0 
    error_moments = a vector of the n+1 moments for each W_i 
    n = the number of moments used in the expansion
    
Outputs: 
    The value of L
"""
function error_bounds(ϵ, error_moments, n)
    return minimum(exp(loggamma(n + 1)) * ϵ ./ error_moments) .^ (1 / (n + 1))
end

"""
    moment_coeffs(M)
    
Calculates the coefficients for the LST expansion.
    
Arguments: 
    moments = an array of shape (num_moments, num types) of the moments
    
Outputs: 
    coeffs = the coefficients in the series expansion near 0
"""
function moment_coeffs(moments)
    coeffs = zeros(Float64, size(moments))

    for n_state in axes(moments, 2), i in axes(moments, 1)
        coeffs[i, n_state] = moments[i, n_state] / exp(loggamma(i + 1))
    end

    return coeffs
end

"""
    lst_s0(s, coeffs)

Calculates the LST around 0 using the Taylor series expansion. 

Arguments: 
    s = the point to evaluate an LST at
    coeffs = the coefficients for the series expansion
        
Outputs: 
    y = the value of the lst 
"""
function lst_s0(s, coeffs)
    y = 1.0

    for (k, c) in enumerate(coeffs)
        y += (-1)^k * c * s^k
    end

    return y
end

"""
    calculate_κ(s, L, λ, h)
    
Helper function that calculates the number of times we need to recursively evaluate 
the extension framework. 
"""
function calculate_κ(s, L, λ, h)
    return max(1, ceil(Int, -log(L / abs(s)) / (λ * h)))
end

"""
    lst_s_rest(s, lst_s0_x, μ, f; ϵ = 0.1)

Calculates the LST everywhere else by recursively applying the functional 
equation that relates the LST to past versions.  

Arguments: 
    s = the point to evaluate the lst at, this should be a scalar in R or C
    lst_s0_x = function which evaluates the LST at s for each Wi 
    μ = the mean of the imbedded process (exp(λ h))
    f = progeny generating function for the imbedded process 
        
Outputs: 
    y = the value of the LSTs for each Wi
"""
function lst_s_rest(s, lst_s0_x, μ, f, λ, h; L=0.1)
    # Use the size of the region about 0 to determine whether 
    # we can simplify things
    if abs(s) <= L
        y = lst_s0_x(s)
    else
        κ = calculate_κ(s, L, λ, h)
        s1 = s * μ^(-κ)

        y = lst_s0_x(s1)

        for _ in 1:κ
            y = f(y)
        end
    end

    return y
end

"""
    lst(s, lst_s0_x, lst_s_rest_x; ϵ = 0.05, Z0 = [1, 0, 0])

Calculates the full LST using the two approaches.

Arguments: 
    s = the point to evaluate the lst at, this should be a scalar in R or C
    lst_s0_x = function which evaluates the LST at s for each Wi in Aϵ
    lst_s_rest_x = function which evaluates the LST at s for each Wi outside Aϵ
    Z0 = initial condition for the branching process 
    
Outputs: 
    out = the lst of W 
"""
function lst(s, lst_s0_x, lst_s_rest_x, Z0; L=0.05)
    y = 0.0

    if abs(s) <= L
        y = lst_s0_x(s)
    else
        y = lst_s_rest_x(s)
    end

    out = prod(y[idx]^z0 for (idx, z0) in enumerate(Z0))

    return out
end

"""
    lst_s0_all(s, coeffs)

Returns the LST for all the initial conditions.
    
Arguments:  
    s = the point to evaluate the lst at, this should be a scalar in R or C
    coeffs = array of size (num_moments, num types) with the coefficients of the 
             series expansion 
             
Outputs: 
    A vector of the LST's for each Wi evaluated at s
"""
function lst_s0_all(s, coeffs)
    return [lst_s0(s, c) for c in eachcol(coeffs)]
end

"""
    construct_lst(coeffs, μ_imbed, F_offspring, ϵ, Z0)

Constructs the LST and returns a function. 

Arguments: 
    coeffs = array of size (num_moments, num types) with the coefficients of the 
             series expansion 
    μ_imbed = the mean of the imbedded process (exp(λ h))
    F_offspring = progeny generating function for the imbedded process 
    L = size (in absolute terms) of region about 0 
    Z0 = initial conditions for the branching process 
    λ = growth rate
    h = step size for the imbedded process
    
Outputs: 
    lst_w = a function for evaluating the LST of W
"""
function construct_lst(coeffs, μ_imbed, F_offspring, L, Z0, λ, h)
    # Set up the lst in the neighbourhood about 0 and make this return the vector of 
    # lsts
    lst_s0_x(s) = lst_s0_all(s, coeffs)
    lst_s_rest_x(s) = lst_s_rest(s, lst_s0_x, μ_imbed, F_offspring, λ, h; L=L)
    lst_w(s) = lst(s, lst_s0_x, lst_s_rest_x, Z0; L=L)
    return lst_w
end

"""
    calculate_BP_contributions(omega)
    
Calculates the BP contributions from the matrix omega which can 
be calculated through either the offspring distributions or the Jacobian
of the approximating deterministic system.

Arguments: 
    Ω = the mean matrix as specified in the paper 
    
Outputs: 
    λ = the growth rate 
    u_norm = the right eigenvector of Ω normalised s.t. u * 1 = 1 and u * v = 1
    v_norm = the left eigenvector of Ω normalised s.t. u * 1 = 1 and u * v = 1
"""
function calculate_BP_contributions(Ω)
    # Compute right and left eigenvectors 
    rvals, rvecs = eigen(Ω)
    lvals, lvecs = eigen(Ω')
    # Get Malthusian parameter
    λ = maximum(rvals)

    rvals = real.(rvals)
    lvals = real.(lvals)
    rvecs = real.(rvecs)
    lvecs = real.(lvecs)
    λ_left = maximum(rvals)
    λ_right = maximum(lvals)
    @assert λ_left ≈ λ_right

    # Get u and v corresponding to the right and left eigenvectors of λ
    r_index = findfirst(x -> x == λ_right, rvals)
    l_index = findfirst(x -> x == λ_left, lvals)
    u = rvecs[:, r_index]
    v = lvecs[:, l_index]

    # Renormalise vectors. 
    u_norm = u / sum(u)
    v_norm = v / sum(u_norm .* v)

    return λ, u_norm, v_norm
end

"""
    F_offspring_ode(s, prob)

Remakes the ODEProblem for the F_i(s, t)'s to solve for a particular value of s.
    
Arguments: 
    s = vector of points to remake the F_i(s, t)'s at and evaluate to approximate the pgf
        of the imbedded process 
    prob = an ODEProblem object constructed for the F_i's 
        
Outputs: 
    sol.u[end] = vector of the values of the imbedded pgf's at s. 
"""
function F_offspring_ode(s, prob)
    prob = remake(prob, u0=s)
    sol = solve(prob, Tsit5())
    return sol.u[end]
end