"""
    diff_linear(lifetimes, n, λ, s_index, num_phis; b = lifetimes)
    
Differentiate the functional equation assuming a linear form of the PGFs:  
f_i(s) = c / a_i + b / a_i * s_j 
Default assumes that b = a_i and in cases where f_i(s) = s_j, this is required.
Returns coefficients of the generating functions and the constant term (0 in this case).

Arguments: 
    lifetimes = the lifetime of an individual
    n = which derivative we're calculating.
    λ = the growth rate
    s_index = the index of the s_j 
    num_phis = the number of phis in the model
    b = (default = lifetimes) the birth rate of type j individuals
    
Outputs: 
    coeffs = the coefficients from differentiating the pgf
    C = the constant vector from differentiating the pgf
"""
function diff_linear(lifetimes, n, λ, s_index, num_phis; b = lifetimes)
    coeffs = zeros(num_phis)
    coeffs[s_index] = -b / (lifetimes + n * λ)
    C = 0.0
    return coeffs, C
end

"""
    diff_quadratic_1D(b, lifetimes, n, λ, phis)

Differentiate the functional equation assuming a quadratic form of the PGFs:  
f(s) = c / a + b / a s^2 
in the special case of a 1D model. 
Returns coefficients of the generating functions and the constant term. 

Arguments: 
    b = the birth rate of type j individuals
    lifetimes = the lifetime of an individual
    n = the derivative number to calculate
    λ = the growth rate
    phis = a vector of the previous phis
    
Outputs: 
    coeff = the coefficient from differentiating the pgf
    C = the constant from differentiating the pgf
"""
function diff_quadratic_1D(b, lifetimes, n, λ, phis)
    # calculate constant contributions
    C = 0.0
    for j in 1:(n - 1)
        C += binomial(n, j) * phis[j] * phis[end - j + 1]
    end
    C *= b / (lifetimes + n * λ)

    coeff = -2 * b / (lifetimes + n * λ)

    return coeff, C
end

"""
    diff_quadratic(b, lifetimes, n, λ, phis; phi_idxs=[1, 2, 3])
    
Differentiate the functional equation assuming a quadratic form of the PGFs: 
f_i(s) = a + b * s_j * s_k
specifying the indices of the pgfs phi_idxs = [i, j, k]. 

Returns coefficients of the generating functions and the constant term. 
    
Arguments: 
    b = the splitting rate of type j individuals
    lifetimes = lifetime of an individual
    n = the derivative number to calculate
    λ = the growth rate
    phis = an array of shape (n-1, number of types) with the previous moments
    phi_idxs = (default is just to provide the form needed) the indices appearing in the pgf
    
Outputs: 
    coeffs = the coefficients from differentiating the pgf
    C = the constant vector from differentiating the pgf
"""
function diff_quadratic(b, lifetimes, n, λ, phis, phi_idxs = [1, 2, 3])
    # calculate constant contributions
    C = 0.0
    for j in 1:(n - 1)
        C += binomial(n, j) * phis[j, phi_idxs[2]] * phis[end - j + 1, phi_idxs[3]]
    end
    C *= b / (lifetimes + n * λ)

    # determine the actual number of phi's we're solving for
    num_phis = size(phis, 2)
    coeffs = zeros(Float64, num_phis)

    # calculate coefficients of the linear equations given the appropriate equivalences
    if phi_idxs[1] == phi_idxs[2]
        coeffs[phi_idxs[1]] = -b / (lifetimes + n * λ)
        coeffs[phi_idxs[3]] = -b / (lifetimes + n * λ)
    elseif phi_idxs[1] == phi_idxs[3]
        coeffs[phi_idxs[1]] = -b / (lifetimes + n * λ)
        coeffs[phi_idxs[2]] = -b / (lifetimes + n * λ)
    elseif phi_idxs[2] == phi_idxs[3]
        coeffs[phi_idxs[2]] = -2 * b / (lifetimes + n * λ)
    else
        coeffs[phi_idxs[2]] = -b / (lifetimes + n * λ)
        coeffs[phi_idxs[3]] = -b / (lifetimes + n * λ)
    end

    return coeffs, C
end

"""
    lhs_coeffs(phi_idx; num_phis = 1)
    
Differentiates the LHS of the functional equation returning the coefficient (always 1.0).

Arguments: 
    phi_idx = the index of the pgf
    num_phis = (default = 1) how many trivial initial conditions the model has 
    
Outputs: 
    coeffs = the coeffs (works for both 1D and ND)
"""
function lhs_coeffs(phi_idx; num_phis = 1)
    if num_phis == 1
        coeffs = 1.0
    else
        coeffs = zeros(num_phis)
        coeffs[phi_idx] = 1.0
    end
    return coeffs
end

"""
    calculate_moments_ND(coeff_func!, num_moments, Ω)
    
Calculates the moments of a ND process using the user specified coefficient functions. 

Arguments: 
    coeff_func! = function which differentiates all pgf's in place 
    num_moments = number of moments to calculate 
    Ω = the mean matrix shape should be (number types, number types)
    
Outputs: 
    moments = a vector of shape (num_moments, number types) with the moments
              where column i is the moments for W_i
"""
function calculate_moments_ND(coeff_func!, num_moments, Ω)
    λ1, u_norm, v_norm = RandomTimeShifts.calculate_BP_contributions(Ω)

    n_phis = length(u_norm)
    moments = zeros(num_moments, n_phis)
    moments[1, :] .= u_norm

    A = zeros(Float64, n_phis, n_phis)
    b = zeros(Float64, n_phis)

    for n in 2:num_moments
        phis = moments[1:(n - 1), :]
        coeff_func!(A, b, phis)
        moments[n, :] .= A \ b
    end

    return moments
end

"""
    calculate_moments_1D(diff; num_moments=31)
    
Calculates the moments of the functional equation for the 1D user specified diff! function. 
    
Arguments: 
    diff = function which calculates the moments in a 1D case
    num_moments = (default = 31) how many moments to calculate.
        
Outputs: 
    moments = vector of moments for W
"""
function calculate_moments_1D(diff; num_moments = 31)
    moments = zeros(Float64, num_moments)
    moments[1] = 1.0

    for n in 2:num_moments
        phis = moments[1:(n - 1), :]
        A, b = diff(phis)
        moments[n] = b / A
    end

    return moments
end