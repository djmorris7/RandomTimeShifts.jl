"""
    diff_linear(lifetimes, n, λ, s_index, num_phis; b = lifetimes)

Differentiate the functional equation assuming a linear form of the PGFs:
f_i(s) = c / a_i + b / a_i * s_j
Default assumes that b = a_i and in cases where f_i(s) = s_j, this is required.
Returns coefficients of the generating functions and the constant term (0 in this case).

# Arguments
    - lifetimes: the lifetime of an individual
    - n: which derivative we're calculating.
    - λ: the growth rate
    - s_index: the index of the s_j
    - num_phis: the number of phis in the model
    - b: (default = lifetimes) the birth rate of type j individuals

# Outputs
    - coeffs: the coefficients from differentiating the pgf
    - C: the constant vector from differentiating the pgf
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

# Arguments
    - b: the birth rate of type j individuals
    - lifetimes: the lifetime of an individual
    - n: the derivative number to calculate
    - λ: the growth rate
    - phis: a vector of the previous phis

# Outputs
    - coeff: the coefficient from differentiating the pgf
    - C: the constant from differentiating the pgf
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

# Arguments
    - b: the splitting rate of type j individuals
    - lifetimes: lifetime of an individual
    - n: the derivative number to calculate
    - λ: the growth rate
    - phis: an array of shape (n-1, number of types) with the previous moments
    - phi_idxs: (default is just to provide the form needed) the indices appearing in the pgf

# Outputs
    - coeffs: the coefficients from differentiating the pgf
    - C: the constant vector from differentiating the pgf
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

# Arguments
    - phi_idx: the index of the pgf
    - num_phis: (default = 1) how many trivial initial conditions the model has

# Outputs
    - coeffs: the coeffs (works for both 1D and ND)
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

# Arguments
    - coeff_func!: function which differentiates all pgf's in place
    - num_moments: number of moments to calculate
    - Ω: the mean matrix shape should be (number types, number types)

# Outputs
    - moments: a vector of shape (num_moments, number types) with the moments
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

# Arguments
    - diff: function which calculates the moments in a 1D case
    - num_moments: (default = 31) how many moments to calculate.

# Outputs
    - moments: vector of moments for W
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

"""
    compute_tilde_constants(c, lifetime, n, λ)

Computes the tilde constants for either the linear or quadratic rates in the functional equation.

# Arguments
    - c: a rate term (i.e. α_ij or β_ikl)
    - lifetime: the lifetime of the individual
    - n: the derivative number (i.e. the moment number)
    - λ: the growth rate

# Outputs
    - c_tilde: the constant term in the quadratic form
"""
function compute_tilde_constants(c, lifetime, n, λ)
    c_tilde = c / (lifetime + n * λ)

    return c_tilde
end

"""
    compute_d_n(βs_i, lifetime, prev_moments, n, λ)

Computes the constant term in the quadratic form for the functional equation.

# Arguments
    - βs_i: the quadratic terms for the ith type, represented as a dictionary with the
            keys being the pgf indices and the values being another dictionary for
            the [i, k, l] => β_ikl that are non-zero
    - lifetime: the lifetime of the individual
    - prev_moments: the previous moments
    - n: the derivative number (i.e. the moment number)
    - λ: the growth rate

# Outputs
    - d_n: the constant term in the quadratic form
"""
function compute_d_n(βs_i, lifetime, prev_moments, n, λ)
    d_n = 0.0
    for (key, val) in βs_i
        i, k, l = key
        β_tilde = compute_tilde_constants(val, lifetime, n, λ)
        d_n +=
            β_tilde * sum(
                binomial(n, r) * prev_moments[r, k] * prev_moments[n - r, l] for
                r in 1:(n - 1)
            )
    end

    return d_n
end

"""
    diff_compute_moments(Ω, αs, βs, lifetimes; num_moments = 5)

Calculates the moments of the functional equation given the linear and quadratic terms.

# Arguments
    - Ω: the mean matrix shape should be (number types, number types)
    - αs: the linear terms, represented as a dictionary with the keys being the pgf indices
          and the values being another dictionary for the [i, j] => α_ij that are non-zero
    - βs: the quadratic terms, represented as a dictionary with the keys being the pgf indices
          and the values being another dictionary for the [i, k, l] => β_ikl that are non-zero
    - lifetimes: the lifetime of each individual
    - num_moments: (default = 5) how many moments to calculate.

# Outputs
    - moments: a vector of shape (num_moments, number types) with the moments
               where column i is the moments for W_i
"""
function calculate_moments_ND(Ω, αs, βs, lifetimes; num_moments = 5)
    # Number of types
    m = length(lifetimes)

    # Get the identity matrix so we can use it to construct the basis vectors
    I_mat = I(m)

    λ, u_norm, v_norm = RandomTimeShifts.calculate_BP_contributions(Ω)

    # Initialize the moments
    moments = zeros(num_moments, m)
    moments[1, :] .= u_norm
    # Initialize the matrix and vector for the linear system
    C = zeros(Float64, m, m)
    d = zeros(Float64, m)

    for n in 2:num_moments
        # Some redundancy here in recalculting the matrix each step, but easier to extend and read
        for i in 1:m
            αs_i = αs[i]
            βs_i = βs[i]

            # Basis vector for the ith moment
            C[i, :] .= I_mat[i, :]
            # Subtract the contributions from the linear parts
            if !isnothing(αs_i)
                for (key, α_ij) in αs_i
                    i, j = key
                    α_ij_tilde = compute_tilde_constants(α_ij, lifetimes[i], n, λ)
                    C[i, :] .-= α_ij_tilde * I_mat[j, :]
                end
            end
            # Subtract the contributions from the quadratic parts
            if !isnothing(βs_i)
                for (key, β_ikl) in βs_i
                    i, k, l = key
                    β_ikl_tilde = compute_tilde_constants(β_ikl, lifetimes[i], n, λ)
                    C[i, :] .-= β_ikl_tilde * (I_mat[k, :] + I_mat[l, :])
                end

                # Compute constant term (which is simply 0 if no quadratic terms)
                d[i] = compute_d_n(βs_i, lifetimes[i], moments, n, λ)
            end
        end
        # Solve and reset the system
        moments[n, :] .= C \ d
        C .= 0.0
        d .= 0.0
    end

    return moments
end

function calculate_moments_ND(coeff_func!, num_moments, Ω)
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
