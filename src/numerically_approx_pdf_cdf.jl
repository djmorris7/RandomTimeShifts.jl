"""
    W_cdf_approx(x, W_cdf, q_star)

Uses the LST for the CDF of W. This adds the point mass in
at x = 0.

# Arguments:
    - x: value to evaluate cdf at
    - W_cdf: function that computes cdf values of W
    - q_star: ultimate extinction probability

# Outputs:
    - CDF value of w
"""
function W_cdf_approx(x, W_cdf, q_star)
    if x == 0.0
        return q_star
    else
        return W_cdf(x)
    end
end

"""
    pdf_from_cdf(y, h)

Numerically differentiates cdf values in y using step size h.

# Arguments:
    - y: cdf values
    - h: step size for differentiation

# Outputs:
    - pdf_vals: the values of the pdf
"""
function pdf_from_cdf(y, h)
    pdf_vals = similar(y)
    num = 0.0
    den = 0.0

    for i in eachindex(y)
        if (i > 1) && (i < length(y))
            num = y[i + 1] - y[i - 1]
            den = 2 * h
        elseif i == 1
            num = y[2] - y[1]
            den = h
        elseif i == length(y)
            num = y[length(y)] - y[length(y) - 1]
            den = h
        end

        pdf_vals[i] = num / den

        #TODO: Deal with the numerical differentiation issues
        # # Deal with numerical differentiation errors.
        # if pdf_vals[i] < 1e-4
        #     pdf_vals[i] = 0.0
        # end
    end

    return pdf_vals
end

"""
    eval_cdf(cdf, x)

Evaluates the cdf at a range of values in a vector x.

# Arguments:
    - cdf: cdf for a random variable
    - x: points to evaluate the cdf at

# Outputs:
    - A vector of CDF values (adjusted to ensure CDF <= 1) which deals with numerical issues.
"""
function eval_cdf(cdf, x)
    # This truncates the results dealing with errors in the numerical approximation
    return [min(1.0, cdf(xi)) for xi in x]
end
