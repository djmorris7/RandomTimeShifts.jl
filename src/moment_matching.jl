"""
    moments_gengamma(n, a, d, p)  
    
Calculates the analytical moments for a generalised Gamma distribution with 
scale parameter a, and two shape parameters d, p. 

Arguments: 
    n = the moment number 
    a = scale parameter
    d = shape parameter 1
    p = shape parameter 2
    
Outputs: 
    The moment
"""
function moments_gengamma(n, a, d, p)
    if (gamma((d + n) / p) == 0) || !isfinite(gamma(d / p)) || (gamma(d / p) == 0)
        return Inf
    else
        return a^n * gamma((d + n) / p) / gamma(d / p)
    end
end

"""
    loss_func(n, pars, moments)
    
Calculates the squared loss function (normalised) to be optimised.
    
Arguments: 
    n = the moment number
    pars = the parameters (a, d, p) of the GG 
    moments = the vector of moments
    
Outputs: 
    loss = the value of the loss function for a particular moment
"""
function loss_func(n, pars, moments)
    a, d, p = pars
    y1 = moments[n]
    y2 = moments_gengamma(n, a, d, p)
    loss = if isinf(y2)
        -Inf
    else
        ((y1 - y2) / round(y1; sigdigits = 1))^2
    end
    return loss
end

"""
    sample_generalized_gamma(pars)

Sample a generalised gamma random variable with parameters (a, d, p) using 
the CDF method. 

Arguments: 
    pars = parameters in form (a, d, p)
    
Outputs: 
    x = a sample from GG(a, d, p)
"""
function sample_generalized_gamma(pars)
    a, d, p = pars
    u = rand()
    y = gamma_inc_inv(d / p, u, 1 - u)
    # Apply transformation to obtain Generalized Gamma random variables
    x = a * y^(1 / p)
    return x
end

"""
    sample_W(n, pars, q1, Z0; no_extinction = true)

Sample realisations of W given pars, extinction probabilities. 

Arguments: 
    n = the number of samples to draw 
    pars = list of pars (a, d, p) for each of the Wi
    q1 = vector of extinction probabilities for each Wi
    Z0 = the initial conditions for the branching process
    no_extinction = (default = true) whether to condition on non-extinction
    
Outputs: 
    w = vector of samples of w
"""
function sample_W(n, pars, q1, Z0; no_extinction = true)
    w = zeros(Float64, n)
    i = 1
    while i <= n
        for j in eachindex(Z0)
            Z0[j] == 0 && continue
            for _ in 1:Z0[j]
                # run a Bernoulli trial to see whether the jth process fades out
                if rand(Bernoulli(1 - q1[j]))
                    w[i] += sample_generalized_gamma(pars[j])
                end
            end
        end
        if (no_extinction && (w[i] > 0)) || !no_extinction
            i += 1
        end
    end
    return w
end

"""
    sample_W(pars, q1, Z0)

Sample a single realisation of W given pars, extinction probabilities. 

Arguments: 
    pars = list of pars (a, d, p) for each of the Wi
    q1 = vector of extinction probabilities for each Wi
    Z0 = the initial conditions for the branching process
    
Outputs: 
    w = samples of w
"""
function sample_W(pars, q1, Z0)
    w = 0.0

    while iszero(w)
        for j in eachindex(Z0)
            Z0[j] == 0 && continue
            for _ in 1:Z0[j]
                # run a Bernoulli trial to see whether the jth process fades out
                if rand(Bernoulli(1 - q1[j]))
                    w += sample_generalized_gamma(pars[j])
                end
            end
        end
    end
    return w
end

"""
    minimise_loss(moments, q1; num_moments_loss = 5, iterations = 10^5)
    
Minimises sum of moments - (analytical moments).

Arguments:
    moments = an array of shape (5, number types) with the moments estimated using the methods 
              from section 3.3 of the paper
    q1 = vector of extinction probabilities starting with an individual of type i
    num_moments_loss = (default = 5) number of moments to use in the expansion. Setting a 
                       default here lets us calculate more moments for other methods. 
    iterations = (default = 10^5) max number of iterations to run in the optimisation.
    
Outputs: 
    pars = an array of parameters with length number of types corresponding to each Wi
"""
function minimise_loss(moments, q1; num_moments_loss = 5, iterations = 10^3)
    # get the number of loss functions to construct
    num_loss_funcs = size(moments, 2)

    weights = ones(num_moments_loss) / num_moments_loss
    # construct the loss functions for all the conditional moments
    loss_funcs = Dict(j => pars -> sum(weights[i] *
                                       loss_func(i, pars, moments[:, j] ./ (1 - q1[j]))
                                       for
                                       i in 1:num_moments_loss) for j in 1:num_loss_funcs)

    # Wide bounds on the parameters 
    lower_bd = [0.0, 0.0, 0.0]
    upper_bd = [50.0, 50.0, 50.0]
    # Initial guess is uninformative and reflects a boring distribution
    x0 = [1.0, 1.0, 1.0]

    # optimise all loss functions using the Conjugate-Gradient method which approximates 
    # the gradient using finite differences. 
    # TODO: 
    # - Possible improvement here to utilise analytical gradient 
    # - Possible source of error is that conjugate-gradient on constraint not good
    #   but so far seems fine for the examples explored 
    opt_res = Dict(j => Optim.optimize(loss_funcs[j],
                                       lower_bd,
                                       upper_bd,
                                       x0,
                                       Fminbox(ConjugateGradient()),
                                       Optim.Options(; allow_f_increases = true,
                                                     iterations = iterations))
                   for j in eachindex(loss_funcs))

    # initialise an array of arrays to hold the parameters of the distributions
    pars = [zeros(3) for _ in 1:num_loss_funcs]
    # make sure they're inserted based on type order 
    for (k, v) in opt_res
        pars[k] .= v.minimizer
    end

    return pars
end
