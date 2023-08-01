"""
    moments_gengamma(n, a, d, p)  
    
Calculates the analytical moments for a generalised Gamma distribution with 
scale parameter a, and two shape parameters d, p. 

### Arguments
    - n: the moment number 
    - a: scale parameter
    - d: shape parameter 1
    - p: shape parameter 2
    
### Outputs
    - The moment
"""
function moments_gengamma(n, a, d, p)
    if (gamma((d + n) / p) == 0) || !isfinite(gamma(d / p)) || (gamma(d / p) == 0)
        return Inf
    else
        return a^n * gamma((d + n) / p) / gamma(d / p)
    end
end

"""
    loss_func(x, moments, q; num_moments_loss = 5)
    
Calculates the squared loss function (normalised) to be optimised.
    
### ### Arguments: 
    - pars: the parameters (a, d, p) of the GG 
    - moments: the vector of moments
    - q: the extinction probability
    - num_moments_loss: (default = 5) the number of moments
    
### Outputs: 
    - loss: the value of the loss function for a particular moment
"""
function loss_func(pars, moments, q; num_moments_loss = 5)
    a, d, p = pars
    loss = 0.0
    for i in 1:num_moments_loss
        y1 = moments[i] / (1 - q)
        y2 = moments_gengamma(i, a, d, p)
        η = round(y1; sigdigits = 1)
        loss += ((y2 - y1))^2 / η
    end
    return loss
end

"""
    ∇loss_func!(g, x, moments, q; num_moments_loss = 5)
    
Calculates the gradient of the loss function (normalised) to be optimised.
    
### Arguments: 
    - g: the gradient which is updated inplace
    - pars: the parameters (a, d, p) of the GG 
    - moments: the vector of moments
    - q: the extinction probability
    - num_moments_loss: (default = 5) the number of moments
    
### Outputs: 
    - nothing, g is updated in place
"""
function ∇loss_func!(g, pars, moments, q; num_moments_loss = 5)
    a, d, p = pars

    g .= 0.0

    for i in 1:num_moments_loss
        ∂a = i * a^(i - 1) * gamma((d + i) / p) / gamma(d / p)
        ∂d = a^i *
             (gamma((d + i) / p) * polygamma(0, (d + i) / p) * (1 / p) * gamma(d / p) -
              gamma((d + i) / p) * gamma(d / p) * polygamma(0, d / p) * (1 / p)) /
             gamma(d / p)^2
        ∂p = a^i *
             (gamma((d + i) / p) * polygamma(0, (d + i) / p) * (-(d + i) / p^2) *
              gamma(d / p) -
              gamma((d + i) / p) * gamma(d / p) * polygamma(0, d / p) * (-d / p^2)) /
             gamma(d / p)^2

        y1 = moments[i] / (1 - q)
        y2 = moments_gengamma(i, a, d, p)
        η = round(y1; sigdigits = 1)
        c = 2 / η * (y2 - y1)

        g[1] += c * ∂a
        g[2] += c * ∂d
        g[3] += c * ∂p
    end
    return nothing
end

"""
    sample_generalized_gamma(pars)

Sample a generalised gamma random variable with parameters (a, d, p) using 
the CDF method. 

### Arguments: 
    - pars: parameters in form (a, d, p)
    
### Outputs: 
    - x: a sample from GG(a, d, p)
"""
function sample_generalized_gamma(pars)
    a, d, p = pars
    u = rand()
    # Use the inverse of the incomplete Gamma function
    y = gamma_inc_inv(d / p, u, 1 - u)
    # Apply transformation to obtain Generalized Gamma random variables
    x = a * y^(1 / p)
    return x
end

"""
    sample_W(n, pars, q1, Z0; no_extinction = true)

Sample realisations of W given pars, extinction probabilities. 

### Arguments: 
    - n: the number of samples to draw 
    - pars: list of pars (a, d, p) for each of the Wi
    - q1: vector of extinction probabilities for each Wi
    - Z0: the initial conditions for the branching process
    - no_extinction: (default = true) whether to condition on non-extinction
    
### Outputs: 
    - w: vector of samples of w
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

### Arguments: 
    - pars: list of pars (a, d, p) for each of the Wi
    - q1: vector of extinction probabilities for each Wi
    - Z0: the initial conditions for the branching process
    
### Outputs: 
    - w: samples of w
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
    - moments: an array of shape (5, number types) with the moments estimated using the methods 
               from section 3.3 of the paper
    - q1: vector of extinction probabilities starting with an individual of type i
    - num_moments_loss: (default = 5) number of moments to use in the expansion. Setting a 
                        default here lets us calculate more moments for other methods. 
    - iterations: (default = 10^5) max number of iterations to run in the optimisation.
    
### Outputs: 
    - pars: an array of parameters with length number of types corresponding to each Wi
"""
function minimise_loss(moments, q1)
    # get the number of loss functions to construct
    num_init_conds = size(moments, 2)

    pars = [zeros(Float64, 3) for _ in 1:num_init_conds]
    # Wide bounds on the parameters 
    lower_bd = [0.0, 0.0, 0.0]
    upper_bd = [50.0, 50.0, 50.0]
    # Initial guess is uninformative and reflects a boring distribution
    x0 = [1.0, 1.0, 1.0]

    # Wide bounds on the parameters 
    lower_bd = 1e-4 * ones(3)
    upper_bd = 50 * ones(3)
    # Initial guess is uninformative and reflects a boring distribution
    x0 = [1.0, 1.0, 1.0]

    inner_optimizer = BFGS()

    for i in eachindex(pars)
        l = x -> loss_func(x, moments[:, i], q1[i])
        ∇l = (g, x) -> ∇loss_func!(g, x, moments[:, i], q1[i])
        sol = Optim.optimize(l, ∇l, lower_bd, upper_bd, x0, Fminbox(inner_optimizer))
        pars[i] = sol.minimizer
    end

    return pars
end
