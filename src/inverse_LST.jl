"""
    preload_beta_eta(N_f_evals)
    
This loads the pre-computed coefficients from http://inverselaplace.org/. These
have been pre-calculated up to > 1000 and can be appropriately applied for a
range of problems. 

### Arguments: 
    - N_f_evals: the maximum number of function evals, this is set to 21 based on the paper 
                 that accompanies this method 
                  
### Outputs: 
    - The constants η and β needed for the LST inversion
"""
function load_cme_hyper_params(N_f_evals)
    cme_params = JSON.parsefile(joinpath(@__DIR__, "iltcme_ext.json"))
    params = cme_params[1]

    for p in cme_params
        if (p["cv2"] < params["cv2"]) && (p["n"] + 1 <= N_f_evals)
            params = p
        end
    end

    # Create complex eta and beta 
    η = params["eta_re"] + params["eta_im"] * im
    β = params["beta_re"] + params["beta_im"] * im

    return η, β
end

"""
    invert_lst(f, x, η, β)
    
Calculates the inverse laplace transform for the function f at the point(s) in x. This currently 
supports the concentrated-matrix-exponential method
which is expressed in the Abatte-Whitt framework. 

### Arguments: 
    - f: a function for the lst of W
    - x: the point to evaluate the lst at 
    - η: constants required for the function
    - β: constants required for the function
    
### Outputs: 
    - res: the cdf value
"""
function invert_lst(f, x, η, β)
    res = sum(e * f(b / x) for (e, b) in zip(η, β))
    res = real(res) / x

    return res
end

"""
    construct_W_cdf_ilst(lst_w, q_star; N_fevals = 21)

Constructs the LST of the CDF of W given the LST of the PDF. 

### Arguments: 
    - lst_w: function for computing the LST of W
    - q_star: the ultimate extinction probability
    - N_fevals: (default = 21) this is set at the default from the CME paper
                but can be adjusted. 
        
### Outputs: 
    - W_cdf: the CDF of W
"""
function construct_W_cdf_ilst(lst_w, q_star; N_fevals = 21)
    η, β = load_cme_hyper_params(N_fevals)
    function W_cdf(x)
        if x == 0
            return q_star
        else
            # Invert the LST (1 - ϕ(θ)) / θ to get the CCDF then take it from 1.
            return 1.0 - invert_lst(s -> (1 - lst_w(s)) / s, x, η, β)
        end
    end

    return W_cdf
end
