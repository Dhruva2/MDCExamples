using MinimallyDisruptiveCurves, MyModelMenagerie, Zygote, DiffEqSensitivity, QuadGK, Dierckx, JuMP, Ipopt

"""
To do:

see if gradients work better if jacobian is defined in the ODE of the model itself. (do this with modelingotolkit)
"""


function run_transformations(tr_array, od,ic,ps)
    nom_prob = ODEProblem(od,ic,tspan,ps, eval_expression=false) 
    for tr in tr_array
        od, ic, ps = transform_problem(nom_prob, tr; unames = first.(ic), pnames = first.(ps))
        nom_prob = ODEProblem(od,ic,tspan, ps, eval_expression=false)
    end 
    return od, ic, ps
end

function disp_mdc(mdc, ps)
    p1 = plot(mdc; pnames=first.(ps), legend=:bottom)
    cc = [mdc.cost(el) for el in eachcol(trajectory(mdc))];
    p2 = plot(distances(mdc), log.(cc), ylabel = "log(cost)", xlabel = "distance", title = "cost over MD curve");
    @show plot(p1,p2, layout=(2,1), size = (1000,500))
end



function l2_hessian(prob, output_map)

    dgdx = x -> ForwardDiff.jacobian(output_map, x)
    p0 = prob.p
    n = length(prob.u0)
    
    function f(du,u,p,t)
        du[1:n] .= prob.f(u[1:n],p0,t)
        du[n+1:2n] .= prob.f(u[n+1:2n],p,t)
        du[2n+1] = sum(abs2, output_map(u[1:n]) - output_map(u[n+1:2n]))
    end

    ol = ODELocalSensitivityProblem(f, vcat(prob.u0,prob.u0,0.), prob.tspan, prob.p)
    sens_sol = solve(ol, DP8(), reltol = 1e-10)
    x, dp = extract_local_sensitivities(sens_sol)
    x = x[n+1:2n, :]
    #dp[i] is matrix for ith paramater
    #we want int_t dyidpj dyi dpk
    o = size(dgdx(prob.u0))[1]
    q = length(prob.p)
    ts = sens_sol.t 
    
    [dp[i] = dp[i][n+1:2n, :] for i in 1:length(dp)]
    output_dp = [zeros(o, length(ts)) for el in dp]

    ## fill output_dp
    for (op, xp) in zip(output_dp, dp)
        for (ocol,xpcol, xcol) in zip(eachcol(op), eachcol(xp), eachcol(x))
            ocol[:] .= dgdx(xcol)*xpcol
        end
    end

    new_odp = vcat([reshape(output_dp[i], 1, :) for i in 1:q]...)
    hessian1 = new_odp*new_odp'
    # dyidpj(i,j) = output_dp[j][i,:]
    # dydp = [Spline1D(ts, dyidpj(i,j)) for i in 1:o, j in 1:q]
    # integrand = [t -> (dydp[i,j](t))'*dydp[i,k](t) for i in 1:n, j in 1:q, k in 1:q]
    hessian2 = zeros(q,q)
    # for j = 1:q
    #     for k = 1:q 
    #         for i = 1:o
    #             integ = t -> (dydp[i,j](t))'*dydp[i,k](t)
    #             hessian2[j,k] += quadgk(integ, ol.tspan[1], ol.tspan[end])[1]
    #         #    += dyidpj(i,j)'*dyidpj(i,k)
    #         end
    #     end
    # end
    return hessian1, hessian2
end


function sparse_init_dir(hessian; orthogonal_to = nothing, λ = 1., start = randn(size(hessian)[1]), trim_level = 1e-5)
    n = size(hessian)[1]
    model = Model(Ipopt.Optimizer)
    JuMP.set_silent(model)
    @variable(model, x[1:n])
    @variable(model, z[1:n])
    for i = 1:n
        set_start_value(x[i], start[i])
    end

    @objective(model, Min, x'*hessian*x + λ*sum(z))
    @constraint(model, x'*x == 1.)
    @constraint(model, z .>= x)
    @constraint(model, z .>= -x)
    if !(orthogonal_to === nothing)
        for el in orthogonal_to
            @constraint(model, x'*el == 0)
        end
    end

    JuMP.optimize!(model)
    out = value.(x)
    out[abs.(out) .< trim_level] .= 0
    val = out'*hessian*out
    return out, val
end