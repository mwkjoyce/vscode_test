module CompDiff

export periodic_laplacian!, logistic!

function periodic_laplacian!(x::Matrix{Float64})
    len = size(x)[1]
    u = zeros(len, len)
    
    for i in 1:len
        for j in 1:len
            # periodic boundary conditions
            t_n = x[mod1(i - 1, len), j]
            b_n = x[mod1(i + 1, len), j]
            
            r_n = x[i, mod1(j - 1, len)]
            l_n = x[i, mod1(j + 1, len)]
            
            u[i, j] = 0.25 * (t_n + b_n + r_n + l_n) - x[i, j]
        end
    end
    return u
end

function logistic!(u::Matrix{Float64}, v::Matrix{Float64})
    #=
    the version of the logistic growth model used is nondimensionalised, and uses weights from my lecture notes
    the commented-out version 'over fits' very quickly and its hard to observe oscilliatory behaviour
    =#
    a = 0.85
    d = 0.1
    b = 0.1
    β = 4.285
    δ = 5
    k = 1
    λ = 3.428
    γ = 0.1

    len = size(u)[1]
    du, dv = zeros(len, len), zeros(len, len)

    du = @. u * (1 - u) - a * u * v / (u + d)
    dv = @. b * v * (1 - v/u) 
    #du = @. β * u * (1 - u) - δ * v * u^2 / (u^2 + k)
    #dv = @. γ * v * (δ * u^2 / (u^2 + k) - λ)
    return du, dv
end
end

using Plots
using ProgressMeter
using .Utilities

function update!(p::Matrix{Float64}, z::Matrix{Float64})
    dp_diff = runge_kutta(p, periodic_laplacian!)
    dz_diff = runge_kutta(z, periodic_laplacian!)
    p .+= dp_diff
    z .+= dz_diff
    dp_source, dz_source = runge_kutta(p, z, logistic!)
    p .+= dp_source
    z .+= dz_source
    return p, z
end

dim = 200
dt = 0.1

parr = zeros(dim, dim)
zarr = zeros(dim, dim)

init!(zarr)
init_slope!(parr; a=false)


p_count = []
z_count = []

anim_len = 500

p = Progress(anim_len)

@gif for i=1:anim_len
    p1 = surface(parr, cbar=false, c=cgrad(:amp))
    p2 = surface(zarr, cbar=false, c=cgrad(:Blues))

    # plot average density of phyto and zoo
    p3 = plot([p_count./(dim.^2), z_count./(dim.^2)], label = ["phyto" "zoo"])
    xaxis!(p3, (0, anim_len))
    yaxis!(p3, (0, 1))
    zaxis!(p1, (0, 1))
    zaxis!(p2, (0, 1))
    plot(p1, p2, p3, layout=(2, 2), size=(800, 800))
    push!(p_count, sum(parr))
    push!(z_count, sum(zarr))

    # becuase dt != 1, run the simulation a few times before
    # inbetween each frame so it moves at a decent pace
    for i in 1:10
        update!(parr, zarr)
    end
    next!(p)
end fps=60

