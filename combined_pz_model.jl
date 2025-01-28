using Plots, ProgressMeter

using .Utilities

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
    the version of the logistic growth model used is nonnensionalised, and uses weights from my lecture notes
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

function literature_fluid_field(X, Y, L, n; α=0, F=0.1)
    uy = @. F * (α * sin(2 * π * Y / L) + (1 - α) * sin(2 * π * X / L))
    ux = @. F * (α * sin(2 * π * X / L) + (1 - α) * sin(2 * π * Y / L))

    uy_grid = transpose(reshape(uy, (n, n)))
    ux_grid = transpose(reshape(ux, (n, n)))

    # ux and uy are vectors n*n long, wanted if doing a quiver plots,
    return ux_grid, uy_grid, ux, uy
end

function advect(P::T) where {T<:Matrix{Float64}}
    # dont update the input, but return matrices to add the the input later
    len = size(P)[1]
    dxP, dyP = zeros(len, len), zeros(len, len)
    uxdxP, uydyP = zeros(len, len), zeros(len, len)

    for i in 1:len
        for j in 1:len
            #periodic boundary conditions, with two steps forwards/backwards also available
            t_n = P[mod1(i - 1, len), j]
            b_n = P[mod1(i + 1, len), j]
            t_2n = P[mod1(i - 2, len), j]
            b_2n = P[mod1(i + 2, len), j]  
            
            r_n = P[i, mod1(j - 1, len)]
            l_n = P[i, mod1(j + 1, len)]
            r_2n = P[i, mod1(j - 2, len)]
            l_2n = P[i, mod1(j + 2, len)]
            
            # second order upwind scheme
            # gives by far the best stability
            # the 0.1 is the grid spacing = L/n
            if ux_grid[i,j] >= 0
                dxP[i,j] = (-r_2n + 4r_n - 3P[i,j])/(2*0.1)
            else
                dxP[i,j] = (3P[i,j] - 4l_n + l_2n)/(2*0.1)
            end
            if uy_grid[i,j] >= 0
                dyP[i,j] = (-t_2n + 4t_n - 3P[i,j])/(2*0.1)
            else
                dyP[i,j] = (3P[i,j] - 4b_n + b_2n)/(2*0.1)
            end
        end
    end
    # the -ve makes it work, i think it has to do with how u⁻ combines a⁺ (wikipedia)
    uxdxP = -ux_grid .* dxP
    uydyP = -uy_grid .* dyP

    return uxdxP .+ uydyP
end

function update!(p::T, z::T) where {T<:Matrix{Float64}}
    #=
    Apply advection, diffusion, then growth
    =#
    dt = 0.1
    uxP, uyZ = dt.*advect(p), dt.*advect(z)
    p .-= uxP
    z .-= uyZ

    dp_diff = runge_kutta(p, periodic_laplacian!, dt)
    dz_diff = runge_kutta(z, periodic_laplacian!, dt)
    p .+= dp_diff
    z .+= dz_diff

    dp_source, dz_source = runge_kutta(p, z, logistic!, dt)
    p .+= dp_source
    z .+= dz_source
    return p, z
end

L = 2
n = 100L
dt = 0.1

x_vals = range(-L/4, 3L/4, n)
y_vals = range(-L/4, 3L/4, n)
X, Y = meshgrid(x_vals, y_vals)

ux_grid, uy_grid, _ = literature_fluid_field(X, Y, L, n)
 
phyto = zeros(n, n)
zoo = zeros(n, n)
init_slope!(phyto; a=false)
init!(zoo)

p_count = []
z_count = []

anim_len = 800
it_per_frame = 10
plot_after = 500

prog = Progress(anim_len)

@gif for i=1:anim_len
    title = plot(title = "Frame: $i/$anim_len", grid = false, showaxis = false, bottom_margin = -50Plots.px)
    p1 = surface(phyto, cbar=false, c=cgrad(:GnBu, scale=:exp, rev=true), xaxis=nothing, yaxis=nothing)
    p2 = surface(zoo, cbar=false, c=cgrad(:amp, scale=:exp, rev=true), xaxis=nothing, yaxis=nothing)
    #p1 = heatmap(phyto, cbar=false, c=cgrad(:GnBu, scale=:exp, rev=true))
    #p2 = heatmap(zoo, cbar=false, c=cgrad(:amp, scale=:exp, rev=true)) 
    zaxis!(p1, (0,1))
    zaxis!(p2, (0,1))

    # plot the average density
    push!(p_count, sum(phyto./(n.^2)))
    push!(z_count, sum(zoo./(n.^2)))
    p3 = plot([p_count, z_count], label=["Phyto" "Zoo"])
    xaxis!(p3, (plot_after, anim_len))
    yaxis!(p3, (0, 1))

    plot(title, p1, p2, p3, layout=@layout([A{0.01h}; [B C]; D]), size=(700, 700))

    for j in 1:it_per_frame
        update!(phyto, zoo)
    end
    next!(prog)
end when i >= plot_after

