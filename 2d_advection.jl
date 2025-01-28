using Plots
using ProgressMeter

meshgrid(x, y) = (repeat(x, outer=length(y)), repeat(y, inner=length(x)))

function literature_fluid_field(X, Y, L=1, n=20; α=0, F=1)
    uy = @. F * (α * sin(2 * π * Y / L) + (1 - α) * sin(2 * π * X / L))
    ux = @. F * (α * sin(2 * π * X / L) + (1 - α) * sin(2 * π * Y / L))

    ux_grid = transpose(reshape(ux, (n, n)))
    uy_grid = transpose(reshape(uy, (n, n)))

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

function init!(x::Matrix{Float64})
    len = size(x)[1]
    a = ceil(len/2)
    b = 10
    for i in 1:len
        for j in 1:len
            x[i, j] = 5exp(-0.5((i/b-a/b)^2 + (j/b-a/b)^2))/b
        end
    end
    return x
end

function update_advec!(x::Matrix{Float64})
    udP = dt.*advect(x)
    x .-= udP
    #dp_diff = runge_kutta(x, periodic_laplacian!)
    #x .+= dp_diff
    return x
end

L = 40
n = L
dt = 0.001

# plotting on the range of -L/4 -> 3L/4 visualises the two vortices (for α = 0) nicely
x_vals = range(-L/4, 3L/4, n)
y_vals = range(-L/4, 3L/4, n)
X, Y = meshgrid(x_vals, y_vals)

ux_grid, uy_grid, _ = literature_fluid_field(X, Y, L, n)

# initialise something to be advected 
phyto = zeros(n, n)
init!(phyto)

anim_len = 200

# create a visual progress meter in the console
prog = Progress(anim_len)

@gif for i=1:anim_len
    # plotting a surface or a heatmap are nice
    it = i*100
    p1 = surface(phyto, cbar=false, c=cgrad(:GnBu, scale=:exp, rev=true), camera=(45, 55), title="Frame: $i , Iteration: $it")
    zaxis!(p1, (0, 1))

    plot(p1, size=(600, 600))

    # run the simulation a few times inbetween plotting to speed up the animation and avoid plotting too much, especially with surface plots
    # using a lower dt value reduces the number of time iterations you need
    for j in 1:100
        update_advec!(phyto)
    end

    #update the progress meter
    next!(prog)
end
