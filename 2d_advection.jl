using Plots
using ProgressMeter

meshgrid(x, y) = (repeat(x, outer=length(y)), repeat(y, inner=length(x)))

function literature_fluid_field(X, Y; L=1, n=20, α=0, F=1)
    uy = @. F * (α * sin(2 * π * Y / L) + (1 - α) * sin(2 * π * X / L)) 
    ux = @. F * (α * sin(2 * π * X / L) + (1 - α) * sin(2 * π * Y / L)) 

    ux_grid = transpose(reshape(ux, (n, n)))
    uy_grid = transpose(reshape(uy, (n, n)))
    return ux_grid, uy_grid, ux, uy
end

function advect(P::T) where {T<:Matrix{Float64}}
    len = size(P)[1]
    dxP, dyP = zeros(len, len), zeros(len, len)
    uxdxP, uydyP = zeros(len, len), zeros(len, len)

    for i in 1:len
        for j in 1:len
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
    # the -ve makes it work, i think it has to do with how 
    # u⁻ wants a⁺
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

function periodic_laplacian!(x::Matrix{Float64})
    len = size(x)[1]
    u = zeros(len, len)
    
    for i in 1:len
        for j in 1:len
            t_n = x[mod1(i - 1, len), j]
            b_n = x[mod1(i + 1, len), j]
            
            r_n = x[i, mod1(j - 1, len)]
            l_n = x[i, mod1(j + 1, len)]
            
            u[i, j] = 0.25 * (t_n + b_n + r_n + l_n) - x[i, j]
        end
    end
    return u
end

function runge_kutta(x, func)
    k1 = func(x)
    k2 = func(x .+ dt.*k1./2)
    k3 = func(x .+ dt.*k2./2)
    k4 = func(x .+ dt.*k3)

    xn_plus_1 = @. dt*(k1 + 2k2 + 2k3 + k4)
    return xn_plus_1
end

function update!(x::Matrix{Float64})
    #udP = runge_kutta(x, advect)
    udP = dt.*advect(x)
    x .-= udP
    #dp_diff = runge_kutta(x, periodic_laplacian!)
    #x .+= dp_diff
    return x
end

L = 2
n = 20L
dt = 0.001

x_vals = range(-L/4, 3L/4, n)
y_vals = range(-L/4, 3L/4, n)
X, Y = meshgrid(x_vals, y_vals)

ux_grid, uy_grid, ux, uy = literature_fluid_field(X, Y;L, n)

phyto = zeros(n, n)
init!(phyto)


u = zeros(n, n)
for i in 1:n
    for j in 1:n
        u[i,j] = sqrt(uy_grid[i,j]^2 + ux_grid[i,j]^2)
        #u[i,j] = uy_grid[i, j] + ux_grid[i,j]
    end
end

anim_len = 200

prog = Progress(anim_len)

phyto_init = copy(phyto)

@gif for i=1:anim_len
    p2 = surface(phyto, cbar=false, camera=(45, 55), title="Frame: $i")
    zaxis!(p2, (0, 1))

    plot(p2, size=(600, 600))

    for i in 1:100
        update!(phyto)
    end
    next!(prog)
end
