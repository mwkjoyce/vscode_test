module Utilities

export init!, meshgrid, init_slope!, runge_kutta

meshgrid(x, y) = (repeat(x, outer=length(y)), repeat(y, inner=length(x)))

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

function init_slope!(x::Matrix{Float64}; a=true)
    len = size(x)[1]
    for i in 1:len
        for j in 1:len
            if a
                x[i,j] = (i + j)/200
            else 
                x[i,j] = 0.5
            end
        end
    end
    return x
end

function runge_kutta(x, y, func)
    #=
    normal eulerian method builds error very quickly
    so use runge-kutta 4 instead
    yₙ₊₁ = yₙ + h/6 * (k₁ + 2k₂ + 2k₃ + k₄)
    =#
    k1x, k1y = func(x, y)
    k2x, k2y = func(x .+ dt.*k1x./2, y .+ dt.*k1y./2)
    k3x, k3y = func(x .+ dt.*k2x./2, y .+ dt.*k2y./2)
    k4x, k4y = func(x .+ dt.*k3x, y .+ dt.*k3y)

    xn_plus_1 = @. dt*(k1x + 2k2x + 2k3x + k4x)/6
    yn_plus_1 = @. dt*(k1y + 2k2y + 2k3y + k4y)/6

    return xn_plus_1, yn_plus_1
end

function runge_kutta(x, func)
    k1 = func(x)
    k2 = func(x .+ dt.*k1./2)
    k3 = func(x .+ dt.*k2./2)
    k4 = func(x .+ dt.*k3)

    xn_plus_1 = @. dt*(k1 + 2k2 + 2k3 + k4)
    return xn_plus_1
end
end