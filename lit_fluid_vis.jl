using Plots

using .Utilities

α = 0
F = 1.0
L = 1

x_vals = range(-1L/4, 3L/4, 20)
y_vals = range(-1L/4, 3L/4, 20)
X, Y = meshgrid(x_vals, y_vals)

ux = @. F * (α * sin(2 * π * X / L) + (1 - α) * sin(2 * π * Y / L)) + eps(Float64)
uy = @. F * (α * sin(2 * π * Y / L) + (1 - α) * sin(2 * π * X / L)) + eps(Float64)

scale = 35/1000 * L

quiver(X, Y, quiver=(scale*ux, scale*uy), size=(500, 500), dpi=400)