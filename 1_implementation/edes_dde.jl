# Implementation of the EDeS model in Julia

# The staticarray implementation yields a 5x speedup over the standard array implementation and a 1.5x speedup
# over the mutating array implementation.

# StaticArray implementation:
# BenchmarkTools.Trial: 10000 samples with 1 evaluation.
#  Range (min … max):  286.875 μs …   6.404 ms  ┊ GC (min … max): 0.00% … 94.82%
#  Time  (median):     302.459 μs               ┊ GC (median):    0.00%
#  Time  (mean ± σ):   312.947 μs ± 197.600 μs  ┊ GC (mean ± σ):  2.87% ±  4.31%

#           ▁▃▃▅▆▇█▇▆▅▇▅▅▅▃▃▂▃▂▂▂▂                                 
#   ▁▃▅▇██▇█████████████████████████▇▇▇▆▆▆▅▅▄▄▃▃▃▃▂▂▂▂▂▁▂▁▁▁▁▁▁▁▁ ▅
#   287 μs           Histogram: frequency by time          335 μs <

#  Memory estimate: 209.14 KiB, allocs estimate: 101.

# Standard array implementation:
# BenchmarkTools.Trial: 2862 samples with 1 evaluation.
#  Range (min … max):  1.390 ms …   7.215 ms  ┊ GC (min … max):  0.00% … 79.23%
#  Time  (median):     1.513 ms               ┊ GC (median):     0.00%
#  Time  (mean ± σ):   1.746 ms ± 915.833 μs  ┊ GC (mean ± σ):  10.63% ± 14.68%

#   █▇▆▇▅                                                       ▁
#   ██████▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▃▆▁▆▆▆▆█▆▆▇▇▆ █
#   1.39 ms      Histogram: log(frequency) by time      6.42 ms <

#  Memory estimate: 3.74 MiB, allocs estimate: 39621.

# Mutating array implementation:
# BenchmarkTools.Trial: 10000 samples with 1 evaluation.
#  Range (min … max):  449.375 μs …   5.355 ms  ┊ GC (min … max): 0.00% … 89.82%
#  Time  (median):     471.250 μs               ┊ GC (median):    0.00%
#  Time  (mean ± σ):   489.068 μs ± 227.343 μs  ┊ GC (mean ± σ):  2.86% ±  5.48%

#     ▂██▆▅▄▃▂▂▂▁▂▂▂▂▁                                             
#   ▁▄███████████████████▆▆▆▅▅▅▄▄▄▃▃▃▃▃▂▂▂▂▂▂▂▂▂▂▁▁▁▂▁▁▁▁▁▁▁▁▁▁▁▁ ▄
#   449 μs           Histogram: frequency by time          551 μs <

#  Memory estimate: 384.52 KiB, allocs estimate: 2919.

using StaticArrays
using DelayDiffEq
using CairoMakie
using BenchmarkTools

# figure theme
set_theme!(theme_minimal())
update_theme!(
  fonts = 
  (regular = "Arial", bold = "Arial bold", italic = "Arial italic"),
  Lines = (
    linewidth = 3.0,
    linestyle = :solid),
  Axis = (
    backgroundcolor = :transparent,
    topspinevisible = false,
    rightspinevisible = false,
    titlesize = 14,
    ticklabelsize = 12,
    xlabelsize = 12,
    xlabelfont = :bold,
    ylabelfont = :bold,
    ylabelsize = 12
  ),
)

function configure_history(Gb)
  h(p, t; idxs = nothing) = typeof(idxs) <: Number ? Gb : ones(5).*Gb
  return h
end

function edesdde(u, h, p, t)
  Ggut, Gpl, Gint, Ipl, Irem = u
  k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, tau_i, tau_d, beta, Gren, EGPb, Km, f, Vg, c1, t_int, sigma, Dmeal, bw, Gb, Ib = p

  Ghist = h(p, t - t_int; idxs=2)

  # gut glucose
  dGgut = sigma * k1^sigma * t^(sigma-1) * exp(-(k1*t)^sigma) * Dmeal - k2 * Ggut

  # plasma glucose
  gliv = EGPb - k3 * (Gpl - Gb) - k4 * beta * Irem
  ggut = k2 * (f / (Vg * bw)) * Ggut
  u_ii =  EGPb * ((Km + Gb)/Gb) * (Gpl / (Km + Gpl))
  u_id = k5 * beta * Irem * (Gpl / (Km + Gpl))
  u_ren = c1 / (Vg * bw) * (Gpl - Gren) * (Gpl > Gren)

  dGpl  = gliv + ggut - u_ii - u_id - u_ren
  dGint = Gpl - Ghist

  # plasma insulin
  i_pnc = beta^(-1) * (k6 * (Gpl - Gb) + (k7 / tau_i) * (Gint + Gb) + k8 * tau_d * dGpl)
  i_liv = k7 * Gb * Ipl / (beta * tau_i * Ib)
  i_int = k9 * (Ipl - Ib)
  dIpl = i_pnc - i_liv - i_int
  dIrem = i_int - k10 * Irem

  return SA[dGgut, dGpl, dGint, dIpl, dIrem]
end

# set parameters
k1 = 0.0105
k2 = 0.28
k3 = 6.07e-3
k4 = 2.35e-4
k5 = 0.0424
k6 = 2.2975
k7 = 1.15
k8 = 7.27
k9 = 3.83e-2
k10 = 2.84e-1
tau_i = 31.0
tau_d = 3.0
beta = 1.0
Gren = 9.0
EGPb = 0.043
Km = 13.2
f = 0.005551
Vg = 17.0 / 70.0
c1 = 0.1
t_int = 30.0
sigma = 1.4

# physiology specific parameters
bw = 70.0
Gb = 5.0
Ib = 10.0

# meal parameters
Dmeal = 75.0e3

# collection of parameters
p = SA[k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, tau_i, tau_d, beta, Gren, EGPb, Km, f, Vg, c1, t_int, sigma, Dmeal, bw, Gb, Ib]

# set initial conditions
u0 = SA[0.0, Gb, Gb, Ib, 0.0]

# set time span
tspan = (0.0, 240.0)

# initialize the history function
h = configure_history(Gb)

# define the problem
prob = DDEProblem(edesdde, u0, h, tspan, p)

# solve the problem
sol = solve(prob, MethodOfSteps(Tsit5()), saveat=0.1)

# visualize the solution
solution_figure = let f = Figure(size=(500,500))

  ax_g_gut = Axis(f[1,1], xlabel="Time [min]", ylabel="Glucose Mass [mg/dL]", title="Gut Glucose")
  ax_g_plasma = Axis(f[1,2], xlabel="Time [min]", ylabel="Glucose Concentration [mM]", title="Plasma Glucose")
  ax_i_plasma = Axis(f[2,1], xlabel="Time [min]", ylabel="Insulin Concentration [mU/L]", title="Plasma Insulin")
  ax_i_int = Axis(f[2,2], xlabel="Time [min]", ylabel="Insulin Concentration [mU/L]", title="Interstitium Insulin")

  lines!(ax_g_gut, sol.t, sol[1,:], color=Makie.wong_colors()[1])
  lines!(ax_g_plasma, sol.t, sol[2,:], color=Makie.wong_colors()[2])
  lines!(ax_i_plasma, sol.t, sol[4,:], color=Makie.wong_colors()[3])
  lines!(ax_i_int, sol.t, sol[5,:], color=Makie.wong_colors()[4])


  f
end