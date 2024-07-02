# Implementation of the EDeS model in Julia

# The staticarray implementation yields a 5x speedup over the standard array implementation and a 1.5x speedup
# over the mutating array implementation.

# StaticArray implementation:
# BenchmarkTools.Trial: 10000 samples with 1 evaluation.
#  Range (min … max):  220.083 μs …   6.390 ms  ┊ GC (min … max): 0.00% … 95.37%
#  Time  (median):     232.583 μs               ┊ GC (median):    0.00%
#  Time  (mean ± σ):   243.159 μs ± 153.544 μs  ┊ GC (mean ± σ):  3.46% ±  5.34%

#           ▁▃▇█▆▇▇▆▄▃▂▂                                           
#   ▂▂▃▄▅▇▇▇█████████████▇█▇▆▆▆▆▅▅▄▄▄▄▃▃▃▃▂▃▂▂▂▂▂▁▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁ ▄
#   220 μs           Histogram: frequency by time          268 μs <

#  Memory estimate: 168.83 KiB, allocs estimate: 27.

# Standard array implementation:
# BenchmarkTools.Trial: 4264 samples with 1 evaluation.
#  Range (min … max):  1.061 ms …   2.863 ms  ┊ GC (min … max): 0.00% … 60.87%
#  Time  (median):     1.131 ms               ┊ GC (median):    0.00%
#  Time  (mean ± σ):   1.171 ms ± 170.121 μs  ┊ GC (mean ± σ):  3.88% ±  9.02%

#   ▅█▆▆▆▆▇▇▆▄▁▁                       ▁                        ▂
#   ████████████▇█▃▃▁▄▁▁▁▃▁▁▃▁▁▁▁▁▁▄██▇████▇▆█▇▆▇▆█▆▇▆▇▅▅▆▅▇▆▇▇ █
#   1.06 ms      Histogram: log(frequency) by time      1.87 ms <

#  Memory estimate: 3.77 MiB, allocs estimate: 39975.

# Mutating array implementation:
# BenchmarkTools.Trial: 10000 samples with 1 evaluation.
#  Range (min … max):  349.792 μs …  4.350 ms  ┊ GC (min … max): 0.00% … 90.94%
#  Time  (median):     364.000 μs              ┊ GC (median):    0.00%
#  Time  (mean ± σ):   374.015 μs ± 79.879 μs  ┊ GC (mean ± σ):  1.39% ±  5.28%

#     ██▄▂▁                                                       
#   ▂▇███████▇▇▅▄▄▄▃▃▂▂▂▂▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁ ▂
#   350 μs          Histogram: frequency by time          483 μs <

#  Memory estimate: 339.23 KiB, allocs estimate: 2455.

using StaticArrays
using OrdinaryDiffEq
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

function edesode(u, p, t)
  Ggut, Gpl, Ipl, Irem = u
  k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, tau_i, tau_d, beta, Gren, EGPb, Km, f, Vg, c1, sigma, Dmeal, bw, Gb, Ib = p

  # gut glucose
  dGgut = sigma * k1^sigma * t^(sigma-1) * exp(-(k1*t)^sigma) * Dmeal - k2 * Ggut

  # plasma glucose
  gliv = EGPb - k3 * (Gpl - Gb) - k4 * beta * Irem
  ggut = k2 * (f / (Vg * bw)) * Ggut
  u_ii =  EGPb * ((Km + Gb)/Gb) * (Gpl / (Km + Gpl))
  u_id = k5 * beta * Irem * (Gpl / (Km + Gpl))
  u_ren = c1 / (Vg * bw) * (Gpl - Gren) * (Gpl > Gren)

  dGpl  = gliv + ggut - u_ii - u_id - u_ren

  # plasma insulin
  i_pnc = beta^(-1) * (k6 * (Gpl - Gb) + (k7 / tau_i) * Gb + k8 * tau_d * dGpl)
  i_liv = k7 * Gb * Ipl / (beta * tau_i * Ib)
  i_int = k9 * (Ipl - Ib)
  dIpl = i_pnc - i_liv - i_int
  dIrem = i_int - k10 * Irem

  return SA[dGgut, dGpl, dIpl, dIrem]
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
sigma = 1.4

# physiology specific parameters
bw = 70.0
Gb = 5.0
Ib = 10.0

# meal parameters
Dmeal = 75.0e3

# collection of parameters
p = SA[k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, tau_i, tau_d, beta, Gren, EGPb, Km, f, Vg, c1, sigma, Dmeal, bw, Gb, Ib]

# set initial conditions
u0 = SA[0.0, Gb, Ib, 0.0]

# set time span
tspan = (0.0, 240.0)

# define the problem
prob = ODEProblem(edesode, u0, tspan, p)

# solve the problem
@benchmark solve($prob, Tsit5(), saveat=0.1)

# # visualize the solution
# solution_figure = let f = Figure(size=(500,500))

#   ax_g_gut = Axis(f[1,1], xlabel="Time [min]", ylabel="Glucose Mass [mg/dL]", title="Gut Glucose")
#   ax_g_plasma = Axis(f[1,2], xlabel="Time [min]", ylabel="Glucose Concentration [mM]", title="Plasma Glucose")
#   ax_i_plasma = Axis(f[2,1], xlabel="Time [min]", ylabel="Insulin Concentration [mU/L]", title="Plasma Insulin")
#   ax_i_int = Axis(f[2,2], xlabel="Time [min]", ylabel="Insulin Concentration [mU/L]", title="Interstitium Insulin")

#   lines!(ax_g_gut, sol.t, sol[1,:], color=Makie.wong_colors()[1])
#   lines!(ax_g_plasma, sol.t, sol[2,:], color=Makie.wong_colors()[2])
#   lines!(ax_i_plasma, sol.t, sol[3,:], color=Makie.wong_colors()[3])
#   lines!(ax_i_int, sol.t, sol[4,:], color=Makie.wong_colors()[4])


#   f
# end