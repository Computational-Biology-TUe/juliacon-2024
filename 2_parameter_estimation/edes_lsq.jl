# Implementation of the EDeS model in Julia

using StaticArrays
using OrdinaryDiffEq
using CairoMakie
using Statistics
using QuasiMonteCarlo
using SimpleNonlinearSolve
using Random
using SciMLBase: NonlinearSolution
Random.seed!(5234)
# using BenchmarkTools

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

data_timepoints = 0:20:240


# simulate some data
sol = Array(solve(prob, Tsit5(), saveat=data_timepoints, save_idxs=[2, 3]))

# generate 10 samples of noisy data around the solution
data = cat([sol .+ randn(size(sol)...) .* mean(sol, dims=2) .* [0.05; 0.15] for _ in 1:10]..., dims=3)

# get the mean and standard deviation of the data
mean_data = mean(data, dims=3)[:,:,1]
std_data = std(data, dims=3)[:,:,1]

prob = remake(prob, u0=SA[0., mean_data[1,1], mean_data[2,1], 0.0])

# visualize the data
data_figure = let f = Figure(size=(500,200))
  ax_1 = Axis(f[1,1], xlabel="Time [min]", ylabel="Concentration [mM]", title="Glucose")
  scatter!(ax_1, data_timepoints, mean_data[1,:], label="Gpl data", color=Makie.wong_colors()[1], markersize=25, marker='∘')
  errorbars!(ax_1, data_timepoints, mean_data[1,:], std_data[1,:], color=Makie.wong_colors()[1], whiskerwidth=8)
  ax_2 = Axis(f[1,2], xlabel="Time [min]", ylabel="Concentration [mU/L]", title="Insulin")
  scatter!(ax_2, data_timepoints, mean_data[2,:], label="Ipl data", color=Makie.wong_colors()[2], markersize=25, marker='∘')
  errorbars!(ax_2, data_timepoints, mean_data[2,:], std_data[2,:], color=Makie.wong_colors()[2], whiskerwidth=8)
  f
end

function construct_parameters(θ, c)

  # Estimated parameters
  k1 = θ[1]
  k5 = θ[2]
  k6 = θ[3]
  k8 = θ[4]
  
  # Fixed parameters
  k2 = c[1]
  k3 = c[2]
  k4 = c[3]
  k7 = c[4]
  k9 = c[5]
  k10 = c[6]
  tau_i = c[7]
  tau_d = c[8]
  beta = c[9]
  Gren = c[10]
  EGPb = c[11]
  Km = c[12]
  f = c[13]
  Vg = c[14]
  c1 = c[15]
  sigma = c[16]
  Dmeal = c[17]
  bw = c[18]
  Gb = c[19]
  Ib = c[20]
  return SA[k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, tau_i, tau_d, beta, Gren, EGPb, Km, f, Vg, c1, sigma, Dmeal, bw, Gb, Ib]
end

# define the loss function
function loss(θ, p)
  problem = p[1]
  glucose_data = p[2]
  glucose_std = p[3]
  insulin_data = p[4]
  insulin_std = p[5]
  data_timepoints = p[6]
  constants = p[7]

  # update the parameters
  p = construct_parameters(θ, constants)

  # solve the problem
  pred = solve(problem, Tsit5(), p=p, saveat=data_timepoints, save_idxs=[2, 3])
  sol = Array(pred)
  vec(reduce(hcat, (sol .- [glucose_data'; insulin_data']) ./ [glucose_std'; insulin_std']))
end

constants = [k2, k3, k4, k7, k9, k10, tau_i, tau_d, beta, Gren, EGPb, Km, f, Vg, c1, sigma, Dmeal, bw, mean_data[1,1], mean_data[2,1]]

# compute initial guess
initial_guess = QuasiMonteCarlo.sample(
  100_000, [0., 0., 0., 0.], [1., 0.5, 20., 20.], LatinHypercubeSample()
)

# compute the loss for the initial guess
initial_loss_values = [sum(abs2, loss(guess, (
  prob, mean_data[1,:], std_data[1,:], mean_data[2,:], std_data[2,:], data_timepoints, constants
))) for guess in eachcol(initial_guess)]

# sort the initial guess by loss
sorted_guesses = initial_guess[:,sortperm(initial_loss_values)[1:20]]

results = NonlinearSolution[]
for guess in eachcol(sorted_guesses)
  try
    res = solve(NonlinearLeastSquaresProblem(loss, vec(guess), (
      prob, mean_data[1,:], std_data[1,:], mean_data[2,:], std_data[2,:], data_timepoints, constants
    )), SimpleGaussNewton(); maxiters=100)
    push!(results, res)
  catch
    continue
  end
end

results

# get the estimated parameters
estimated_parameters = construct_parameters(results[1].u, constants)

# solve the problem with the estimated parameters
estimated_solution = solve(prob, Tsit5(), p=estimated_parameters, saveat=data_timepoints, save_idxs=[2, 3])

# visualize the estimated solution
# visualize the data
solution_figure = let f = Figure(size=(500,200))
  ax_1 = Axis(f[1,1], xlabel="Time [min]", ylabel="Concentration [mM]", title="Glucose")
  scatter!(ax_1, data_timepoints, mean_data[1,:], label="Gpl data", color=Makie.wong_colors()[1], markersize=25, marker='∘')
  errorbars!(ax_1, data_timepoints, mean_data[1,:], std_data[1,:], color=Makie.wong_colors()[1], whiskerwidth=8)
  ax_2 = Axis(f[1,2], xlabel="Time [min]", ylabel="Concentration [mU/L]", title="Insulin")
  scatter!(ax_2, data_timepoints, mean_data[2,:], label="Ipl data", color=Makie.wong_colors()[2], markersize=25, marker='∘')
  errorbars!(ax_2, data_timepoints, mean_data[2,:], std_data[2,:], color=Makie.wong_colors()[2], whiskerwidth=8)

  lines!(ax_1, estimated_solution.t, estimated_solution[1,:], color=Makie.wong_colors()[1], label="Gpl estimated")
  lines!(ax_2, estimated_solution.t, estimated_solution[2,:], color=Makie.wong_colors()[2], label="Ipl estimated")
  f
end