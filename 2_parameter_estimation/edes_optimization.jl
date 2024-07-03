# Implementation of the EDeS model in Julia

using StaticArrays
using OrdinaryDiffEq
using CairoMakie
using Statistics
using QuasiMonteCarlo
using Optimization, OptimizationOptimJL
using Random
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
  sum(abs2, (sol .- [glucose_data'; insulin_data']) ./ [glucose_std'; insulin_std']), solve(problem, Tsit5(), p=p, saveat=1, save_idxs=[2, 3])
end

constants = [k2, k3, k4, k7, k9, k10, tau_i, tau_d, beta, Gren, EGPb, Km, f, Vg, c1, sigma, Dmeal, bw, mean_data[1,1], mean_data[2,1]]

# compute initial guess
initial_guess = QuasiMonteCarlo.sample(
  10_000, [0., 0., 0., 0.], [1., 0.5, 20., 20.], LatinHypercubeSample()
)

# compute the loss for the initial guess
initial_loss_values = [loss(guess, (
  prob, mean_data[1,:], std_data[1,:], mean_data[2,:], std_data[2,:], data_timepoints, constants
))[1] for guess in eachcol(initial_guess)]

# sort the initial guess by loss
sorted_guesses = initial_guess[:,sortperm(initial_loss_values)[50]]

optfunc = OptimizationFunction(loss, Optimization.AutoForwardDiff())
optprob = OptimizationProblem(optfunc, sorted_guesses, (
  prob, mean_data[1,:], std_data[1,:], mean_data[2,:], std_data[2,:], data_timepoints, constants
),lb = [0., 0., 0., 0.], ub = [1., 0.5, 20., 20.])


function create_callback(predictions)

  callback = function (state, l, pred)
    push!(predictions, Array(pred))
    false
  end

  callback
end

# starting point
p_start = construct_parameters(sorted_guesses, constants)

# solve the problem
pred_start = Array(solve(prob, Tsit5(), p=p_start, saveat=1, save_idxs=[2, 3]))

# initialize the predictions
predictions_LBFGS = Matrix{Float64}[pred_start]
predictions_BFGS = Matrix{Float64}[pred_start]
predictions_NelderMead = Matrix{Float64}[pred_start]
predictions_ParticleSwarm = Matrix{Float64}[pred_start]
predictions = [
  predictions_NelderMead,
  predictions_ParticleSwarm,
  predictions_BFGS,
  predictions_LBFGS
]

# select optimization algorithms
optalgs = [
  NelderMead(),
  ParticleSwarm(),
  BFGS(),
  LBFGS()
]

optsols = [solve(optprob, alg, callback=create_callback(preds)) for (alg, preds) in zip(optalgs, predictions)]

step = Observable(1)
predictions[1][1]
xs = 0:1:240
ys_1 = [@lift(pred[$step <= length(pred) ? $step : length(pred)][1,:]) for pred in predictions]
ys_2 = [@lift(pred[$step <= length(pred) ? $step : length(pred)][2,:]) for pred in predictions]


optim_progress_fig = let f = Figure(size=(600,300))
  ax_1 = Axis(f[1,1], xlabel="Time [min]", ylabel="Concentration [mM]", title=@lift "Glucose (step: $($step))"   )
  scatter!(ax_1, data_timepoints, mean_data[1,:], label="Data", color=:black, markersize=25, marker='∘')
  errorbars!(ax_1, data_timepoints, mean_data[1,:], std_data[1,:], color=:black, whiskerwidth=8, label="Data")
  ax_2 = Axis(f[1,2], xlabel="Time [min]", ylabel="Concentration [mU/L]", title=@lift "Insulin (step: $($step))")
  scatter!(ax_2, data_timepoints, mean_data[2,:], label="Ipl data", color=:black, markersize=25, marker='∘')
  errorbars!(ax_2, data_timepoints, mean_data[2,:], std_data[2,:], color=:black, whiskerwidth=8)

  for (i, algname) in enumerate(["Nelder-Mead", "Particle Swarm", "BFGS", "LBFGS"])
    lines!(ax_1, xs, ys_1[i], color=Makie.wong_colors()[2+i], linewidth=1, label=algname)
    lines!(ax_2, xs, ys_2[i], color=Makie.wong_colors()[2+i], linewidth=1)
  end
  f[2,1:2] = Legend(f[2,1:2], ax_1, title="Algorithm", orientation=:horizontal, merge=true)
  f
end

# record the optimization progress
final_step = 200
record(optim_progress_fig, "2_parameter_estimation/figures/Edes_optimization.gif", 1:1:final_step, framerate=10) do t
  step[] = t
end