# Optimizing EDES model parameters on the predict data
using StaticArrays
using OrdinaryDiffEq
using CairoMakie
using Statistics
using QuasiMonteCarlo
using Optimization, OptimizationOptimJL
using Random
using DelimitedFiles
using SciMLBase: OptimizationSolution
Random.seed!(5234)

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

# load the data
glucose = readdlm("2_parameter_estimation/predict/glucose.csv", ';')
insulin = readdlm("2_parameter_estimation/predict/insulin.csv", ';')

timepoints = readdlm("2_parameter_estimation/predict/timepoints.csv", ';')
t_glucose = Float64.(timepoints[1, 2:end])
t_insulin = Float64.(timepoints[2, [2:6; 8]])

# average the data
glucose_mean = mean(glucose, dims=1)[:]
glucose_std = std(glucose, dims=1)[:]
insulin_mean = mean(insulin, dims=1)[:]
insulin_std = std(insulin, dims=1)[:]

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

function default_parameters(Dmeal, bw, Gb, Ib)
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

  # collection of parameters
  SA[k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, tau_i, tau_d, beta, Gren, EGPb, Km, f, Vg, c1, sigma, Dmeal, bw, Gb, Ib]
end

function construct_parameters(θ, c)

  # Estimated parameters
  k1 = θ[1]
  k5 = θ[2]
  k6 = θ[3]
  
  # Fixed parameters
  k2, k3, k4, k7, k8, k9, k10, tau_i, tau_d, beta, Gren, EGPb, Km, f, Vg, c1, sigma, Dmeal, bw, Gb, Ib = c

  return SA[k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, tau_i, tau_d, beta, Gren, EGPb, Km, f, Vg, c1, sigma, Dmeal, bw, Gb, Ib]
end

# define the loss function for the population data
function loss_individual(θ, p)
  problem = p[1]
  glucose_data = p[2]
  insulin_data = p[3]
  glucose_timepoints = p[4]
  insulin_timepoints = p[5]
  constants = p[6]

  # update the parameters
  p = construct_parameters(θ, constants)
  data_timepoints = sort(unique([glucose_timepoints; insulin_timepoints]))

  # solve the problem
  pred = solve(problem, Tsit5(), p=p, saveat=data_timepoints, save_idxs=[2, 3])
  sol = Array(pred)
  sum(abs2, (sol[1,data_timepoints .∈ Ref(glucose_timepoints)] .- glucose_data) ./ maximum(glucose_data)) + sum(abs2, (sol[2,data_timepoints .∈ Ref(insulin_timepoints)] .- insulin_data) ./ maximum(insulin_data))
end

function optimize(glucose, insulin, t_glucose, t_insulin)

  constants = default_parameters(86e3, 72.88, glucose[1], insulin[1])[[2:4; 7:end]]
  constants = SA[constants...]

  prob = ODEProblem(edesode, SA[0.0, glucose[1], insulin[1], 0.0], (0.0, 240.0))

  # compute initial guess
  initial_guess = QuasiMonteCarlo.sample(
    10_000, [0., 0., 0.], [1., 0.5, 20.], LatinHypercubeSample()
  )

  # compute the loss for the initial guess
  initial_loss_values = [loss_individual(guess, (prob, glucose, insulin, t_glucose, t_insulin, constants)) for guess in eachcol(initial_guess)]

  # sort the initial guess by loss
  sorted_guesses = initial_guess[:,partialsortperm(initial_loss_values,1)]

  optfunc = OptimizationFunction(loss_individual, Optimization.AutoForwardDiff())
  optprob = OptimizationProblem(optfunc, sorted_guesses,(prob, glucose, insulin, t_glucose, t_insulin, constants),lb = [0., 0., 0.], ub = [1., 0.5, 20.])

  solve(optprob, LBFGS())
end


individual_optsols = OptimizationSolution[]

for i in axes(glucose, 1)
  optsol = optimize(glucose[i,:], insulin[i,:], t_glucose, t_insulin)
  push!(individual_optsols, optsol)
end

# errors
k1 = [sol.u[1] for sol in individual_optsols]
k5 = [sol.u[2] for sol in individual_optsols]
k6 = [sol.u[3] for sol in individual_optsols]

# visualize the parameters
fig_params = let f = Figure(size=(400,400))
  ax = Axis(f[1,1], xlabel="k5 (Insulin Sensitivity)", ylabel="k6 (Insulin Capacity)")
  scatter!(ax, k5, k6, color=:black, markersize=9)
  f
end

function plot_individual(optsols, glucose, insulin, t_glucose, t_insulin, i)

  glucose = glucose[i,:]
  insulin = insulin[i,:]
  optsol = optsols[i]

  prob = ODEProblem(edesode, SA[0.0, glucose[1], insulin[1], 0.0], (0.0, 240.0))

  constants = default_parameters(86e3, 72.88, glucose[1], insulin[1])[[2:4; 7:end]]
  constants = SA[constants...]
  p_opt = construct_parameters(optsol.u, constants)
  simulation_opt = solve(prob, Tsit5(), p=p_opt, saveat=0.1, save_idxs=[2, 3])

  f = Figure(size=(600,300))
    ax_glucose = Axis(
    f[1, 1],
      xlabel = "Time [min]",
      ylabel = "Glucose [mg/dL]",
      title = "Optimized EDES (Glucose)"
  )
  ax_insulin = Axis(
    f[1, 2],
      xlabel = "Time [min]",
      ylabel = "Insulin [mU/L]",
      title = "Optimized EDES (Insulin)"
  )

  lines!(ax_glucose, simulation_opt.t, simulation_opt[1,:], color=Makie.wong_colors()[1])
  scatter!(ax_glucose, t_glucose, glucose, color=:black, marker='∘', markersize=25)
  lines!(ax_insulin, simulation_opt.t, simulation_opt[2,:], color=Makie.wong_colors()[1])
  scatter!(ax_insulin, t_insulin, insulin, color=:black, marker='∘', markersize=25)

  f
end

bad_fit = plot_individual(individual_optsols, glucose, insulin, t_glucose, t_insulin, 6)
save("2_parameter_estimation/figures/bad_fit.png", bad_fit, px_per_unit=2)

good_fit = plot_individual(individual_optsols, glucose, insulin, t_glucose, t_insulin, errors[1])
save("2_parameter_estimation/figures/good_fit.png", good_fit, px_per_unit=2)
