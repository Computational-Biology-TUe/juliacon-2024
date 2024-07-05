# Implementation of the EDeS model in Julia

using StaticArrays
using OrdinaryDiffEq
using CairoMakie
using Statistics
using QuasiMonteCarlo
using Optimization, OptimizationOptimJL
using Random
using ProgressMeter
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

# We first start with parameter estimation

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
  sum(abs2, (sol .- [glucose_data'; insulin_data']) ./ [glucose_std'; insulin_std'])
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

optsol = solve(optprob, LBFGS())

# Profile Likelihood Analysis

function loss_pla(loss, pla_param, fixed_parameter_value, npar)

  # compute the order of the parameters
  parameter_order = zeros(Int64,npar)
  parameter_order[[1:pla_param-1; (pla_param+1):npar]] .= 1:npar-1
  parameter_order[pla_param] = npar

  function _loss_pla(θ, p)
    # construct the full parameter vector
    θ_full = [θ; fixed_parameter_value][parameter_order]
    loss(θ_full, p)
  end
end

function likelihood(pla_param_index, loss, θ_init, lb, ub, p)
  function _likelihood(pla_param)
    loss_function = loss_pla(loss, pla_param_index, pla_param, length(θ_init)+1)
    optfunc = OptimizationFunction(loss_function, Optimization.AutoForwardDiff())
    optprob = OptimizationProblem(optfunc, θ_init, p, lb = lb, ub = ub)

    optsol = solve(optprob, LBFGS())
    optsol.objective
  end
end

function run_pla(pla_param, param_range, param_optim, loss, initial_guess, lb_pla, ub_pla, p)

  # setup PLA likelihood
  pla_likelihood = likelihood(
    pla_param, 
    loss, 
    initial_guess, 
    lb_pla, ub_pla, p
  )

  param_range = sort(unique([param_range; param_optim]))
  optim_index = findfirst(x -> x == param_optim, param_range)
  pla_likelihood_values = Float64[]
  prog = Progress(length(param_range), dt=0.1, desc="Computing Likelihood Profile for $pla_param")
  for px in param_range
    l = try
      pla_likelihood(px)
    catch
      println("Error at $px")
      NaN
    end
    next!(prog)
    push!(pla_likelihood_values, l)
  end

  return pla_likelihood_values, optim_index
end


likelihood_profiles = Vector{Float64}[]
optim_indices = Int64[]
for pla_param in 1:4

  npar = length(optsol.u)
  initial_guess = optsol.u[[1:pla_param-1; (pla_param+1):npar]]
  lb = [0.008, 0.037, 1., 0.]
  ub = [0.017, 0.06, 3., 20.]
  lb_pla = lb[[1:pla_param-1; (pla_param+1):npar]] .* 0
  ub_pla = ub[[1:pla_param-1; (pla_param+1):npar]] .* 1e3
  pla_range = LinRange(lb[pla_param], ub[pla_param], 50)
  pla_likelihood_values, optim_index = run_pla(
    pla_param, pla_range, optsol.u[pla_param], loss, initial_guess, lb_pla, ub_pla, (
      prob, mean_data[1,:], std_data[1,:], mean_data[2,:], std_data[2,:], data_timepoints, constants
    )
  )
  push!(likelihood_profiles, pla_likelihood_values)
  push!(optim_indices, optim_index)
end


# create a PLA plot
pla_plot = let f = Figure(size=(600,600))
  lb = [0.008, 0.037, 1., 0.]
  ub = [0.017, 0.06, 3., 20.]
  for i in 1:4
    row = div(i-1,2) + 1
    col = i % 2 == 0 ? 2 : 1
    ax = Axis(f[row,col], xlabel="Parameter Value", ylabel="Objective", title="Parameter $i", limits=(nothing, (0, optsol.objective * 5)))
    pla_range = LinRange(lb[i], ub[i], 50)
    param_range = sort(unique([pla_range; optsol.u[i]]))
    lines!(ax, param_range, likelihood_profiles[i], label="Likelihood Profile", color=Makie.wong_colors()[1])
    scatter!(ax, [param_range[optim_indices[i]]], [likelihood_profiles[i][optim_indices[i]]], label="Optimal Value", color=Makie.wong_colors()[2], markersize=10)
    hlines!(ax, [optsol.objective*2], color=Makie.wong_colors()[3], linestyle=:dash, linewidth=3, label="2x Objective")
    hlines!(ax, [optsol.objective*3], color=Makie.wong_colors()[4], linestyle=:dash, linewidth=3, label="3x Objective")

    if i == 1
      f[3,1:2] = Legend(f[3,1:2], ax, orientation=:horizontal)
    end

  end

  f

end

save("3_identifiability/edes_profile_likelihood.png", pla_plot)