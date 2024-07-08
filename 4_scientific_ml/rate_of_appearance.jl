# Optimizing EDES model parameters on the predict data
using StaticArrays
using OrdinaryDiffEq
using CairoMakie
using Statistics
using QuasiMonteCarlo
using Optimization, OptimizationOptimJL, OptimizationOptimisers
using Random
using DelimitedFiles
using SimpleChains
using ComponentArrays
using Trapz
using DataInterpolations

lambda = 0.0
n_initializations = 50
Random.seed!(524)

rng = Random.GLOBAL_RNG

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

# define the activation function
softplus(x) = log(1 + exp(x))
rbf(x) = exp(-x^2)

# define the neural network
chain = SimpleChain(
    static(1),
    TurboDense{true}(rbf, 6),
    TurboDense{true}(rbf, 6),
    TurboDense{true}(softplus, 1)
)

function insulin_interpolator(mins, insulin_timepoints)

  steady_state_timepoints_start = [-60, -50, -40, -30]
  insulin_start = repeat([mins[1]], length(steady_state_timepoints_start))

  steady_state_timepoints_end = insulin_timepoints[end] .+ [60, 120, 240, 360, 480]
  insulin_end = repeat([mins[1]], length(steady_state_timepoints_end))

  CubicSpline([insulin_start; mins...; insulin_end], [steady_state_timepoints_start; insulin_timepoints; steady_state_timepoints_end]);

end

function minimal_model_ude(chain, I)

  function mmode(u, p, t)
    G, X = u
    p1, p2, p3, f, Vg, Dmeal, Gb = p.ode

    dG = -G*X - p3*(G - Gb) + (f/Vg)*Dmeal*chain([t], p.neural)[1]
    dX = -p1*X + p2*(I(t) - I(0))
    return SA[dG, dX]
  end
end

# define the loss function for the population data
function loss_population_regularized(θ, p)
  problem = p[1]
  glucose_data = p[2]
  glucose_std = p[3]
  glucose_timepoints = p[4]
  constants = p[5]
  lambda = p[6]
  chain = p[7]

  # update the parameters
  p_ode = ComponentArray(ode = constants, neural = θ)

  # solve the problem
  pred = solve(problem, Tsit5(), p=p_ode, saveat=glucose_timepoints)
  sol = Array(pred)
  data_error = sum(abs2, (sol[1,glucose_timepoints .∈ Ref(glucose_timepoints)] .- glucose_data) ./ glucose_std)
  ra = [chain([t], θ)[1] for t in glucose_timepoints]
  auc_regularizer = abs(trapz(glucose_timepoints, ra)-1.) * lambda
  return data_error + auc_regularizer
end

constants = SA[4.91e-2, 2.75e-5, 4.61e-2, 5.551e-3, 18.57, 85.5e3, glucose_mean[1]]

prob = ODEProblem(minimal_model_ude(chain, insulin_interpolator(insulin_mean, t_insulin)), SA[glucose_mean[1], 0.0], (0.0, 240.0))

# compute initial guess
initial_guess = [SimpleChains.init_params(chain, rng=rng) for _ in 1:10_000]

# compute the loss for the initial guess
initial_loss_values = [loss_population_regularized(guess, (prob, glucose_mean, glucose_std, t_glucose, constants, lambda, chain)) for guess in initial_guess]

# sort the initial guess by loss
sorted_guesses = initial_guess[partialsortperm(initial_loss_values,1:n_initializations)]
optimized_parameters = []
losses = []
for i in 1:n_initializations

  println("Optimization $i")
  try 
    optfunc = OptimizationFunction(loss_population_regularized, Optimization.AutoForwardDiff())
    optprob = OptimizationProblem(optfunc, sorted_guesses[i], (prob, glucose_mean, glucose_std, t_glucose, constants, lambda, chain))
    optsol_1 = solve(optprob, ADAM(0.01), maxiters=500)

    optprob_2 = OptimizationProblem(optfunc, Vector{Float64}(optsol_1.u), (prob, glucose_mean, glucose_std, t_glucose, constants, lambda, chain))
    optsol_2 = solve(optprob_2, BFGS(initial_stepnorm=0.01), reltol=1e-6)

    p_opt_2 = ComponentArray(ode = constants, neural = optsol_2.u)
    push!(losses, optsol_2.objective)
    push!(optimized_parameters, p_opt_2)
  catch
    println("Optimization $i failed")
  end
end

# sort the optimized parameters by loss
loss_top10 = partialsortperm(losses, 1:25)

fig_opt = let f = Figure(size=(600,300))

  ax_glucose = CairoMakie.Axis(
    f[1, 1],
      xlabel = "Time [min]",
      ylabel = "Glucose [mg/dL]",
      title = "Optimized Minimal Model (Glucose)"
  )
  ax_insulin = CairoMakie.Axis(
    f[1, 2],
      xlabel = "Time [min]",
      ylabel = "Insulin [mU/L]",
      title = "Optimized Minimal Model (Ra)"
  )

  for opt in optimized_parameters[loss_top10]
    simulation_opt_2 = solve(prob, Tsit5(), p=opt, saveat=0.1)
    ra_times = 0:0.1:360
    lines!(ax_glucose, simulation_opt_2.t, simulation_opt_2[1,:], color=(Makie.wong_colors()[1], 0.2))
    lines!(ax_insulin, ra_times, [chain([t], opt.neural)[1] for t in ra_times], color=(Makie.wong_colors()[1], 0.2), linewidth=2)
  end
  scatter!(ax_glucose, t_glucose, glucose_mean, color=:black, marker='∘', markersize=25)
  errorbars!(ax_glucose, t_glucose, glucose_mean, glucose_std, color=:black, whiskerwidth=8)

  

  f
end

save("4_scientific_ml/lambda_$(round(lambda; digits=2)).png", fig_opt)