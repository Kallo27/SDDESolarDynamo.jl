# FUNCTIONS NEDEED FOR THE MODEL

module SDDESolarDynamo

"""
Include the following lines in the main script:
---
push!(LOAD_PATH, "./src")
include("../src/SDDESolarDynamo.jl")
using .SDDESolarDynamo
---

"""

# EXPORTED FUNCTIONS
export f, drift, noise!, bfield, f_dist, reduced_fourier_spectrum, select_best_particles, postprocessing_params


# REQUIRED PACKAGES
using StochasticDelayDiffEq
using SpecialFunctions
using Distances
using FFTW
using DocStringExtensions
using DataFrames
using StatsBase


# Box-shaped function for the magnetic field range
"""
$(TYPEDSIGNATURES)
Box-shaped function modeling the effective strength of the magnetic field 
as a smooth window function, turning on at B_min and off at B_max.

# PARAMETERS
- B: Input magnetic field (scalar or array).
- B_max::Float64=10: Maximum magnetic field threshold.
- B_min::Float64=1: Minimum magnetic field threshold.

# RETURNS
- Array or scalar with same shape as `B`, representing the field effectiveness.
"""
function f(B, B_max = 10, B_min = 1)
  return 1 / 4 * (1 .+ erf.(B .^ 2 .- B_min ^ 2)) .* (1 .- erf.(B .^ 2 .- B_max ^ 2))
end
  
# Drift function for the SDDE
"""
$(TYPEDSIGNATURES)
Drift function for the stochastic delay differential equation (SDDE) describing 
the deterministic evolution of the solar magnetic field, including delayed feedback from past states.

# PARAMETERS
- du: Array where the drift result is stored.
- u: Current state vector [B, dB].
- h: History function.
- p: Vector of parameters [N, T, tau, sigma, Bmax].
- t: Current time.

# RETURNS
- Nothing. Modifies `du` in-place.
"""
function drift(du, u, h, p, t)
  N, T, tau, sigma, Bmax = p
  lags = (T, )
  Bhist = h(p, t - lags[1])[1]
  B, dB = u

  du[1] = dB
  du[2] = - ((2 / tau) * dB + (B / tau^2) + (N / tau^2) * Bhist * f(Bhist, Bmax))
end

# Noise function for the SDDE
"""
$(TYPEDSIGNATURES)
Noise term for the stochastic delay differential equation (SDDE), 
modeling random fluctuations in the solar dynamo with amplitude set by physical parameters.

# PARAMETERS
- du: Array where the noise contribution is stored.
- u: Current state vector [B, dB].
- h: History function.
- p: Vector of parameters [N, T, tau, sigma, Bmax].
- t: Current time.

# RETURNS
- Nothing. Modifies `du` in-place.
"""
function noise!(du, u, h, p, t)
  N, T, tau, sigma, Bmax = p
  du[1] = 0
  du[2] = (sigma * Bmax)/(tau^(3/2))
end

# SDDE problem solver
"""
$(TYPEDSIGNATURES)
Solves the stochastic delay differential equation (SDDE) for the magnetic 
field dynamics, returning the full time evolution over the specified interval.

# PARAMETERS
- θ: Vector of parameters [tau, T, N, sigma, Bmax].
- Tsim: Vector defining the simulation time interval.
- dt: Time step for the solver.

# RETURNS
- sol: Solution object containing time series of the simulated magnetic field.
"""
function bfield(θ, Tsim, dt)
    τ, T, Nd, sigma, Bmax = θ
  lags = (T, )
    h(p, t) = [Bmax, 0.]
  B0 = [Bmax, 0.]
    tspan = (Tsim[1], Tsim[2])

    prob = SDDEProblem(drift, noise!, B0, h, tspan, θ; constant_lags = lags)
    solve(prob, EM(), dt = dt, saveat = 1.0)
end

# Distance function in the sABC algorithm
"""
$(TYPEDSIGNATURES)
Computes the distance metric used in the sequential Approximate Bayesian Computation (sABC) 
algorithm, by comparing Fourier-based summary statistics of simulated and observed magnetic field data.

# PARAMETERS
- θ::Vector{Float64}: Model parameters.
- type::Int64=1: Type of distance function (currently unused).
- indices: Indices to use in the reduced Fourier spectrum comparison.
- fourier_data::Vector{Float64}: Observed Fourier summary statistics.
- Tsim: Time span for simulation.
- dt: Time step for integration.

# RETURNS
- rho: Vector of Euclidean distances between simulated and observed Fourier statistics.
"""
function f_dist(θ::Vector{Float64}; type::Int64 = 1, indices::Union{Vector{Int64}, StepRange{Int64, Int64}} = 1:6:120, fourier_data::Vector{Float64}, Tsim = Tsim, dt = dt)
  sol = bfield(θ, Tsim, dt)
  
  simulated_data = sol[1,:].^2
  fourier_stats = reduced_fourier_spectrum(simulated_data, indices)

  rho = [euclidean(fourier_stats[i], fourier_data[i]) for i in 1:length(fourier_stats)]
  return rho
end

# Function for the summary statistics
"""
$(TYPEDSIGNATURES)
Computes a reduced set of Fourier coefficients from a time series, serving as 
summary statistics for comparing simulated and observed magnetic field data.

# PARAMETERS
- u::Vector{Float64}: Time series data (typically B²).
- indices: Indices of Fourier components to retain.

# RETURNS
- Vector{Float64}: Reduced set of Fourier coefficients.
"""
function reduced_fourier_spectrum(u::Vector{Float64}, indices::Union{Vector{Int64}, StepRange{Int64, Int64}} = 1:6:120)
  fourier_transform = abs.(fft(u))
  return fourier_transform[indices]
end

# Function to select best behaving particles
"""
$(TYPEDSIGNATURES)
Computes the MSE over the distances (rhos) and selects the k best particles.

# PARAMETERS
- df_rho::DataFrame: Dataframe containing the rhos.
- df_pop::DataFrame: Dataframe containing the population (particles).
- k::Int=10: Number of best particles to extract.

# RETURNS
- best_indices::Vector{Int64}: indices of the best particles.
- best_particles::DataFrame: Dataframe containing the best particles
"""
function select_best_particles(df_rho::DataFrame, df_pop::DataFrame; k::Int=10)
    # Square all values and compute row-wise sum of squares
    df_squared = DataFrame()
    for col in names(df_rho)
      df_squared[!, col] = df_rho[!, col].^2
    end

    row_sums = [sum(row) for row in eachrow(df_squared)]

    best_indices = partialsortperm(row_sums, 1:k)
    best_particles = df_pop[best_indices, :]

    return best_indices, best_particles
end

# Function to postprocess particles results for analysis
"""
$(TYPEDSIGNATURES)
Computes the mean, mode or hybrid procedure to get the best simulation params.

# PARAMETERS
- df_pop::DataFrame: Dataframe containing the population (particles).
- strategy::Symbol: Type of postprocessing to apply.

# RETURNS
- par::Vector{Int64}: Postprocessed simulation parameters.
- title_str::String: Title for the plot based on the chosen strategy.
"""
function postprocessing_params(param_samples::DataFrame; strategy::Symbol = :mode)
  n_cols = size(param_samples, 2)
  par = Vector{Float64}(undef, n_cols)

  if strategy == :mode
    for col in 1:n_cols
      par[col] = mode(param_samples[:, col])
    end
    title_str = "Mode param data"

  elseif strategy == :mean
    for col in 1:n_cols
      par[col] = mean(param_samples[:, col])
    end
    title_str = "Mean param data"

  elseif strategy == :hybrid
    # Start with mode
    for col in 1:n_cols
      par[col] = mode(param_samples[:, col])
    end

    # Apply filters
    N_vals = param_samples[:, 1]
    N_vals = N_vals[N_vals .> 3]
    par[1] = mean(N_vals)

    T_vals = param_samples[:, 2]
    T_vals = T_vals[(T_vals .> 0.5) .& (T_vals .< 4)]
    par[2] = mean(T_vals)

    title_str = "Hybrid param data"

  else
    error("Unknown strategy: $strategy. Use :mode, :mean or :hybrid.")
  end

  println("Selected parameters ($strategy): ", par)

  return par, title_str
end

end