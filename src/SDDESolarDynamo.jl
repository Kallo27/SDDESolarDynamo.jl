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
export f, drift, noise!, bfield, f_dist, reduced_fourier_spectrum


# REQUIRED PACKAGES
using StochasticDelayDiffEq
using SpecialFunctions
using Distances
using FFTW


# Box-shaped function for the magnetic field range 
function f(B, B_max = 10, B_min = 1)
  return 1 / 4 * (1 .+ erf.(B .^ 2 .- B_min ^ 2)) .* (1 .- erf.(B .^ 2 .- B_max ^ 2))
end
  
# Drift function for the SDDE
function drift(du, u, h, p, t)
  N, T, tau, sigma, Bmax = p
  lags = (T, )
  Bhist = h(p, t - lags[1])[1]
  B, dB = u

  du[1] = dB
  du[2] = - ((2 / tau) * dB + (B / tau^2) + (N / tau^2) * Bhist * f(Bhist, Bmax))
end

# Noise function for the SDDE
function noise!(du, u, h, p, t)
  N, T, tau, sigma, Bmax = p
  du[1] = 0
  du[2] = (sigma * Bmax)/(tau^(3/2))
end

# SDDE problem solver
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
function f_dist(θ::Vector{Float64}; type::Int64 = 1, indices::Union{Vector{Int64}, StepRange{Int64, Int64}} = 1:6:120, fourier_data::Vector{Float64}, Tsim = Tsim, dt = dt)
  sol = bfield(θ, Tsim, dt)
  
  simulated_data = sol[1,:].^2
  fourier_stats = reduced_fourier_spectrum(simulated_data, indices)

  rho = [euclidean(fourier_stats[i], fourier_data[i]) for i in 1:length(fourier_stats)]
  return rho
end

# Function for the summary statistics
function reduced_fourier_spectrum(u::Vector{Float64}, indices::Union{Vector{Int64}, StepRange{Int64, Int64}} = 1:6:120)
  fourier_transform = abs.(fft(u))
  return fourier_transform[indices]
end



end