"""
  module VisualizationTools

This module provides a set of functions for visualizing data, including Fourier spectrum analysis, time series plots, 
parameter estimation plots, and statistical behavior of simulation results.

Include the following lines in the main script:

```Julia
push!(LOAD_PATH, "./src")
include("../src/VisualizationTools.jl")
using .VisualizationTools
```

"""
module VisualizationTools

# EXPORTED FUNCTIONS
export plot_fourier_spectrum, 
       u_eps_plot, rho_plot, 
       rho_history, 
       post_plotting_real, 
       post_plotting_sim, 
       plot_data


# REQUIRED PACKAGES
using Plots
using CairoMakie
using PairPlots
using FFTW
using DocStringExtensions

"""
$(TYPEDSIGNATURES)
Plots the Fourier spectrum of a given signal.

# PARAMETERS
- `t`: Time values.
- `u`: Signal values.
- `indices`: Indices where vertical lines should be drawn.
- `save::Bool=false`: If `true`, saves the plot as `"fourier_spectrum.png"`.

# RETURNS
- The function does not return a value but displays the plot.
"""
function plot_fourier_spectrum(t::Union{Vector{Float64}, Vector{Int64}}, u::Vector{Float64}, indices::Union{Vector{Int64}, StepRange{Int64, Int64}}; save::Bool=false)
  Fs = 1 / (t[2] - t[1])
  N = length(u)
  freqs = fftfreq(N, Fs)

  fourier_transform_sim = abs.(fft(u))

  frequencies = freqs[(freqs .> 0) .& (freqs .<= 0.5)]
  amplitudes = abs.(fourier_transform_sim[(freqs .> 0) .& (freqs .<= 0.5)])

  p = Plots.plot(frequencies, amplitudes, xlabel="Frequency (Hz)", ylabel="Magnitude", title="Magnitude Spectrum", legend=false)

  seq = 0:1:length(frequencies)
  seq = seq / length(frequencies)
  for pos in indices
    vline!(p, [seq[pos]], line=:dash, color=:red, label=false)
  end

  display(p)
  if save
    savefig(p, "fourier_spectrum.png")
  end
end


"""
$(TYPEDSIGNATURES)
Plots the time series of magnetic field strength.

# PARAMETERS
- `t::Union{Vector{Float64}, Vector{Int64}}`: Time values.
- `u::Vector{Float64}`: Magnetic field strength values.

# RETURNS
- The function does not return a value but displays the plot.
"""
function plot_data(t::Union{Vector{Float64}, Vector{Int64}}, u::Vector{Float64})
  p = Plots.plot(t, u, label = "B(t)", xlabel = "Time", ylabel = "Magnetic Field Strength", 
  title = "Simulated data", linewidth = 1)
  display(p)
  savefig(p, "simulated_data_plot.png")
end
  
# function for plotting u and epsilon behaviour in the sABC simulation
"""
$(TYPEDSIGNATURES)
Plots the history of epsilon and u in the sABC simulation.

# PARAMETERS
- `eps_hist`: History of epsilon values.
- `u_hist`: History of u values.

# RETURNS
- The function does not return a value but displays the plot.
"""
function u_eps_plot(eps_hist, u_hist)
  p1 = Plots.plot(vec(Matrix(eps_hist)), title="Epsilon History", xlabel="Iteration", legend=false, yscale=:log10)
  p2 = Plots.plot(vec(Matrix(u_hist)), title="U History", xlabel="Iteration", legend=false)

  combined_plot = Plots.plot(p1, p2, layout=(1, 2), size=(1000, 400))

  display(combined_plot)
  savefig(combined_plot, "u_eps_plot.png")
end

# function for plotting of rhos behaviour in the sABC simulation
"""
$(TYPEDSIGNATURES)
Plots the history of rhos in the sABC simulation.

# PARAMETERS
- `rho_hist`: History of rho values.
- `style::String="together"`: Style of plotting. Accepted values are `"together"` or `"divided"`.

# RETURNS
- The function does not return a value but displays the plot.
"""
function rho_plot(rho_hist; style::String = "together")
  if style == "divided"
    rho_plots = []

    for i in 1:20
      push!(rho_plots, Plots.plot(Matrix(rho_hist)[i, :], title="Rho History Stat $i", xlabel="Iteration", legend=false, yscale=:log10))
    end

    combined_plot = Plots.plot(rho_plots..., layout=(5, 4), size=(1200, 1800))

    display(combined_plot)
    savefig(combined_plot, "rho_divided.png")

  elseif style == "together"
    labels = ["rho$i" for i in 1:size(Matrix(rho_hist), 1)]

    all_rho_plot = Plots.plot(title="All Rho History", xlabel="Iteration", ylabel="Rho", legend=true, yscale=:log10)
    for i in 1:size(Matrix(rho_hist), 1)
      Plots.plot!(1:size(Matrix(rho_hist), 2), Matrix(rho_hist)[i, :], label=labels[i])
    end

    display(all_rho_plot)
    savefig(all_rho_plot, "rho_together.png")
  else
    throw(ErrorException("Invalid style: $style. The accepted styles are \"divided\" or \"together\""))
  end
end 

# Function for plotting the posterior as a corner plot
"""
$(TYPEDSIGNATURES)
Plots the posterior distribution as a corner plot.

# PARAMETERS
- `post_par`: Posterior parameters to plot.

# RETURNS
- The function does not return a value but displays the plot.
"""
function post_plotting_real(post_par)
  p = pairplot(post_par)

  display(p)
  CairoMakie.save("posteriors.png", p)
end

"""
$(TYPEDSIGNATURES)
Plots the posterior distribution with true parameter values as a corner plot.

# PARAMETERS
- `posterior_params`: Posterior parameters to plot.
- `true_vals`: True values of parameters for comparison.

# RETURNS
- The function does not return a value but displays the plot.
"""
function post_plotting_sim(posterior_params, true_vals)
  p = pairplot(
    posterior_params,
    PairPlots.Truth(
        (;
            N_value = true_vals[1],
            T_value = true_vals[2],
            tau_value = true_vals[3],
            sigma_value = true_vals[4],
            Bmax_value = true_vals[5]
        ),
        
        label="True Values"
    )
  )

  display(p)
  CairoMakie.save("posteriors.png", p)
end

# Function for plotting the rhos in a loglog scale
"""
$(TYPEDSIGNATURES)
Plots the history of rhos in a loglog scale.

# PARAMETERS
- `rho_hist`: History of rho values.

# RETURNS
- The function does not return a value but displays the plot.
"""
function rho_history(rho_hist)
  labels = ["rho$i" for i in 1:size(Matrix(rho_hist), 1)]
  
  all_rho_plot = Plots.plot(title = "All Rho History", xlabel = "Iteration", ylabel = "Rho", legend = true, yscale = :log10, xscale = :log10)
  
  for i in 1:size(Matrix(rho_hist), 1)
      Plots.plot!(1:size(Matrix(rho_hist), 2), Matrix(rho_hist)[i, :], label = labels[i])
  end
  
  display(all_rho_plot)
end


end