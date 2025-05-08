# FUNCTIONS NEEDED FOR SAVING THE RESULTS OF A SIMULATION

module DirUtils

"""
Include the following lines in the main script:
---
push!(LOAD_PATH, "./src")
include("../src/DirUtils.jl")
using .DirUtils
---

"""

# EXPORTED FUNCTIONS
export load_sunspots_data, create_directory, write_params_sim, save_solution, get_prior_string, save_sabc_params, save_result, switch_dir, load_param_sim, load_result, load_param_sim_data, load_solution

# REQUIRED PACKAGES
using SimulatedAnnealingABC
using StochasticDelayDiffEq
using Distributions
using DataFrames
using CSV
using XLSX
using DocStringExtensions

# function to load the dataset
"""
$(TYPEDSIGNATURES)
Loads the dataset (real data).

# PARAMETERS
- `filename::String="SN Usoskin Brehm.xlsx"`: name of the file containing the dataset.

# RETURNS
- `data`: DataFrame instance containing the extracted data.
"""
function load_sunspots_data(filename::String = "SN Usoskin Brehm.xlsx")
    filepath = "./data/" * filename
    data = DataFrame(
        year = Int[],
        open_magn_flux = Float64[],
        open_magn_flux_err = Float64[],
        ssa_open_magn_flux = Float64[],
        sunspots_num = Float64[],
        sunspots_err = Float64[],
        ssa_sunspots = Float64[]
    )

    XLSX.openxlsx(filepath) do file
        sheet = file["Data"]

    for row in XLSX.eachrow(sheet)
        if isa(row[2], Number)
            push!(data, (
                year = row[2],
                open_magn_flux = row[3],
                open_magn_flux_err = row[4],
                ssa_open_magn_flux = row[5],
                sunspots_num = row[7],
                sunspots_err = row[8],
                ssa_sunspots = row[9]
            ))
            end
        end
    end

    return data
end




# function to create a new directory for each simulations, in order to store the needed files
function create_directory(dir_type::String)
    base_path = pwd()
    i = 1
  
    if dir_type == "real"
        base_path = joinpath(base_path, "real_data_sim")
        dir_name = "real$i"
    elseif dir_type == "synthetic"
        base_path = joinpath(base_path, "synthetic_data_sim")
        dir_name = "synthetic$i"
    else
        error("Invalid directory type. Choose either 'real' or 'synthetic'.")
    end
  
    dir_path = joinpath(base_path, dir_name)
  
    while isdir(dir_path)
        i += 1
        if dir_type == "real"
            dir_name = "real$i"
        elseif dir_type == "synthetic"
            dir_name = "synthetic$i"
        end
        dir_path = joinpath(base_path, dir_name)
    end
  
    mkpath(dir_path)
    println("Directory created at: $dir_path")
    cd(dir_path)
end

# function to save the params of the simulated data
function write_params_sim(par::Vector{Float64}, Tsim::Vector{Int64}, dt::Float64)
  df = DataFrame(
    Parameter = ["N", "T", "tau", "sigma", "B_max", "tspan", "dt"],
    Value = [par[1], par[2], par[3], par[4], par[5], Tsim, dt]
  )

  CSV.write("parameters.csv", df)
end

# function to save the solution of the SDDE for the simulated data
function save_solution(sol::Union{RODESolution, Vector{RODESolution}})
    curr_path = pwd()
    filename = "synthetic_sol.csv"
    path = joinpath(curr_path, filename)

    solution_df = DataFrame(Time = sol.t, u = sol[1, :], du = sol[2, :])
    CSV.write(filename, solution_df)
    println("Solution saved to file: $path")
end

# function to save the prior as a string
function get_prior_string(prior)
  parts = []
  for d in prior.dists
    if isa(d, Uniform)
      push!(parts, "Uniform($(minimum(d)), $(maximum(d)))")
    else
      error("Unsupported distribution type: $(typeof(d))")
    end
  end
  
  return "product_distribution(" * join(parts, ", ") * ")"
end

# function to save the sabc parameters
function save_sabc_params(prior, n_particles::Int, n_simulation::Int, v::Float64, type::Int, indices::Union{Vector{Int}, StepRange{Int64, Int64}})
  curr_path = pwd()
  filename = "sabc_params.csv"
  path = joinpath(curr_path, filename)
    
  sabc_params = DataFrame(
    Parameter = ["prior", "n_particles", "n_simulation", "v", "type", "indices"],
    Value = [get_prior_string(prior), n_particles, n_simulation, v, type, string(indices)]
  )
    
 CSV.write(filename, sabc_params) 
 println("Parameters saved to: $path")
end

# Function to save the result object of a sABC algorithm
function save_result(result::SimulatedAnnealingABC.SABCresult{Vector{Float64}, Float64})
  curr_path = pwd()
  filenames = ["eps_hist.csv", "u_hist.csv", "rho_hist.csv"]
  variables = [result.state.ϵ_history, result.state.u_history, result.state.ρ_history]

  for (filename, variable) in zip(filenames, variables)
    labels = string.(1:size(variable, 1))
    path = joinpath(curr_path, filename)
    CSV.write(path, DataFrame(variable, labels))
    println("$filename data saved to: $path")
  end

  filename = "pop.csv"
  path = joinpath(curr_path, filename)

  param_samples = hcat(result.population...)

  posterior_params = DataFrame(
    N_value = param_samples[1, :],
    T_value = param_samples[2, :],
    tau_value = param_samples[3, :],
    sigma_value = param_samples[4, :],
    Bmax_value = param_samples[5, :]
  )

  CSV.write(path, posterior_params)
  println("Posterior parameters saved to: $path")

  filename = "rho.csv"
  path = joinpath(curr_path, filename)

  rho = result.ρ

  rho_values = DataFrame(rho, :auto)

  CSV.write(path, rho_values)
  println("Rho values saved to: $path")
end

function switch_dir(dir_type::String, i::Int64 = 1)
    curr_path = pwd()

    if dir_type == "real"
        dir = "real_data_sim"
        dir_name = "real$i"
    elseif dir_type == "Synthetic"
        dir = "synthetic_data_sim"
        dir_name = "synthetic$i"
    else
        error("Invalid directory type. Choose either 'real' or 'synthetic'.")
    end

    path = joinpath(curr_path, dir, dir_name)
  
    if isdir(path)
        cd(path)
        println("Moved to: $path")
    else
        throw(ErrorException("Directory does not exist: $path"))
    end
end

# function to load the parameters of the simulation
function load_param_sim(filename::String = "sabc_params.csv")
  df = CSV.read(filename, DataFrame)
  return df
end

# function to load the result object of a sABC algorithm
function load_result()
  curr_path = pwd()
  
  filenames = ["eps_hist.csv", "u_hist.csv", "rho_hist.csv"]
  variables = []
  for filename in filenames
      path = joinpath(curr_path, filename)
      push!(variables, CSV.read(path, DataFrame))
      println("Data loaded from file: $path")
  end
  eps_hist, u_hist, rho_hist = variables

  filename = "pop.csv"
  path = joinpath(curr_path, filename)
  posterior_params = CSV.read(path, DataFrame)
  println("Posterior parameters loaded from file: $path")
  
  return eps_hist, u_hist, rho_hist, posterior_params
end

function load_param_sim_data(filename::String = "parameters.csv")
  df = CSV.read(filename, DataFrame)
  return df
end

"""load solution"""
function load_solution(filename::String = "simulated_sol.csv")
  df = CSV.read(filename, DataFrame)

  t = Vector(df.Time)
  u = Vector(df.u)
  du = Vector(df.du)

  return t, u, du
end



end