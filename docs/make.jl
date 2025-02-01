# Add the src directory to the LOAD_PATH
push!(LOAD_PATH, "./src")

# Include the SDDESolarDynamo.jl file from the src directory
include("../src/SDDESolarDynamo.jl")

# Use the module
using .SDDESolarDynamo
using Documenter


makedocs(;
    modules=[SDDESolarDynamo],
    authors="Lorenzo Calandra Buonaura, Lucrezia Rossi, Andrea Turci",
    repo="https://github.com/Kallo27/SDDESolarDynamo.jl",  # Keep this as a string
    sitename="SDDESolarDynamo.jl",
    checkdocs=:exports,
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://Kallo27.github.io/SDDESolarDynamo.jl",
        edit_link="main",
        assets=String[],
        inventory_version="0.1.0",
        repolink="https://github.com/Kallo27/SDDESolarDynamo.jl"  # Set repolink explicitly
    ),
    pages= [
        "Home" => "index.md",
        #"Getting started" => "usage.md",
        #"Modules" => [
        #    "Model" => "modules/model.md",
        #    "DirUtils" => "modules/dirmanaging.md",
        #    "VisualizationTools" => "modules/visualization.md"
        #],
        #"API" => "api.md",
        #"FAQ" => "faq.md"
    ],
)

#deploydocs(;
#    repo="github.com/Kallo27/SSDESolarDynamo.jl",
#    devbranch="main",
#    target="gh-pages",
#)