# Add the src directory to the LOAD_PATH
push!(LOAD_PATH, "./src")

# Include the module files from the src directory
include("../src/SDDESolarDynamo.jl")
include("../src/DirUtils.jl")
include("../src/VisualizationTools.jl")

# Use the modules
using .SDDESolarDynamo
using .DirUtils
using .VisualizationTools

using Documenter

# Generate documentation
makedocs(;
    modules=[SDDESolarDynamo, DirUtils, VisualizationTools],  # Include all modules
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
        "Getting started" => "usage.md",
        "Modules" => [
            "SDDESolarDynamo" => "modules/sddesolardynamo.md",
            "DirUtils" => "modules/dirutils.md",
            "VisualizationTools" => "modules/visualizationtools.md"
        ],
        "API" => "api.md",
        "FAQ" => "faq.md"
    ],
)
