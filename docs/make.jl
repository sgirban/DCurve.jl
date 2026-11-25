using DCurve
using Documenter

DocMeta.setdocmeta!(DCurve, :DocTestSetup, :(using DCurve); recursive=true)

makedocs(;
    modules=[DCurve],
    authors="Girban Sava",
    sitename="DCurve.jl",
    format=Documenter.HTML(;
        canonical="https://sgirban.github.io/DCurve.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/sgirban/DCurve.jl.git",
    devbranch="main",
    push_preview=true,
)
