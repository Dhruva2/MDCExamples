### A Pluto.jl notebook ###
# v0.18.1

using Markdown
using InteractiveUtils

# ╔═╡ 0ef42fc1-0428-45f0-a64a-d3dc45ae1794
Pkg.activate(".")

# ╔═╡ 5dc73524-3488-4f6f-b393-11e3333f3235
Pkg.add("https://github.com/Dhruva2/NeuronBuilder.git")

# ╔═╡ af643208-9f97-11ec-223b-2737428114c7
using OrdinaryDiffEq, ModelingToolkit, MinimallyDisruptiveCurves, Statistics

# ╔═╡ 6ee8bb56-1422-4528-9a84-4e5336c331cd
using NeuronBuilder

# ╔═╡ Cell order:
# ╟─af643208-9f97-11ec-223b-2737428114c7
# ╠═0ef42fc1-0428-45f0-a64a-d3dc45ae1794
# ╠═5dc73524-3488-4f6f-b393-11e3333f3235
# ╠═6ee8bb56-1422-4528-9a84-4e5336c331cd
