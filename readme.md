## Pluto notebooks containing tutorials on MinimallyDisruptiveCurves.jl


- Note that dependencies can take a while to load. Even more so on Julia 1.9, where more stuff is precompiled during package installation, which happens....every time you spin up the notebook.
- While you're waiting, the notebook_htmls/ folder has html printouts of the outputs .
- A weird Pluto.jl bug: evolving two-sided MDCs (which involves evolving two curves on separate threads and merging them) hangs. This doesn't happen in a normal Julia REPL. So don't evolve a curve where the span of the curve crosses through zero, i.e. of the form $(-a, b)$ where $a, b > 0$. 