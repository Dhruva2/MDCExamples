"""
Only observe the product

What do we want? The briggs-haldane approximation: ddt [ES] = 0. This means that kon/(koff + kcat) is practically unidentifiable.

https://chem.libretexts.org/Bookshelves/Biological_Chemistry/Supplemental_Modules_(Biological_Chemistry)/Enzymes/Enzymatic_Kinetics/Michaelis-Menten_Kinetics_1
"""



function MichaelisMenten(input)
    
    @variables t
    @parameters kon koff kcat
    D = Differential(t)
    @variables pos(t) vel(t) inp(t)

    eqs = [D(pos) ~ vel,
        D(vel) ~ input(t) - (1 / m) * (c * vel + k * pos),
  ]

    ps = [k,c,m] .=>  [1.,4.,1.] 
    ics = [pos, vel] .=> [1.,0.]
    od = ODESystem(eqs, t, first.(ics), first.(ps); name=:MichaelisMenten)
  # tspan = (0.,100.)
  # prob = ODEProblem(od, ics, tspan, ps)
    return od, ics, tspan, ps
end