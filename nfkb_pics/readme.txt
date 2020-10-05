## the nfkb mdc curves

curve 2 is k2 against c1, in case the colours are not clear

curve 1: 98 seconds
curve 2: 122 seconds
curve 3: 252 seconds
Curve 4: 196`z seconds
curve 5: 410 seconds

curve 4:  as t1 increased, although the cost stayed low, the integrator suddenly became unstable. So left it

Curve 4: as t1 decreased, a compensatory change kept the cost low. Without this compensation, the cost explodes much faster (though only to about 1.2, even if t1 goes to minus infinity in log space). So eventually, compensation helps to balance out decrease in t1.



