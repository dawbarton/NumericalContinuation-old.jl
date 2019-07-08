"""
    specialize(prob)

Return a specialized problem structure, typically a parameterized structure
for speed. It is assumed that once `specialize` is called, no further changes
to the problem structure are made. `specialize` should not change the problem
in any material way (e.g., the number of equations or variables used).

Note that the resulting specialized problem might share data with the original
problem structure.
"""
function specialize end

specialize(prob) = prob  # default fall-back

"""
    setuseroptions(tbx, values::Dict{Symbol, Any})

Set the internal options of a toolbox structure.
"""
function setuseroptions! end  
