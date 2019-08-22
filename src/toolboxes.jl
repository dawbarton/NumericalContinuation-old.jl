getsubproblems(tbx::AbstractToolbox) = ()
Base.nameof(tbx::AbstractToolbox) = tbx.name

function Base.show(io::IO, tbx::AbstractToolbox{T}) where T
    print(io, string(nameof(typeof(tbx))))
    (T !== Float64) && print(io, "{$T}")
    print(io, "(:$(nameof(tbx)))")
end