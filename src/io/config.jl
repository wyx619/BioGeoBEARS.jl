# Configuration readers.

"""
    read_times(path) -> Vector{Float64}

Time-period boundaries for a time-stratified analysis, youngest first.

R requires the file to be sorted ascending and to exclude `0`; both are checked
here too, because a file in the wrong order silently produces a wrong likelihood
rather than an error.
"""
function read_times(path::AbstractString)
    isfile(path) || throw(ArgumentError("times file not found: $(path)"))

    times = Float64[]
    for (lineno, raw) in enumerate(readlines(path))
        line = strip(raw)
        isempty(line) && continue
        v = tryparse(Float64, line)
        v === nothing &&
            throw(ArgumentError("$(path):$(lineno): expected a number, got $(repr(raw))"))
        push!(times, v)
    end

    isempty(times) && throw(ArgumentError("times file is empty: $(path)"))
    issorted(times) ||
        throw(ArgumentError("$(path): times must be sorted youngest to oldest, got $(times)"))
    0.0 in times &&
        throw(ArgumentError("$(path): time 0 must not be listed; the present is implicit"))
    return times
end

"""
    read_matrix_with_names(path) -> (rownames, colnames, Matrix{Float64})

Read a labelled numeric matrix such as the `controls` / dispersal-multiplier
files:

```
	K	O	M	H
P_mariniana_Kokee2	20	2	10	1
```

The header line supplies the column names; every following line is a row name
followed by one number per column.  Fields may be separated by tabs or spaces.
"""
function read_matrix_with_names(path::AbstractString)
    isfile(path) || throw(ArgumentError("matrix file not found: $(path)"))

    lines = readlines(path)
    idx = findfirst(l -> !isempty(strip(l)), lines)
    idx === nothing && throw(ArgumentError("matrix file is empty: $(path)"))

    colnames = String.(split(strip(lines[idx])))
    isempty(colnames) &&
        throw(ArgumentError("$(path):$(idx): missing column names"))

    rownames = String[]
    rows = Vector{Float64}[]
    for lineno in (idx + 1):length(lines)
        line = strip(lines[lineno])
        isempty(line) && continue
        fields = split(line)
        length(fields) == length(colnames) + 1 ||
            throw(ArgumentError("$(path):$(lineno): expected a row name and $(length(colnames)) numbers, got $(length(fields)) fields"))
        push!(rownames, String(fields[1]))
        push!(rows, _parse_row(fields[2:end], path, lineno))
    end

    isempty(rows) && throw(ArgumentError("$(path): no data rows"))
    return rownames, colnames, permutedims(reduce(hcat, rows))
end

function _parse_row(fields, path, lineno)
    vals = Vector{Float64}(undef, length(fields))
    for (j, f) in enumerate(fields)
        v = tryparse(Float64, f)
        v === nothing &&
            throw(ArgumentError("$(path):$(lineno): expected a number, got $(repr(f))"))
        vals[j] = v
    end
    return vals
end

"""
    read_area_matrix(path) -> (colnames, Matrix{Float64})

Read a labelled square matrix with a column header and one numeric row per
area, optionally terminated by a line containing `END`.  Rows may or may not
carry a leading row name: R's `dispersal_multipliers.txt` and
`Hawaii_KOMH_distances.txt` are bare numbers, while `controls` files name each
row, so the reader decides by counting the fields against the header.

This is what backs `dispersal_multipliers_fn` and `distsfn`.
"""
function read_area_matrix(path::AbstractString)
    isfile(path) || throw(ArgumentError("matrix file not found: $(path)"))
    lines = [l for l in readlines(path) if !isempty(strip(l)) && strip(l) != "END"]
    isempty(lines) && throw(ArgumentError("matrix file is empty: $(path)"))

    colnames = String.(split(strip(lines[1])))
    rows = Vector{Float64}[]
    for (lineno, line) in enumerate(lines[2:end])
        lineno += 1
        fields = split(line)
        if length(fields) == length(colnames)
            push!(rows, parse.(Float64, fields))
        elseif length(fields) == length(colnames) + 1
            push!(rows, parse.(Float64, fields[2:end]))
        else
            throw(ArgumentError("$(path):$(lineno): expected $(length(colnames)) " *
                                "columns, got $(length(fields))"))
        end
    end
    mat = permutedims(reduce(hcat, rows))
    n = length(colnames)
    size(mat) == (n, n) ||
        throw(ArgumentError("$(path): expected a square $(n) x $(n) matrix, got " *
                            "$(size(mat))"))
    return colnames, mat
end

"""
    read_areas_allowed(path) -> Vector{Matrix{Float64}}

Read a list of 1/0 matrices that restrict which area combinations may form a
range, youngest period first.  Blocks are separated by blank lines and the file
may end with a line containing `END`.

The same parser serves `areas_adjacency` files — R has two functions because the
two options mean different things to the model, not because the format differs.
"""
read_areas_allowed(path::AbstractString) = _read_matrix_blocks(path, "areas_allowed")

read_areas_adjacency(path::AbstractString) = _read_matrix_blocks(path, "areas_adjacency")

function _read_matrix_blocks(path::AbstractString, what::AbstractString)
    isfile(path) || throw(ArgumentError("$(what) file not found: $(path)"))

    blocks = Matrix{Float64}[]
    colnames = String[]
    rows = Vector{Float64}[]
    in_block = false

    flush_block!() = begin
        if in_block
            isempty(rows) &&
                throw(ArgumentError("$(path): a block has names but no rows"))
            mat = permutedims(reduce(hcat, rows))
            size(mat, 1) == size(mat, 2) ||
                throw(ArgumentError("$(path): matrix is $(size(mat, 1)) x $(size(mat, 2)), but must be square"))
            size(mat, 1) == length(colnames) ||
                throw(ArgumentError("$(path): $(length(colnames)) area names but $(size(mat, 1)) rows"))
            push!(blocks, mat)
        end
        empty!(rows)
        in_block = false
    end

    for (lineno, raw) in enumerate(readlines(path))
        line = strip(raw)
        if line == "END"
            flush_block!()
            return blocks
        elseif isempty(line)
            flush_block!()
        elseif !in_block
            colnames = String.(split(line))
            in_block = true
        else
            fields = split(line)
            length(fields) == length(colnames) ||
                throw(ArgumentError("$(path):$(lineno): $(length(colnames)) area names but $(length(fields)) numbers"))
            push!(rows, _parse_row(fields, path, lineno))
        end
    end

    flush_block!()
    isempty(blocks) && throw(ArgumentError("$(path): no matrices found"))
    return blocks
end

"""
    prune_states(ss::StateSpace, allowed) -> StateSpace

Drop every state whose areas are not all mutually allowed by `allowed`, a
square 0/1 matrix in the same area order as `ss.areanames`.  The null range is
always kept, matching `prune_states_list()`.

Note this can leave `max_range_size` larger than any surviving state; that field
describes how the space was generated, not what survived.
"""
function prune_states(ss::StateSpace, allowed::AbstractMatrix)
    n = nareas(ss)
    size(allowed) == (n, n) ||
        throw(ArgumentError("areas_allowed matrix is $(size(allowed)), expected ($n, $n)"))

    keep = Vector{Vector{Int}}()
    for state in ss.states
        isempty(state) && (push!(keep, state); continue)
        all(a -> all(b -> !iszero(allowed[a, b]), state), state) && push!(keep, state)
    end

    return StateSpace(ss.areanames, keep, ss.include_null_range, ss.max_range_size)
end
