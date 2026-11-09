# Lagrange-PHYLIP input and output.

"""
    read_lagrange_phylip(path; block_all_question_marks=true) -> TipRanges

Read a LAGRANGE geography file.

The format is a header line followed by one line per tip:

```
19	4 (K O M H)
P_mariniana_Kokee2	1000
P_mariniana_Oahu	0100
```

The header gives the number of tips, the number of areas, and optionally the
area names in parentheses.  Without names the areas are called `A`, `B`, `C`, …
as in R.  Each data line is a tip name followed by one character per area:
`'1'` present, `'0'` absent, `'?'` unknown.

Two of R's checks are reproduced because they catch real data-entry mistakes: a
row whose length differs from the header's area count, and a file with duplicate
area names.  `block_all_question_marks` additionally rejects a tip coded
entirely `'?'`, which no BioGeoBEARS analysis can use.

Unlike R — which flattens the whole file with `scan()` and reshapes the token
stream — lines are parsed one at a time, so a stray blank line or a trailing
newline cannot shift every subsequent tip by one column.
"""
function read_lagrange_phylip(path::AbstractString;
                              block_all_question_marks::Bool = true)
    isfile(path) ||
        throw(ArgumentError("geography file not found: $(path)"))

    lines = readlines(path)
    isempty(lines) &&
        throw(ArgumentError("geography file is empty: $(path)"))

    header = split(strip(lines[1]))
    length(header) >= 2 ||
        throw(ArgumentError("$(path):1: expected '<ntips> <nareas> (areas...)', got $(repr(lines[1]))"))

    ntips_declared = _parse_int(header[1], path, 1)
    nareas = _parse_int(header[2], path, 1)
    nareas >= 1 ||
        throw(ArgumentError("$(path):1: nareas must be positive, got $nareas"))

    if length(header) > 2
        areanames = [_strip_parens(tok) for tok in header[3:end]]
        length(areanames) == nareas ||
            throw(ArgumentError("$(path):1: header declares $nareas areas but names $(length(areanames))"))
    else
        areanames = _default_area_names(nareas)
    end

    tipnames = String[]
    rows = Vector{Char}[]

    for (lineno, raw) in enumerate(lines[2:end])
        line = strip(raw)
        isempty(line) && continue
        fields = split(line)
        length(fields) == 2 ||
            throw(ArgumentError("$(path):$(lineno + 1): expected '<tip> <0/1/? string>', got $(repr(raw))"))

        name = String(fields[1])
        codes = collect(String(fields[2]))
        length(codes) == nareas ||
            throw(ArgumentError("$(path):$(lineno + 1): tip $(repr(name)) has $(length(codes)) codes but the header declares $nareas areas"))
        bad = setdiff(unique(codes), ['0', '1', '?'])
        isempty(bad) ||
            throw(ArgumentError("$(path):$(lineno + 1): tip $(repr(name)) contains $(bad); only 0, 1 and ? are allowed"))

        push!(tipnames, name)
        push!(rows, codes)
    end

    length(tipnames) == ntips_declared ||
        throw(ArgumentError("$(path): header declares $ntips_declared tips but the file has $(length(tipnames))"))
    length(unique(tipnames)) == length(tipnames) ||
        throw(ArgumentError("$(path): duplicate tip names"))

    data = permutedims(reduce(hcat, rows))     # tips x areas
    tips = TipRanges(tipnames, areanames, data)

    if block_all_question_marks
        for i in 1:ntips(tips)
            if all(==('?'), view(tips.data, i, :))
                throw(ArgumentError(
                    "tip $(repr(tips.tipnames[i])) is coded entirely '?'; " *
                    "a tip must be observed in at least one area"))
            end
        end
    end

    return tips
end

_strip_parens(tok::AbstractString) = replace(tok, "(" => "", ")" => "")

function _parse_int(tok::AbstractString, path, lineno)
    v = tryparse(Int, tok)
    v === nothing &&
        throw(ArgumentError("$(path):$(lineno): expected an integer, got $(repr(tok))"))
    return v
end

# R falls back to the first `nareas` letters of the alphabet.
function _default_area_names(nareas::Integer)
    nareas <= 26 ||
        throw(ArgumentError("at most 26 unnamed areas can be letter-coded, got $nareas"))
    return [string(c) for c in 'A':'Z'][1:nareas]
end

"""
    write_lagrange_phylip(path, tips::TipRanges)

Write `tips` back in LAGRANGE format, so Julia-side fixtures stay in the same
dialect as the R side.
"""
function write_lagrange_phylip(path::AbstractString, tips::TipRanges)
    open(path, "w") do io
        println(io, ntips(tips), '\t', nareas(tips), " (", join(tips.areanames, " "), ")")
        for i in 1:ntips(tips)
            println(io, tips.tipnames[i], '\t', String(view(tips.data, i, :)))
        end
    end
    return path
end
