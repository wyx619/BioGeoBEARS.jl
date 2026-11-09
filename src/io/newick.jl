# Newick tree input.

"""
    read_tree(source; strict=true) -> BGTree

Read a rooted, binary phylogeny.

`source` may be a file path, a Newick string, or an `IO`.  A string is treated
as a path when that file exists and as Newick text otherwise, so both

    read_tree("data/Psychotria_5.2.newick")
    read_tree("((A:1,B:1):1,C:2);")

work.  Parsing is delegated to `NewickTree.readnw`; the result is converted to
a [`BGTree`](@ref) and renumbered.

Node numbering matches `ape::read.tree()`: tips in Newick order, then internal
nodes in depth-first preorder, so the root is `ntip + 1`.  With `strict = true`
(the default) a tree that is not binary, that has a node with more than two
children, or that is missing a branch length raises an `ArgumentError` instead
of being silently coerced.
"""
function read_tree(source::AbstractString; strict::Bool = true)
    if isfile(source)
        return _read_newick_text(read(source, String); strict = strict)
    end
    occursin('(', source) ||
        throw(ArgumentError(
            "read_tree: '$source' is neither an existing file nor a Newick string"))
    return _read_newick_text(source; strict = strict)
end

read_tree(io::IO; strict::Bool = true) = _read_newick_text(read(io, String); strict = strict)

function _read_newick_text(text::AbstractString; strict::Bool = true)
    root = try
        NewickTree.readnw(String(text))
    catch err
        throw(ArgumentError("read_tree: could not parse Newick input: $(sprint(showerror, err))"))
    end
    return _from_newick(root; strict = strict, root_edge = _parse_root_edge(text))
end

"""
    _parse_root_edge(text) -> Float64

Read the stem length written after the closing parenthesis, as in
`((A:1,B:1):1,C:2):0.5;`.  `NewickTree` drops it, so it is recovered from the
raw text with an anchored match.  Returns `0.0` when the tree has no stem.
"""
function _parse_root_edge(text::AbstractString)
    m = match(r"\)\s*:\s*([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\s*;\s*$", text)
    m === nothing && return 0.0
    return parse(Float64, m.captures[1])
end

"""
    _from_newick(root; strict=true) -> BGTree

Convert a `NewickTree` node tree into a [`BGTree`](@ref).

Two passes over the parsed tree: the first counts tips and internal nodes, the
second assigns node numbers in depth-first preorder while recording parent
pointers, children and branch lengths.  The pruningwise order is then derived
from the number arrays alone, so no node identity is ever needed.
"""
function _from_newick(root; strict::Bool = true, root_edge::Float64 = 0.0)
    ntip = 0
    nnode = 0
    stack = [root]
    while !isempty(stack)
        node = pop!(stack)
        if isleaf(node)
            ntip += 1
        else
            nnode += 1
            append!(stack, node.children)
        end
    end
    nnode >= 1 || throw(ArgumentError("read_tree: tree has no internal nodes"))
    ntip >= 2 || throw(ArgumentError("read_tree: tree has fewer than 2 tips"))

    total = ntip + nnode
    tip_labels = Vector{String}(undef, ntip)
    node_labels = Vector{String}(undef, nnode)
    parent = zeros(Int, total)
    left = zeros(Int, total)
    right = zeros(Int, total)
    edge_length = zeros(Float64, total)

    itip = 0
    iinternal = ntip
    stack = Tuple{typeof(root),Int}[(root, 0)]
    while !isempty(stack)
        (node, parent_number) = pop!(stack)
        if isleaf(node)
            itip += 1
            number = itip
            tip_labels[number] = node.data.name
        else
            iinternal += 1
            number = iinternal
            node_labels[number - ntip] = node.data.name
        end

        parent[number] = parent_number
        if parent_number == 0
            edge_length[number] = 0.0
        else
            len = node.data.distance
            if isnan(len)
                strict && throw(ArgumentError(
                    "read_tree: node '$number' has no branch length"))
                len = 0.0
            end
            len < 0 && throw(ArgumentError(
                "read_tree: node '$number' has a negative branch length ($len)"))
            edge_length[number] = len

            if left[parent_number] == 0
                left[parent_number] = number
            elseif right[parent_number] == 0
                right[parent_number] = number
            else
                strict && throw(ArgumentError(
                    "read_tree: node '$parent_number' has more than two children; " *
                    "BioGeoBEARS requires a binary tree"))
            end
        end

        if !isleaf(node)
            # Reverse push so children pop left-to-right and keep their Newick order.
            for child in Iterators.reverse(node.children)
                push!(stack, (child, number))
            end
        end
    end

    postorder = _postorder(left, right, root_node_number = ntip + 1)
    return check_tree(BGTree(tip_labels, node_labels, parent, left, right,
                             edge_length, postorder, ntip, nnode, root_edge))
end
