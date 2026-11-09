# Tree structure and traversal.

"""
    BGTree

Rooted, binary phylogeny in the BioGeoBEARS layout.

Node numbering follows `ape`: tips are `1:ntip` in Newick (left-to-right) order,
and internal nodes are `ntip+1 : ntip+nnode` in depth-first preorder, so the
root is always node `ntip + 1`.  This is exactly the numbering R's `prt()`
reports, which keeps the two implementations directly comparable.

`postorder` holds every non-root node with both children before their parent
and the left child before the right one: the pruningwise order that the
likelihood downpass walks.

`edge_length[i]` is the length of the branch from node `i` to its parent; the
root's entry is `0.0` (it has no parent branch).  `root_edge` is the optional
stem below the root, written after the closing parenthesis in Newick
(`((A,B),C):0.5;`); it is `0.0` when absent, which matches ape's
`phy.root.edge`.
"""
struct BGTree
    tip_labels::Vector{String}
    node_labels::Vector{String}
    parent::Vector{Int}
    left::Vector{Int}
    right::Vector{Int}
    edge_length::Vector{Float64}
    postorder::Vector{Int}
    ntip::Int
    nnode::Int
    root_edge::Float64
end

ntips(tr::BGTree) = tr.ntip
nnodes(tr::BGTree) = tr.ntip + tr.nnode
root_node(tr::BGTree) = tr.ntip + 1

is_tip(tr::BGTree, i::Integer) = 1 <= i <= tr.ntip
is_internal(tr::BGTree, i::Integer) = tr.ntip < i <= nnodes(tr)
is_root(tr::BGTree, i::Integer) = i == root_node(tr)

parent_node(tr::BGTree, i::Integer) = tr.parent[i]
left_child(tr::BGTree, i::Integer) = tr.left[i]
right_child(tr::BGTree, i::Integer) = tr.right[i]
branch_length(tr::BGTree, i::Integer) = tr.edge_length[i]

"""
    root_edge(tr) -> Float64

Length of the stem below the root, or `0.0` when the tree has none.  Only used
when the likelihood is asked to integrate below the lowest fork.
"""
root_edge(tr::BGTree) = tr.root_edge

"""
    pruning_order(tr::BGTree) -> Vector{Int}

Non-root nodes in pruningwise (postorder) order: children before parents, left
before right.
"""
pruning_order(tr::BGTree) = tr.postorder

"""
    node_depths(tr::BGTree) -> Vector{Float64}

Distance from the root to every node.  The root is `0.0`; tips carry the tree
height, possibly differing in the last few digits for a nearly-ultrametric tree.
"""
function node_depths(tr::BGTree)
    depths = zeros(Float64, nnodes(tr))
    # Node numbers are not monotone along a path (the root is ntip+1 and tips are
    # 1:ntip), so walk the reverse of the pruningwise order, which visits every
    # parent before its children.
    for n in Iterators.reverse(tr.postorder)
        depths[n] = depths[tr.parent[n]] + tr.edge_length[n]
    end
    return depths
end

"""
    tree_height(tr::BGTree) -> Float64

Root-to-tip distance of the deepest node.
"""
tree_height(tr::BGTree) = maximum(node_depths(tr))

"""
    node_times_bp(tr::BGTree) -> Vector{Float64}

Time before present for every node; the root gets the tree height and the tips
get approximately zero.
"""
node_times_bp(tr::BGTree) = tree_height(tr) .- node_depths(tr)

"""
    prt_table(tr::BGTree)

The columns of R's `prt()` that describe tree structure, as a named tuple of
vectors indexed by node number.  `node_type` is `"tip"`, `"internal"` or
`"root"`.
"""
function prt_table(tr::BGTree)
    depths = node_depths(tr)
    height = maximum(depths)
    n = nnodes(tr)
    types = Vector{String}(undef, n)
    for i in 1:n
        types[i] = is_tip(tr, i) ? "tip" : (is_root(tr, i) ? "root" : "internal")
    end
    return (
        node = collect(1:n),
        ancestor = copy(tr.parent),
        edge_length = copy(tr.edge_length),
        node_ht = depths,
        time_bp = height .- depths,
        node_type = types,
    )
end

"""
    check_tree(tr::BGTree)

Verify the structural invariants of a `BGTree`: consistent vector lengths, a
binary topology, a single root, and children that agree with their parent
pointers.  Throws `ArgumentError` on the first violation and returns `tr`
otherwise.
"""
function check_tree(tr::BGTree)
    n = nnodes(tr)
    length(tr.parent) == n == length(tr.left) == length(tr.right) ==
        length(tr.edge_length) ||
        throw(ArgumentError("BGTree vectors have inconsistent lengths"))
    length(tr.tip_labels) == tr.ntip ||
        throw(ArgumentError("tip_labels has $(length(tr.tip_labels)) entries, ntip is $(tr.ntip)"))
    length(tr.node_labels) == tr.nnode ||
        throw(ArgumentError("node_labels has $(length(tr.node_labels)) entries, nnode is $(tr.nnode)"))

    roots = findall(==(0), tr.parent)
    length(roots) == 1 ||
        throw(ArgumentError("expected exactly one root, found $(length(roots))"))
    roots[1] == root_node(tr) ||
        throw(ArgumentError("root is node $(roots[1]), expected $(root_node(tr))"))

    for i in 1:n
        if is_tip(tr, i)
            (tr.left[i] == 0 && tr.right[i] == 0) ||
                throw(ArgumentError("tip $i has children"))
        else
            (tr.left[i] != 0 && tr.right[i] != 0) ||
                throw(ArgumentError("internal node $i is not binary"))
            for c in (tr.left[i], tr.right[i])
                tr.parent[c] == i ||
                    throw(ArgumentError("node $c lists parent $(tr.parent[c]), but is a child of $i"))
            end
        end
    end

    length(tr.postorder) == n - 1 ||
        throw(ArgumentError("postorder has $(length(tr.postorder)) entries, expected $(n - 1)"))
    return tr
end

"""
    _postorder(left, right; root_node_number) -> Vector{Int}

Every node except the root, in postorder: both children before their parent,
left child before right.  Iterative, so deep trees cannot overflow the stack.
"""
function _postorder(left::Vector{Int}, right::Vector{Int}; root_node_number::Int)
    out = Int[]
    sizehint!(out, length(left) - 1)
    stack = Tuple{Int,Int}[(root_node_number, 0)]
    while !isempty(stack)
        (node, state) = stack[end]
        if left[node] == 0                 # tip
            pop!(stack)
            push!(out, node)
        elseif state == 0
            stack[end] = (node, 1)
            push!(stack, (left[node], 0))
        elseif state == 1
            stack[end] = (node, 2)
            push!(stack, (right[node], 0))
        else
            pop!(stack)
            push!(out, node)
        end
    end
    pop!(out)                              # the root has no parent branch
    return out
end

function Base.show(io::IO, tr::BGTree)
    print(io, "BGTree: ", tr.ntip, " tips, ", tr.nnode, " internal nodes")
end
