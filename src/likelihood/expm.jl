# Matrix exponentiation and transition probabilities.

"""
    r_sum(v, exact = true) -> Float64

R's `sum()` for a double vector.

R accumulates the terms sequentially in a `long double` (a 64-bit mantissa on
x86-64) and rounds once at the end, whereas Julia uses pairwise summation in
`Float64`.  The two can differ by an ulp, which is enough to flip a BSM sample,
so the R-faithful paths use this instead of `sum`.

`exact = false` falls back to `sum(v)`.
"""
function r_sum(v, exact::Bool = true)
    exact || return Float64(sum(v))
    acc = BigFloat(0.0; precision = 64)
    for x in v
        acc = acc + BigFloat(x; precision = 64)
    end
    return Float64(acc)
end

"""
    transition_matrix(Q, t; exact = true) -> Matrix{Float64}

`exp(t·Q)`, the transition matrix over time `t`, with
`P[i, j] = Pr(j at time t | i at time 0)`.

`Q` is a rate matrix (rows sum to zero), so every row of `P` sums to one.  A
zero-length branch returns the identity, which also keeps the null-range row
(a row of zeros) from being exponentiated.

The dense path is [`expm_expokit`](@ref), a port of EXPOKIT's `DGPADM`.  With
`exact = true` it is not merely numerically equivalent to R's
`expokit_dgpadm_Qmat()`: it performs the same floating-point operations in the
same order. BSM sampling can depend on those details.
`exact = false` swaps the reference BLAS/LAPACK kernels for the
optimised ones, which is faster but differs in the last bits.
"""
function transition_matrix(Q::AbstractMatrix, t::Real; exact::Bool = true)
    n = size(Q, 1)
    size(Q, 2) == n ||
        throw(ArgumentError("transition_matrix needs a square matrix, got $(size(Q))"))
    t == 0 && return Matrix{Float64}(I, n, n)
    # `exact` reproduces R's EXPOKIT + netlib order; the fast path is Padé with
    # scaling and squaring from LinearAlgebra, which is several times quicker on
    # the 16-256 state spaces a typical analysis has.
    exact && return expm_expokit(Q, t)
    return exp(Matrix{Float64}(Q) .* float(t))
end

"""
    expm_expokit(Q, t; ideg = 6, exact = true) -> Matrix{Float64}

`exp(t·Q)` by EXPOKIT's `DGPADM`: an irreducible rational Padé approximation of
degree `ideg` combined with scaling and squaring.

This is a line-by-line port of `rexpokit/src/lapack/my_expokit.f`, the routine
BioGeoBEARS calls through `expokit_dgpadm_Qmat()`.  R passes `t(Q)` as the
Fortran array and reads the result back row-major, so the net effect is
`exp(t·Q)`; the two transposes are reproduced here because they change which
operands the matrix products see.
"""
function expm_expokit(Q::AbstractMatrix, t::Real; ideg::Integer = 6)
    n = size(Q, 1)
    size(Q, 2) == n ||
        throw(ArgumentError("expm_expokit needs a square matrix, got $(size(Q))"))
    ideg >= 2 || throw(ArgumentError("ideg must be at least 2, got $ideg"))
    n == 0 && return Matrix{Float64}(undef, 0, 0)
    H = permutedims(Matrix{Float64}(Q))          # R hands Fortran t(Q)
    return permutedims(_dgpadm(H, float(t), Int(ideg)))
end

# `exp(t·H)` for a dense `H`, following DGPADM statement by statement.
function _dgpadm(H::Matrix{Float64}, t::Float64, ideg::Int)
    m = size(H, 1)

    # scaling: hnorm = ||t*H||_inf, with the row sums accumulated in the
    # Fortran's own order (one pass over columns per row).
    hnorm = 0.0
    for i in 1:m
        s = 0.0
        for j in 1:m
            s += abs(H[i, j])
        end
        hnorm = max(hnorm, s)
    end
    hnorm = abs(t * hnorm)
    hnorm == 0.0 && return Matrix{Float64}(I, m, m)

    ns = max(0, trunc(Int, log(hnorm) / log(2.0)) + 2)
    scale = t / ldexp(1.0, ns)
    scale2 = scale * scale

    # Padé coefficients, in the Fortran's recurrence order.
    c = Vector{Float64}(undef, ideg + 1)
    c[1] = 1.0
    ii = ideg + 1
    jj = 2 * ideg + 1
    for k in 1:ideg
        c[k + 1] = (c[k] * Float64(ii - k)) / Float64(k * (jj - k))
    end
    cp = c[ideg]
    cq = c[ideg + 1]

    # H2 = scale2 * H * H
    H2 = zeros(Float64, m, m)
    _ref_gemm!(H2, H, H, scale2, 0.0)

    # p = cp*I, q = cq*I, plus the free workspace block.
    p = zeros(Float64, m, m)
    q = zeros(Float64, m, m)
    for j in 1:m
        p[j, j] = cp
        q[j, j] = cq
    end
    buf = zeros(Float64, m, m)

    # Horner: the Fortran rotates three blocks; `buf` is always the free one.
    iodd = 1
    k = ideg - 1
    while true
        src = iodd == 1 ? q : p
        _ref_gemm!(buf, src, H2, 1.0, 0.0)
        for j in 1:m
            buf[j, j] += c[k]
        end
        if iodd == 1
            q = buf
        else
            p = buf
        end
        buf = src
        iodd = 1 - iodd
        k -= 1
        k > 0 || break
    end

    # (+/-)(I + 2*(p\q))
    if iodd == 1
        old = q
        _ref_gemm!(buf, old, H, scale, 0.0)
        q = buf
        buf = old
    else
        old = p
        _ref_gemm!(buf, old, H, scale, 0.0)
        p = buf
        buf = old
    end

    # iq := iq - ip  (DAXPX with da = -1)
    @inbounds for idx in eachindex(q, p)
        q[idx] = q[idx] + (-1.0) * p[idx]
    end

    # DGESV: solve iq * X = ip, overwriting ip with X.
    _ref_gesv!(q, p)

    @inbounds for idx in eachindex(p)     # DSCALX with da = 2
        p[idx] = 2.0 * p[idx]
    end
    for j in 1:m
        p[j, j] += 1.0
    end

    if ns == 0 && iodd == 1
        @inbounds for idx in eachindex(p)
            p[idx] = -1.0 * p[idx]
        end
        return p
    end

    # squaring: exp(tH) = (exp(tH/2^ns))^(2^ns); the Fortran resets iodd here
    iput = p
    iodd = 1
    for _ in 1:ns
        if iodd == 1
            _ref_gemm!(q, p, p, 1.0, 0.0)
            iput = q
        else
            _ref_gemm!(p, q, q, 1.0, 0.0)
            iput = p
        end
        iodd = 1 - iodd
    end
    return iput
end

# ---------------------------------------------------------------------------
# netlib BLAS / LAPACK kernels
# ---------------------------------------------------------------------------
#
# R's rexpokit ships the netlib reference implementations, whose summation
# order differs from MKL's.  Reproducing the loops below is what makes the
# exponential bit-identical to R's.

# `C = alpha*A*B + beta*C`, square, no transposes.  This is the exact loop nest
# of netlib `dgemm`'s `NOTA && NOTB` branch: the inner loop is over rows of a
# column, so `C[:, j] += (alpha*B[l, j]) * A[:, l]`.
function _ref_gemm!(C::Matrix{Float64}, A::Matrix{Float64}, B::Matrix{Float64},
                    alpha::Float64, beta::Float64)
    m = size(C, 1)
    @inbounds for j in 1:m
        if beta == 0.0
            for i in 1:m
                C[i, j] = 0.0
            end
        elseif beta != 1.0
            for i in 1:m
                C[i, j] = beta * C[i, j]
            end
        end
        for l in 1:m
            temp = alpha * B[l, j]
            @simd for i in 1:m
                C[i, j] = C[i, j] + temp * A[i, l]
            end
        end
    end
    return C
end



# netlib `dgemv`'s `'N'` branch: `y = A*x`, accumulated one column at a time.
# R computes `P %*% v` the same way, so this ordering matters for bit-exactness.
function _ref_gemv!(y::AbstractVector{Float64}, A::AbstractMatrix{Float64},
                    x::AbstractVector{Float64})
    m = size(A, 1)
    fill!(y, 0.0)
    @inbounds for j in 1:size(A, 2)
        xj = x[j]
        xj == 0.0 && continue
        @simd for i in 1:m
            y[i] = y[i] + xj * A[i, j]
        end
    end
    return y
end

# netlib `dgemv`'s `'T'` branch: `y = A'*x`, a plain dot product per column.
function _ref_gemvT!(y::AbstractVector{Float64}, A::AbstractMatrix{Float64},
                     x::AbstractVector{Float64})
    @inbounds for j in 1:size(A, 2)
        s = 0.0
        for i in 1:size(A, 1)
            s = s + A[i, j] * x[i]
        end
        y[j] = s
    end
    return y
end

# netlib `dgesv` for a small system: unblocked `dgetf2` factorisation followed
# by `dgetrs`.  `A` is overwritten by its LU factors, `B` by the solution.
function _ref_gesv!(A::Matrix{Float64}, B::Matrix{Float64})
    n = size(A, 1)
    nrhs = size(B, 2)
    ipiv = Vector{Int}(undef, n)

    @inbounds for j in 1:n
        # pivot: first index of the largest magnitude in the column
        p = j
        amax = abs(A[j, j])
        for i in j+1:n
            if abs(A[i, j]) > amax
                amax = abs(A[i, j])
                p = i
            end
        end
        ipiv[j] = p
        if A[p, j] != 0.0
            if p != j
                for col in 1:n
                    A[j, col], A[p, col] = A[p, col], A[j, col]
                end
            end
            for i in j+1:n
                A[i, j] = A[i, j] * (1.0 / A[j, j])
            end
        end
        if j < n
            for col in j+1:n
                if A[j, col] != 0.0
                    for i in j+1:n
                        A[i, col] = A[i, col] - A[i, j] * A[j, col]
                    end
                end
            end
        end
    end

    # dgetrs: apply the row interchanges, then solve L*Y = PB and U*X = Y.
    @inbounds for i in 1:n-1
        if ipiv[i] != i
            for col in 1:nrhs
                B[i, col], B[ipiv[i], col] = B[ipiv[i], col], B[i, col]
            end
        end
    end
    @inbounds for col in 1:nrhs
        for k in 1:n
            if B[k, col] != 0.0
                for i in k+1:n
                    B[i, col] = B[i, col] - B[k, col] * A[i, k]
                end
            end
        end
        for k in n:-1:1
            if B[k, col] != 0.0
                B[k, col] = B[k, col] / A[k, k]
                for i in 1:k-1
                    B[i, col] = B[i, col] - B[k, col] * A[i, k]
                end
            end
        end
    end
    return B
end

"""
    apply_transition(Q, t, v; exact = true) -> Vector{Float64}

`exp(t·Q) · v`.

`Q` is normally sparse (the rate matrix produced by `qmatrix`), but R's
likelihood path densifies it and uses `expokit_dgpadm_Qmat()`, so `exact = true`
goes through the same [`expm_expokit`](@ref) port. `exact = false` densifies
`Q` and calls `LinearAlgebra.exp`.

Both paths satisfy `exp(tA)·0 = 0`, so an unreachable segment stays
at zero instead of producing NaNs.
"""
apply_transition(Q::AbstractMatrix, t::Real, v::AbstractVector; exact::Bool = true) =
    exact ? _ref_gemv!(zeros(Float64, size(Q, 1)), _dense_transition(Q, t), v) :
            transition_matrix(Q, t; exact = false) * v

# `exp(t·Q)` as a dense matrix, shared by the exact matvec path.
_dense_transition(Q, t) = transition_matrix(Q, t; exact = true)

function apply_transition(Q::SparseMatrixCSC, t::Real, v::AbstractVector;
                          exact::Bool = true)
    t == 0 && return Vector{Float64}(v)
    exact && return _ref_gemv!(zeros(Float64, size(Q, 1)), _dense_transition(Q, t), v)
    return transition_matrix(Q, t; exact = false) * v
end

"""
    apply_transition_transpose(Q, t, v; exact = true) -> Vector{Float64}

`exp(t·Q)' · v`, the direction R's uppass uses (`v %*% expokit_dgpadm_Qmat2(...)`).

With `exact = true` the product accumulates in the same order as netlib
`dgemv`'s `'T'` branch, so it is bit-identical to R.
"""
function apply_transition_transpose(Q::AbstractMatrix, t::Real, v::AbstractVector;
                                    exact::Bool = true)
    exact || return transition_matrix(Q, t; exact = false)' * v
    return _ref_gemvT!(zeros(Float64, size(Q, 2)), _dense_transition(Q, t), v)
end
