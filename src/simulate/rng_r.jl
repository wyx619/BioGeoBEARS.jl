# R-compatible random number generation.

const _MT_N = 624
const _MT_M = 397
const _MT_MATRIX_A = 0x9908b0df
const _MT_UPPER_MASK = 0x80000000
const _MT_LOWER_MASK = 0x7fffffff

"""
    RRng

R's Mersenne-Twister generator, seeded exactly as `set.seed()` does.
"""
mutable struct RRng
    mt::Vector{UInt32}
    mti::Int
end

"""
    RRng(seed) -> RRng

Equivalent to R's `set.seed(seed)` followed by using the global stream.

R does **not** seed Mersenne-Twister with the standard `init_genrand`
recurrence, nor with the historical `MT_sgenrand()` (which builds each word
from the high 16 bits of two LCG values).  Modern R's `RNG_Init()` first
scrambles the seed with 50 LCG steps and then stores one **full 32-bit** LCG
value per state word:

```
for (j = 0; j < 50; j++) seed = 69069 * seed + 1;
for (j = 0; j < 625; j++) { seed = 69069 * seed + 1; i_seed[j] = seed; }
```

`i_seed[0]` is the `mti` slot and is overwritten by `FixupSeeds()` with 624;
`i_seed[1..624]` become `mt[1..624]`.  Hence one LCG step is consumed for the
`mti` slot before the first state word.

All arithmetic is 32-bit unsigned (`Int32` in R's headers is `unsigned int`).
"""
function RRng(seed::Integer)
    s = UInt32(seed & 0xffffffff)
    for _ in 1:50
        s = UInt32(69069) * s + UInt32(1)
    end

    # i_seed[0] (the mti slot) consumes one LCG step as well.
    s = UInt32(69069) * s + UInt32(1)

    mt = Vector{UInt32}(undef, _MT_N)
    for i in 1:_MT_N
        s = UInt32(69069) * s + UInt32(1)
        mt[i] = s
    end
    return RRng(mt, _MT_N + 1)
end

function _mt_twist!(rng::RRng)
    mt = rng.mt
    @inbounds for kk in 1:(_MT_N - _MT_M)
        y = (mt[kk] & _MT_UPPER_MASK) | (mt[kk + 1] & _MT_LOWER_MASK)
        mt[kk] = mt[kk + _MT_M] ⊻ (y >> 1) ⊻ (y & 0x1 == 0x1 ? _MT_MATRIX_A : UInt32(0))
    end
    @inbounds for kk in (_MT_N - _MT_M + 1):(_MT_N - 1)
        y = (mt[kk] & _MT_UPPER_MASK) | (mt[kk + 1] & _MT_LOWER_MASK)
        mt[kk] = mt[kk + (_MT_M - _MT_N)] ⊻ (y >> 1) ⊻ (y & 0x1 == 0x1 ? _MT_MATRIX_A : UInt32(0))
    end
    y = (mt[_MT_N] & _MT_UPPER_MASK) | (mt[1] & _MT_LOWER_MASK)
    mt[_MT_N] = mt[_MT_M] ⊻ (y >> 1) ⊻ (y & 0x1 == 0x1 ? _MT_MATRIX_A : UInt32(0))
    rng.mti = 1
    return rng
end

"""
    r_unif_rand(rng) -> Float64

One draw from R's `unif_rand()`, in `[0, 1)`.
"""
function r_unif_rand(rng::RRng)
    rng.mti > _MT_N && _mt_twist!(rng)
    y = rng.mt[rng.mti]
    rng.mti += 1
    y ⊻= (y >> 11)
    y ⊻= (y << 7) & 0x9d2c5680
    y ⊻= (y << 15) & 0xefc60000
    y ⊻= (y >> 18)
    x = Float64(y) * 2.3283064365386963e-10
    # R's fixup(): keep the draw strictly inside (0, 1).
    i2_32m1 = 2.328306437080797e-10
    x <= 0.0 && return 0.5 * i2_32m1
    (1.0 - x) <= 0.0 && return 1.0 - 0.5 * i2_32m1
    return x
end

r_runif(rng::RRng) = r_unif_rand(rng)

"""
    r_runif(rng, n) -> Vector{Float64}

`n` draws from R's `runif(n)`.
"""
r_runif(rng::RRng, n::Integer) = [r_unif_rand(rng) for _ in 1:n]

# Ahrens–Dieter q table, copied verbatim from R's src/nmath/sexp.c.
const _EXP_Q = (
    0.6931471805599453, 0.9333736875190459, 0.9888777961838675,
    0.9984959252914960, 0.9998292811061389, 0.9999833164100727,
    0.9999985691438767, 0.9999998906925558, 0.9999999924734159,
    0.9999999995283275, 0.9999999999728814, 0.9999999999985598,
    0.9999999999999289, 0.9999999999999968, 0.9999999999999999,
    1.0000000000000000,
)

"""
    r_exp_rand(rng) -> Float64

R's `exp_rand()`: a standard exponential deviate.  Consumes one uniform almost
always, and more only when the first lands in the tail.
"""
function r_exp_rand(rng::RRng)
    a = 0.0
    u = r_unif_rand(rng)
    while u <= 0.0 || u >= 1.0
        u = r_unif_rand(rng)
    end
    while true
        u += u
        u > 1.0 && break
        a += _EXP_Q[1]
    end
    u -= 1.0
    u <= _EXP_Q[1] && return a + u

    i = 0
    ustar = r_unif_rand(rng)
    umin = ustar
    while true
        ustar = r_unif_rand(rng)
        umin > ustar && (umin = ustar)
        i += 1
        u > _EXP_Q[i + 1] || break
    end
    return a + umin * _EXP_Q[1]
end

"""
    r_rexp(rng, rate) -> Float64

R's `rexp(1, rate)`.
"""
r_rexp(rng::RRng, rate::Real) = (1.0 / rate) * r_exp_rand(rng)

"""
    _revsort!(a, ib)

R's `revsort()`: heapsort `a` into **descending** order, permuting `ib` with it.
The C code indexes 1..n, so this is a direct transcription.
"""
function _revsort!(a::Vector{Float64}, ib::Vector{Int})
    n = length(a)
    n <= 1 && return nothing

    l = (n >> 1) + 1
    ir = n
    local ra::Float64, ii::Int
    while true
        if l > 1
            l -= 1
            ra = a[l]
            ii = ib[l]
        else
            ra = a[ir]
            ii = ib[ir]
            a[ir] = a[1]
            ib[ir] = ib[1]
            ir -= 1
            if ir == 1
                a[1] = ra
                ib[1] = ii
                return nothing
            end
        end
        i = l
        j = l << 1
        while j <= ir
            if j < ir && a[j] > a[j + 1]
                j += 1
            end
            if ra > a[j]
                a[i] = a[j]
                ib[i] = ib[j]
                i = j
                j += i
            else
                j = ir + 1
            end
        end
        a[i] = ra
        ib[i] = ii
    end
end

"""
    r_sample_prob(rng, prob; replace=true) -> Int
    r_sample_prob(rng, prob, k; replace=true) -> Vector{Int}

R's `sample(1:length(prob), k, replace=replace, prob=prob)`, returning 1-based
indices.  Probabilities are normalised by their positive sum as R's
`FixupProb()` does, then sorted descending with `revsort()`; the draw itself
follows `ProbSampleReplace` / `ProbSampleNoReplace` line by line.

For `replace=true`, R switches to Walker's alias method when more than 200
entries have `n * p[i] > 0.1`; that branch is reproduced as well, since it
consumes the stream differently (one `R_unif_index` plus one `unif_rand` per
draw instead of a single `unif_rand`).
"""
function r_sample_prob(rng::RRng, prob::AbstractVector{<:Real}; replace::Bool = true)
    return r_sample_prob(rng, prob, 1; replace = replace)[1]
end

function r_sample_prob(rng::RRng, prob::AbstractVector{<:Real}, k::Integer;
                       replace::Bool = true)
    n = length(prob)
    n >= 1 || throw(ArgumentError("prob must have at least one entry"))
    k >= 0 || throw(ArgumentError("k must be non-negative"))

    p = Float64.(prob)
    s = 0.0
    npos = 0
    for x in p
        isfinite(x) || throw(ArgumentError("NA/Inf in probability vector"))
        x < 0 && throw(ArgumentError("negative probability"))
        if x > 0
            npos += 1
            s += x
        end
    end
    npos > 0 || throw(ArgumentError("no positive probabilities"))
    (!replace && k > npos) && throw(ArgumentError("too few positive probabilities"))
    p ./= s

    if replace
        nc = count(x -> n * x > 0.1, p)
        if nc > 200
            return _walker_prob_sample_replace(rng, p, k)
        end
    end

    perm = collect(1:n)
    _revsort!(p, perm)

    out = Vector{Int}(undef, k)
    if replace
        for i in 2:n
            p[i] += p[i - 1]
        end
        for i in 1:k
            rU = r_unif_rand(rng)
            j = 1
            while j < n
                rU <= p[j] && break
                j += 1
            end
            out[i] = perm[j]
        end
        return out
    end

    totalmass = 1.0
    n1 = n - 1
    for i in 1:k
        rT = totalmass * r_unif_rand(rng)
        mass = 0.0
        j = 1
        while j <= n1
            mass += p[j]
            rT <= mass && break
            j += 1
        end
        out[i] = perm[j]
        totalmass -= p[j]
        for kk in j:n1
            p[kk] = p[kk + 1]
            perm[kk] = perm[kk + 1]
        end
        n1 -= 1
    end
    return out
end

# Walker's alias method, transcribed from R's `walker_ProbSampleReplace()`.
# `p` is already normalised (R's `FixupProb()` has run); the array is modified
# in place exactly as R does.
function _walker_prob_sample_replace(rng::RRng, p::Vector{Float64}, nans::Integer)
    n = length(p)
    HL = Vector{Int}(undef, n)
    q = Vector{Float64}(undef, n)
    a = Vector{Int}(undef, n)

    # Direct transcription of R's pointers: `H` fills HL from the front,
    # `L` fills it from the back.  Kept in 0-based terms so the pointer
    # arithmetic matches the C line for line.
    h = 0                      # H - HL
    l = n                      # L - HL
    for i0 in 0:(n - 1)
        q[i0 + 1] = p[i0 + 1] * n
        if q[i0 + 1] < 1.0
            HL[h + 1] = i0 + 1
            h += 1
        else
            l -= 1
            HL[l + 1] = i0 + 1
        end
    end

    if h > 0 && l < n
        for k0 in 0:(n - 2)
            i = HL[k0 + 1]
            j = HL[l + 1]
            a[i] = j
            q[j] += q[i] - 1.0
            if q[j] < 1.0
                l += 1
            end
            l >= n && break
        end
    end
    for i in 1:n
        q[i] += i - 1              # R adds the 0-based index
    end

    out = Vector{Int}(undef, nans)
    for i in 1:nans
        k0 = r_unif_index(rng, n)          # 0-based
        rU = k0 + r_unif_rand(rng)
        # `a` already holds 1-based indices, so no +1 here (R's C stores
        # 0-based ones and adds 1 at the end).
        out[i] = rU < q[k0 + 1] ? k0 + 1 : a[k0 + 1]
    end
    return out
end

"""
    r_rbits(rng, bits) -> Float64

R's `rbits()`: `bits+1` uniforms scaled to 16-bit chunks, combined and masked
to `bits` bits.
"""
function r_rbits(rng::RRng, bits::Integer)
    v = Int64(0)
    n = 0
    while n <= bits
        v1 = Int64(floor(r_unif_rand(rng) * 65536))
        v = 65536 * v + v1
        n += 16
    end
    return Float64(v & ((Int64(1) << bits) - 1))
end

"""
    r_unif_index(rng, dn) -> Int

R's `R_unif_index()` with the default `Sample_kind = REJECTION`: a uniform
integer in `0:dn-1`.  Used by the unweighted `sample.int` path and by Walker's
alias method.
"""
function r_unif_index(rng::RRng, dn::Integer)
    dn <= 0 && return 0
    bits = ceil(Int, log2(dn))
    dv = r_rbits(rng, bits)
    while dn <= dv
        dv = r_rbits(rng, bits)
    end
    return Int(dv)
end

"""
    r_sample_int(rng, n, k; replace=true) -> Vector{Int}

R's `sample.int(n, k, replace=replace)` **without** a `prob` argument, i.e. the
uniform path.  Returns 1-based indices.
"""
function r_sample_int(rng::RRng, n::Integer, k::Integer; replace::Bool = true)
    n > 0 || throw(ArgumentError("n must be positive"))
    (!replace && k > n) && throw(ArgumentError("cannot sample more than n without replacement"))
    out = Vector{Int}(undef, k)
    if replace || k < 2
        for i in 1:k
            out[i] = r_unif_index(rng, n) + 1
        end
        return out
    end
    x = collect(1:n)
    m = n
    for i in 1:k
        j = r_unif_index(rng, m) + 1
        out[i] = x[j]
        x[j] = x[m]
        m -= 1
    end
    return out
end
