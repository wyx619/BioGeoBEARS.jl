# Run from the repository root with BioGeoBEARS 1.1.4, ape and cladoRcpp.
suppressMessages(library(BioGeoBEARS))
suppressMessages(library(ape))
suppressMessages(library(cladoRcpp))
q <- matrix(c(0, 0, 0, 0, -0.4, 0.4, 0, 0.2, -0.2), 3, byrow = TRUE)
coo <- list(as.integer(c(0, 0, 1, 1)), as.integer(c(0, 0, 1, 1)),
            as.integer(c(0, 1, 0, 1)), c(1, 1, 1, 1))
cases <- list(balanced = "test/data/bsm_balanced.newick", unbalanced = "test/data/bsm_unbalanced.newick")
out <- list()
rng_out <- list()
for (name in names(cases)) {
  tr <- read.tree(cases[[name]])
  for (seed in c(1, 7, 12345)) {
    set.seed(seed)
    x <- simulate_biogeog_history(tr, q, coo, 1)
    rng_out[[length(rng_out) + 1]] <- data.frame(tree = name, seed = seed, index = 1:8, value = sprintf("%.17g", runif(8)))
    out[[length(out) + 1]] <- data.frame(tree = name, seed = seed,
                                         node = seq_along(x), state = x + 1)
  }
}
write.table(do.call(rbind, out), "test/baselines/forward_r.tsv", sep = "\t", row.names = FALSE, quote = FALSE)
options(digits = 17)
write.table(do.call(rbind, rng_out), "test/baselines/forward_rng_r.tsv", sep = "\t", row.names = FALSE, quote = FALSE)
