# Run from the repository root: Rscript test/reference/generate_pruning_order.R
library(ape)
trees <- c(
  "((a:1,b:1):1,(c:1,d:1):1);",
  "(((a:1,b:1):1,c:1):1,((d:1,e:1):1,f:1):1);",
  "((((a:1,b:1):1,c:1):1,d:1):1,(e:1,f:1):1);",
  "(((a:0,b:0.0000001):1,(c:1,d:1):1):1,(e:1,f:1):1);",
  "((a:1,b:1):1,(((c:1,d:1):1,e:1):1,f:1):1);"
)
lines <- vapply(trees, function(s) {
  tr <- reorder(read.tree(text = s), "pruningwise")
  paste(s, paste(unique(tr$edge[, 1]), collapse = ","), sep = "\t")
}, "")
writeLines(c("newick\tinternal_order", lines), "test/baselines/bsm_pruning_order.tsv")
