# Run from the repository root with BioGeoBEARS 1.1.4 and ape installed.
suppressMessages(library(BioGeoBEARS))
suppressMessages(library(ape))
options(digits = 17)
trees <- lapply(c("((a:1,b:1):1,(c:1,d:1):1);",
                  "((b:1,a:1):1,(d:1,c:1):1);",
                  "((a:1,c:1):1,(b:1,d:1):1);",
                  "(((a:0.5,x:0.5):0.5,b:1):1,(c:1,d:1):1);"),
                function(x) read.tree(text = x))
rows <- do.call(rbind, lapply(seq_along(trees), function(i) {
  table <- prt(trees[[i]], printflag = FALSE, get_tipnames = TRUE)
  p <- (seq_len(nrow(table)) + i) / 30
  data.frame(s1 = p, s2 = 1 - p, tipnames = table$tipnames)
}))
rows <- as.matrix(rows)
master <- tempfile(fileext = ".newick")
write.tree(trees[[1]], file = master)
invisible(capture.output(ref <- suppressWarnings(BioGeoBEARS:::summarize_stateprobs_on_master_tree(master, rows, rows))))
out <- data.frame(matches = ref$nummatches_at_node, ref$stateprobs_nodes_master_tr)
write.table(out, "test/baselines/multiple_trees_r.tsv", sep = "\t", row.names = FALSE, quote = FALSE, na = "NA")
