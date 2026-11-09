# Run from the repository root with BioGeoBEARS, ape and cladoRcpp.
suppressMessages(library(BioGeoBEARS))
suppressMessages(library(ape))
models <- c("DEC", "DEC+J", "DIVALIKE", "DIVALIKE+J", "BAYAREALIKE", "BAYAREALIKE+J")
out <- list()
for (model in models) {
  ro <- define_BioGeoBEARS_run()
  ro$trfn <- "test/data/bsm_balanced.newick"; ro$geogfn <- "test/data/bsm_small.data"
  ro$max_range_size <- 3; ro$include_null_range <- TRUE; ro$timesfn <- NA
  ro$printlevel <- 0; ro$force_sparse <- FALSE; ro$num_cores_to_use <- 1
  ro <- readfiles_BioGeoBEARS_run(ro)
  pt <- ro$BioGeoBEARS_model_object@params_table
  pt["d", "est"] <- 0.3; pt["e", "est"] <- 0.1
  pt["j", "est"] <- if (grepl("\\+J", model)) 0.1 else 0
  if (grepl("^DIVALIKE", model)) {
    pt["s", "type"] <- "fixed"; pt["s", "est"] <- 0
    pt["ysv", "type"] <- "2-j"; pt["ys", "type"] <- "ysv*1/2"
    pt["y", "type"] <- "ysv*1/2"; pt["v", "type"] <- "ysv*1/2"
    pt["mx01v", "type"] <- "fixed"; pt["mx01v", "est"] <- 0.5
  }
  if (grepl("^BAYAREALIKE", model)) {
    pt["s", "type"] <- "fixed"; pt["s", "est"] <- 0
    pt["v", "type"] <- "fixed"; pt["v", "est"] <- 0
    pt["ysv", "type"] <- "1-j"; pt["ys", "type"] <- "ysv*1/1"; pt["y", "type"] <- "1-j"
    pt["mx01y", "type"] <- "fixed"; pt["mx01y", "est"] <- 0.9999
  }
  ro$BioGeoBEARS_model_object@params_table <- pt
  ro$BioGeoBEARS_model_object <- calc_linked_params_BioGeoBEARS_model_object(ro$BioGeoBEARS_model_object)
  invisible(capture.output(mats <- get_Qmat_COOmat_from_BioGeoBEARS_run_object(ro)))
  for (tree in c("balanced", "unbalanced")) for (seed in c(1, 7, 12345)) {
    tr <- read.tree(file.path("test/data", paste0("bsm_", tree, ".newick")))
    set.seed(seed)
    x <- simulate_biogeog_history(tr, mats$Qmat, mats$COO_weights_columnar, 1)
    out[[length(out) + 1]] <- data.frame(model = model, tree = tree, seed = seed,
                                         node = seq_along(x), state = x + 1)
  }
}
write.table(do.call(rbind, out), "test/baselines/forward_models_r.tsv", sep = "\t", row.names = FALSE, quote = FALSE)
