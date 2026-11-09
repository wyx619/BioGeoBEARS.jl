# Run from the repository root with BioGeoBEARS, ape, cladoRcpp and minqa.
suppressMessages(library(BioGeoBEARS))
suppressMessages(library(ape))
options(digits = 17)
paths <- file.path("test/data", paste0("bsm_", c("balanced", "unbalanced"), ".newick"))
trees <- lapply(paths, read.tree)
tipranges <- getranges_from_LagrangePHYLIP("test/data/bsm_small.data")
areas <- getareas_from_tipranges_object(tipranges)
states <- cladoRcpp::rcpp_areas_list_to_states_list(areas, maxareas = 3, include_null_range = TRUE)
tops <- list(); bottoms <- list(); fits <- list()
for (i in seq_along(trees)) {
  ro <- define_BioGeoBEARS_run()
  ro$trfn <- paths[i]; ro$geogfn <- "test/data/bsm_small.data"
  ro$max_range_size <- 3; ro$include_null_range <- TRUE
  ro$use_detection_model <- FALSE; ro$timesfn <- NA
  ro$num_cores_to_use <- 1; ro$printlevel <- 0; ro$force_sparse <- FALSE
  phy <- reorder(trees[[i]], "pruningwise")
  tc <- BioGeoBEARS:::tipranges_to_tip_condlikes_of_data_on_each_state(tipranges = tipranges, phy = phy, include_null_range = TRUE)
  likelihood <- function(x, anc = FALSE) {
    invisible(capture.output(res <- calc_loglike_for_optim(params = c(d = x[1], e = x[2]),
      BioGeoBEARS_run_object = ro, phy = phy, tip_condlikes_of_data_on_each_state = tc,
      print_optim = FALSE, calc_ancprobs = anc, return_what = "all",
      areas_list = seq_along(areas) - 1, states_list = states, force_sparse = FALSE,
      cluster_already_open = NULL)))
    if (is.list(res)) res else list(total_loglikelihood = res)
  }
  opt <- minqa::bobyqa(c(0.1, 0.05), function(x) -likelihood(x)$total_loglikelihood,
    lower = c(1e-8, 1e-8), upper = c(3, 3), control = list(maxfun = 2000, rhobeg = 0.02, rhoend = 1e-6))
  stopifnot(opt$ierr == 0)
  res <- likelihood(opt$par, TRUE)
  labels <- prt(trees[[i]], printflag = FALSE, get_tipnames = TRUE)$tipnames
  tops[[i]] <- cbind(res$ML_marginal_prob_each_state_at_branch_top_AT_node, labels)
  bottoms[[i]] <- cbind(res$ML_marginal_prob_each_state_at_branch_bottom_below_node, labels)
  fits[[i]] <- data.frame(tree = i, d = opt$par[1], e = opt$par[2], lnL = -opt$fval)
}
invisible(capture.output(summary <- suppressWarnings(BioGeoBEARS:::summarize_stateprobs_on_master_tree(
  paths[1], do.call(rbind, tops), do.call(rbind, bottoms)))))
write.table(do.call(rbind, fits), "test/baselines/multiple_trees_fit_r.tsv", sep = "\t", row.names = FALSE, quote = FALSE)
write.table(data.frame(matches = summary$nummatches_at_node, summary$stateprobs_nodes_master_tr,
                       summary$stateprobs_corners_master_tr), "test/baselines/multiple_trees_fit_summary_r.tsv",
            sep = "\t", row.names = FALSE, quote = FALSE, na = "NA")
