# Run from the repository root with BioGeoBEARS, ape, cladoRcpp and minqa.
suppressMessages(library(BioGeoBEARS))
suppressMessages(library(ape))
options(digits = 17)
paths <- file.path("test/data", paste0("multitree_fossil", 1:2, ".newick"))
trees <- lapply(paths, read.tree)
tipranges <- getranges_from_LagrangePHYLIP("test/data/multitree_fossil.data")
areas <- getareas_from_tipranges_object(tipranges)
states <- cladoRcpp::rcpp_areas_list_to_states_list(areas, maxareas = 3, include_null_range = TRUE)
tops <- bottoms <- fits <- list()
for (i in seq_along(trees)) {
  ro <- define_BioGeoBEARS_run()
  ro$trfn <- paths[i]; ro$geogfn <- "test/data/multitree_fossil.data"
  ro$timesfn <- "test/data/bsm_small_times.txt"
  ro$areas_allowed_fn <- "test/data/bsm_small_areas.txt"
  ro$max_range_size <- 3; ro$include_null_range <- TRUE; ro$allow_null_tips <- FALSE
  ro$use_detection_model <- FALSE; ro$num_cores_to_use <- 1
  ro$printlevel <- 0; ro$force_sparse <- FALSE
  ro$return_condlikes_table <- TRUE; ro$calc_TTL_loglike_from_condlikes_table <- TRUE
  ro <- readfiles_BioGeoBEARS_run(ro)
  ro <- section_the_tree(inputs = ro, make_master_table = TRUE, plot_pieces = FALSE)
  phy <- reorder(trees[[i]], "pruningwise")
  tc <- BioGeoBEARS:::tipranges_to_tip_condlikes_of_data_on_each_state(tipranges = tipranges, phy = phy, include_null_range = TRUE)
  ro$lists_of_states_lists_0based <- lapply(ro$list_of_areas_allowed_mats, function(m)
    BioGeoBEARS:::prune_states_list(states_list_0based_index = states, areas_allowed_mat = m))
  ro$master_states_list <- states
  ro$state_indices_0based_all_timeperiods <- states
  ro$all_geog_states_list_usually_inferred_from_areas_maxareas <- states
  likelihood <- function(x, anc = FALSE) {
    p <- ro$BioGeoBEARS_model_object@params_table
    p["d", c("init", "est")] <- x[1]; p["e", c("init", "est")] <- x[2]
    ro$BioGeoBEARS_model_object@params_table <- p
    invisible(capture.output(res <- BioGeoBEARS:::calc_loglike_sp_stratified(
      tc, phy, Qmat = NULL, spPmat = NULL, min_branchlength = 1e-6, return_what = "all",
      probs_of_states_at_root = NULL, rootedge = TRUE, sparse = FALSE, printlevel = 0,
      use_cpp = TRUE, input_is_COO = FALSE, spPmat_inputs = NULL, cppSpMethod = 3,
      cluster_already_open = NULL, calc_ancprobs = anc, include_null_range = TRUE,
      fixnode = NULL, fixlikes = NULL, inputs = ro, allareas = areas, all_states_list = states,
      return_condlikes_table = TRUE, calc_TTL_loglike_from_condlikes_table = TRUE)))
    res
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
  "test/data/bsm_balanced.newick", do.call(rbind, tops), do.call(rbind, bottoms)))))
write.table(do.call(rbind, fits), "test/baselines/multitree_fossil_strat_fit.tsv", sep = "\t", row.names = FALSE, quote = FALSE)
write.table(data.frame(matches = summary$nummatches_at_node, summary$stateprobs_nodes_master_tr,
                       summary$stateprobs_corners_master_tr), "test/baselines/multitree_fossil_strat_summary.tsv",
            sep = "\t", row.names = FALSE, quote = FALSE, na = "NA")
