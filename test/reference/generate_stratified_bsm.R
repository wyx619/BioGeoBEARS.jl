# Run from the repository root with BioGeoBEARS 1.1.4 and minqa installed.
# Output defaults to temp/; pass an existing directory as the first argument.
suppressMessages(library(BioGeoBEARS)); suppressMessages(library(ape)); suppressMessages(library(cladoRcpp))

d0 <- "test/data/psychotria_m0strat"
args <- commandArgs(trailingOnly = TRUE)
outdir <- if (length(args)) args[1] else "temp"
stopifnot(dir.exists(outdir))
seed <- if (length(args) >= 2) as.integer(args[2]) else 12345L
d_rate <- if (length(args) >= 3) as.numeric(args[3]) else 0.1
e_rate <- if (length(args) >= 4) as.numeric(args[4]) else 0.05
models <- if (length(args) >= 5) args[5] else c("DIVALIKE", "DIVALIKE+J", "BAYAREALIKE", "BAYAREALIKE+J")
small_tree <- if (length(args) >= 6) args[6] else NULL
areas_mode <- if (length(args) >= 7) args[7] else "allareas"
fit_reference <- length(args) >= 8 && args[8] == "fit"
rng_reference <- length(args) >= 8 && args[8] == "rng"

run_one <- function(model) {
  ro <- define_BioGeoBEARS_run()
  ro$trfn <- file.path(d0, "Psychotria_5.2.newick")
  ro$geogfn <- file.path(d0, "Psychotria_geog.data")
  ro$timesfn <- file.path(d0, "timeperiods.txt")
  ro$areas_allowed_fn <- file.path(d0, "areas_allowed_all1s.txt")
  ro$max_range_size <- 4
  if (!is.null(small_tree)) {
    ro$trfn <- file.path("test/data", paste0("bsm_", small_tree, ".newick"))
    ro$geogfn <- "test/data/bsm_small.data"
    ro$timesfn <- "test/data/bsm_small_times.txt"
    ro$areas_allowed_fn <- file.path("test/data", paste0("bsm_small_", areas_mode, ".txt"))
    ro$max_range_size <- 3
  }
  ro$include_null_range <- TRUE
  ro$allow_null_tips <- FALSE
  ro$use_detection_model <- FALSE
  ro$num_cores_to_use <- 1
  ro$printlevel <- 0
  ro$force_sparse <- FALSE
  ro$return_condlikes_table <- TRUE
  ro$calc_TTL_loglike_from_condlikes_table <- TRUE
  ro$calc_ancprobs <- TRUE
  ro <- readfiles_BioGeoBEARS_run(ro)
  ro <- section_the_tree(inputs = ro, make_master_table = TRUE, plot_pieces = FALSE)

  pt <- ro$BioGeoBEARS_model_object@params_table
  pt["d", "est"] <- d_rate; pt["d", "init"] <- d_rate
  pt["e", "est"] <- e_rate; pt["e", "init"] <- e_rate
  if (grepl("\\+J", model)) { pt["j", "est"] <- 0.1; pt["j", "init"] <- 0.1 }
  if (model %in% c("DIVALIKE", "DIVALIKE+J")) {
    pt["s", "type"] <- "fixed"; pt["s", "init"] <- 0; pt["s", "est"] <- 0
    pt["ysv", "type"] <- "2-j"; pt["ys", "type"] <- "ysv*1/2"
    pt["y", "type"] <- "ysv*1/2"; pt["v", "type"] <- "ysv*1/2"
    pt["mx01v", "type"] <- "fixed"; pt["mx01v", "init"] <- 0.5
    pt["mx01v", "est"] <- 0.5
  }
  if (model %in% c("BAYAREALIKE", "BAYAREALIKE+J")) {
    pt["s", "type"] <- "fixed"; pt["s", "init"] <- 0; pt["s", "est"] <- 0
    pt["v", "type"] <- "fixed"; pt["v", "init"] <- 0; pt["v", "est"] <- 0
    pt["ysv", "type"] <- "1-j"; pt["ys", "type"] <- "ysv*1/1"
    pt["y", "type"] <- "1-j"
    pt["mx01y", "type"] <- "fixed"; pt["mx01y", "init"] <- 0.9999
    pt["mx01y", "est"] <- 0.9999
  }
  ro$BioGeoBEARS_model_object@params_table <- pt

  phy <- reorder(read.tree(ro$trfn), "pruningwise")
  tipranges <- getranges_from_LagrangePHYLIP(ro$geogfn)
  tipcl <- BioGeoBEARS:::tipranges_to_tip_condlikes_of_data_on_each_state(
    tipranges = tipranges, phy = phy, include_null_range = TRUE)
  areas <- getareas_from_tipranges_object(tipranges)
  sl <- cladoRcpp::rcpp_areas_list_to_states_list(areas = areas, maxareas = ro$max_range_size,
                                                  include_null_range = TRUE)
  lists <- list()
  for (ti in 1:length(ro$timeperiods)) {
    lists[[ti]] <- BioGeoBEARS:::prune_states_list(
      states_list_0based_index = sl,
      areas_allowed_mat = ro$list_of_areas_allowed_mats[[ti]])
  }
  ro$lists_of_states_lists_0based <- lists
  ro$master_states_list <- sl
  ro$state_indices_0based_all_timeperiods <- sl
  ro$all_geog_states_list_usually_inferred_from_areas_maxareas <- sl

  if (fit_reference) {
    with_j <- grepl("\\+J", model)
    objective <- function(x) {
      params <- ro$BioGeoBEARS_model_object@params_table
      params["d", c("init", "est")] <- x[1]
      params["e", c("init", "est")] <- x[2]
      if (with_j) params["j", c("init", "est")] <- x[3]
      ro$BioGeoBEARS_model_object@params_table <- params
      ro$BioGeoBEARS_model_object <- calc_linked_params_BioGeoBEARS_model_object(ro$BioGeoBEARS_model_object)
      invisible(capture.output(z <- BioGeoBEARS:::calc_loglike_sp_stratified(
        tipcl, phy, Qmat = NULL, spPmat = NULL, min_branchlength = 1e-6,
        return_what = "all", probs_of_states_at_root = NULL, rootedge = TRUE,
        sparse = FALSE, printlevel = 0, use_cpp = TRUE, input_is_COO = FALSE,
        spPmat_inputs = NULL, cppSpMethod = 3, cluster_already_open = NULL,
        calc_ancprobs = FALSE, include_null_range = TRUE, fixnode = NULL,
        fixlikes = NULL, inputs = ro, allareas = areas, all_states_list = sl,
        return_condlikes_table = TRUE, calc_TTL_loglike_from_condlikes_table = TRUE)))
      -z$total_loglikelihood
    }
    start <- c(d_rate, e_rate, if (with_j) 0.1)
    lower <- c(1e-8, 1e-8, if (with_j) 1e-5)
    j_upper <- if (model == "DIVALIKE+J") 2 else if (model == "BAYAREALIKE+J") 1 else 3
    upper <- c(3, 3, if (with_j) j_upper - 1e-5)
    if (model == "DEC") {
      opt <- optim(start, objective, method = "L-BFGS-B", lower = lower, upper = upper,
                   control = list(maxit = 300, factr = 1e7))
    } else {
      result <- minqa::bobyqa(start, objective, lower = lower, upper = upper,
                             control = list(maxfun = 3000, rhobeg = 0.02, rhoend = 1e-6))
      opt <- list(par = result$par, value = result$fval, convergence = result$ierr)
    }
    stopifnot(opt$convergence == 0, all(is.finite(opt$par)))
    params <- ro$BioGeoBEARS_model_object@params_table
    params["d", c("init", "est")] <- opt$par[1]
    params["e", c("init", "est")] <- opt$par[2]
    if (with_j) params["j", c("init", "est")] <- opt$par[3]
    ro$BioGeoBEARS_model_object@params_table <- params
    ro$BioGeoBEARS_model_object <- calc_linked_params_BioGeoBEARS_model_object(ro$BioGeoBEARS_model_object)
    fit_tag <- gsub("\\+", "J", model)
    header <- if (model == "DEC") "d\te\tlnL" else "d\te\tj\tlnL"
    values <- if (model == "DEC") c(opt$par, -opt$value) else
      c(opt$par[1:2], if (with_j) opt$par[3] else 0, -opt$value)
    writeLines(c(header, paste(sprintf("%.17g", values), collapse="\t")),
               file.path(outdir, paste0("stratified_fit_", fit_tag, ".tsv")))
  }

  tmpres <- BioGeoBEARS:::calc_loglike_sp_stratified(tipcl, phy, Qmat = NULL,
    spPmat = NULL, min_branchlength = 1e-6, return_what = "all",
    probs_of_states_at_root = NULL, rootedge = TRUE, sparse = FALSE,
    printlevel = 0, use_cpp = TRUE, input_is_COO = FALSE, spPmat_inputs = NULL,
    cppSpMethod = 3, cluster_already_open = NULL, calc_ancprobs = TRUE,
    include_null_range = TRUE, fixnode = NULL, fixlikes = NULL, inputs = ro,
    allareas = areas, all_states_list = sl, return_condlikes_table = TRUE,
    calc_TTL_loglike_from_condlikes_table = TRUE)

  res <- list(inputs = ro, cluster_already_open = NULL,
              state_indices_0based_all_timeperiods = sl,
              lnL = tmpres$total_loglikelihood)
  for (nm in names(tmpres)) res[[nm]] <- tmpres[[nm]]
  ml <- tmpres$ML_marginal_prob_each_state_at_branch_top_AT_node
  tag <- gsub("\\+", "J", model)
  if (!is.null(small_tree)) tag <- paste(small_tree, areas_mode, tag, seed, sep = "_")
  writeLines(c("node", sapply(seq_len(nrow(ml)), function(i)
    paste(sprintf("%.17g", ml[i, ]), collapse = "\t"))),
    file.path(outdir, paste0("r_model_", tag, "_marg.txt")))
  smi <- BioGeoBEARS:::get_inputs_for_stochastic_mapping(res, printlevel = 0)
  out <- capture.output(sm <- BioGeoBEARS:::stochastic_mapping_on_stratified(
    res, smi, maxtries = 40000, seedval = seed))
  if (rng_reference) {
    writeLines(c("uniform", sprintf("%.17g", runif(16))),
               file.path(outdir, paste0("r_model_", tag, "_rng.txt")))
    writeLines(c("lnL", sprintf("%.17g", tmpres$total_loglikelihood)),
               file.path(outdir, paste0("r_model_", tag, "_lnL.txt")))
  }
  if (fit_reference && model == "DEC") {
    invisible(capture.output(sm2 <- BioGeoBEARS:::stochastic_mapping_on_stratified(
      res, smi, maxtries = 40000, seedval = seed + 1)))
    dm <- matrix(1, nrow = length(areas), ncol = length(areas))
    clado <- data.frame(time_bp = c(0.5, 1, 2),
                        clado_event_txt = rep("AB->AB,C", 3), clado_dispersal_to = rep("C", 3))
    ana <- data.frame(abs_event_time = c(0.5, 1, 2),
                      event_txt = rep("AB->ABC", 3), dispersal_to = rep("C", 3))
    clado <- BioGeoBEARS:::simulate_source_area_clado(clado, areas, dmat = dm)
    ana <- BioGeoBEARS:::simulate_source_area_ana(ana, areas, dmat = dm)
    writeLines(c("source", clado$clado_dispersal_from, ana$ana_dispersal_from),
               file.path(outdir, "stratified_batch_sources.txt"))
    invisible(capture.output(sm <- BioGeoBEARS:::stochastic_mapping_on_stratified(
      res, smi, maxtries = 40000, seedval = seed)))
    writeLines(c("uniform", sprintf("%.17g", runif(16))),
               file.path(outdir, "stratified_fit_rng.txt"))
  }

  mt <- sm$master_table_cladogenetic_events
  writeLines(c("node\tstratum\tpiecenum\tpiececlass\tsampled_states_AT_nodes\tsampled_states_AT_brbots\tclado_event_type\tclado_event_txt",
    apply(mt, 1, function(r) paste(r[c("node","stratum","piecenum","piececlass",
      "sampled_states_AT_nodes","sampled_states_AT_brbots","clado_event_type",
      "clado_event_txt")], collapse = "\t"))),
    file.path(outdir, paste0("r_model_", tag, "_map.tsv")))
  at <- sm$table_w_anagenetic_events
  con <- file(file.path(outdir, paste0("r_model_", tag, "_ana.tsv")), "w")
  if (is.null(at) || is.null(nrow(at))) {
    writeLines("node\tabs_event_time\tevent_type\tevent_txt", con)
  } else {
  writeLines(paste(c("row", names(at)), collapse = "\t"), con)
  for (i in seq_len(nrow(at))) {
    lens <- sapply(at, function(col) length(col[[i]]))
    ne <- max(lens)
    for (k in seq_len(ne)) {
      vals <- sapply(names(at), function(nm) {
        v <- at[[nm]][[i]]
        if (length(v) >= k) as.character(v[k]) else ""
      })
      writeLines(paste(c(i, vals), collapse = "\t"), con)
    }
  }
  }
  close(con)
  if (fit_reference && model != "DEC") {
    for (suffix in c("marg.txt", "map.tsv", "ana.tsv")) {
      file.copy(file.path(outdir, paste0("r_model_", tag, "_", suffix)),
                file.path(outdir, paste0("stratified_fit_", fit_tag, "_", suffix)),
                overwrite = TRUE)
    }
  }
  if (fit_reference) {
    batch_dir <- tempfile("biogeo_bsm_")
    dir.create(batch_dir)
    invisible(capture.output(batch <- runBSM(res, smi, maxnum_maps_to_try = 2,
      nummaps_goal = 2, save_after_every_try = FALSE, savedir = batch_dir,
      seedval = 0, wait_before_save = 0)))
    stopifnot(length(batch$RES_clado_events_tables) == 2)
    batch_prefix <- file.path(outdir, paste0("stratified_batch_", fit_tag))
    writeLines(c("uniform", sprintf("%.17g", runif(16))), paste0(batch_prefix, "_rng.txt"))
    clado <- batch$RES_clado_events_tables
    ana <- batch$RES_ana_events_tables
    for (i in seq_along(clado)) {
      mt <- clado[[i]]
      write.table(mt[, c("node", "stratum", "piececlass", "sampled_states_AT_nodes", "clado_event_txt")],
                  paste0(batch_prefix, "_map", i, ".tsv"), sep = "\t", row.names = FALSE, quote = FALSE, na = "NA")
      at <- ana[[i]]
      con <- file(paste0(batch_prefix, "_ana", i, ".tsv"), "w")
      writeLines("node\tabs_event_time\tevent_txt", con)
      if (is.data.frame(at)) for (row in seq_len(nrow(at))) {
        if (length(at$abs_event_time[[row]])) for (k in seq_along(at$abs_event_time[[row]])) {
          writeLines(paste(at$node[[row]], sprintf("%.17g", at$abs_event_time[[row]][k]),
                           at$event_txt[[row]][k], sep = "\t"), con)
        }
      }
      close(con)
    }
    # Reset to the stream left by runBSM, before the diagnostic draws above.
    invisible(capture.output(batch <- runBSM(res, smi, maxnum_maps_to_try = 2,
      nummaps_goal = 2, save_after_every_try = FALSE, savedir = batch_dir,
      seedval = 0, wait_before_save = 0)))
    dm <- matrix(1, length(areas), length(areas))
    for (i in seq_along(clado)) {
      clado[[i]] <- BioGeoBEARS:::simulate_source_area_clado(clado[[i]], areas, dmat = dm)
      ana[[i]] <- BioGeoBEARS:::simulate_source_area_ana(ana[[i]], areas, dmat = dm)
    }
    invisible(capture.output(counts <- count_ana_clado_events(clado, ana, areas, areas)))
    write.table(counts$summary_counts_BSMs, paste0(batch_prefix, "_summary.tsv"),
                sep = "\t", row.names = FALSE, quote = FALSE)
    if (model == "DEC") {
      invisible(capture.output(window_counts <- count_ana_clado_events(clado, ana, areas, areas, timeperiod = c(1, 2))))
      write.table(window_counts$summary_counts_BSMs, paste0(batch_prefix, "_window.tsv"),
                  sep = "\t", row.names = FALSE, quote = FALSE)
    }
  }
  cat(sprintf("MODEL %-14s lnL %.12f  map dumped\n", model,
              tmpres$total_loglikelihood))
}

for (mdl in models) {
  run_one(mdl)
}
cat("done\n")
