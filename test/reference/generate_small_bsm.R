# Run from the repository root: Rscript test/reference/generate_small_bsm.R
# Requires R BioGeoBEARS 1.1.4, ape and cladoRcpp.
suppressMessages(library(BioGeoBEARS))
suppressMessages(library(ape))

exdir <- "test/data"
outdir <- "test/baselines"

make_res <- function(d, e, j, tree) {
  ro <- define_BioGeoBEARS_run()
  ro$trfn <- file.path(exdir, paste0("bsm_", tree, ".newick"))
  ro$geogfn <- file.path(exdir, "bsm_small.data")
  ro$max_range_size <- 3
  ro$include_null_range <- TRUE
  ro$use_detection_model <- FALSE
  ro$timesfn <- NA
  ro$num_cores_to_use <- 1
  ro$printlevel <- 0
  ro$force_sparse <- FALSE
  pt <- ro$BioGeoBEARS_model_object@params_table
  pt["d", "est"] <- d; pt["d", "init"] <- d
  pt["e", "est"] <- e; pt["e", "init"] <- e
  pt["j", "est"] <- j; pt["j", "init"] <- j
  pt["j", "type"] <- "free"
  ro$BioGeoBEARS_model_object@params_table <- pt

  phy <- reorder(read.tree(ro$trfn), "pruningwise")
  tipranges <- getranges_from_LagrangePHYLIP(ro$geogfn)
  tipcl <- BioGeoBEARS:::tipranges_to_tip_condlikes_of_data_on_each_state(
    tipranges = tipranges, phy = phy, include_null_range = TRUE)
  areas <- getareas_from_tipranges_object(tipranges)
  states_list <- cladoRcpp::rcpp_areas_list_to_states_list(
    areas = areas, maxareas = 3, include_null_range = TRUE)

  mr <- calc_loglike_for_optim(params = c(d = d, e = e, j = j),
    BioGeoBEARS_run_object = ro, phy = phy,
    tip_condlikes_of_data_on_each_state = tipcl, print_optim = FALSE,
    calc_ancprobs = TRUE, return_what = "all",
    areas_list = seq(0, length(areas) - 1, 1), states_list = states_list,
    force_sparse = FALSE, cluster_already_open = NULL)

  list(inputs = ro, lnL = mr$total_loglikelihood,
    ML_marginal_prob_each_state_at_branch_top_AT_node =
      mr$ML_marginal_prob_each_state_at_branch_top_AT_node,
    relative_probs_of_each_state_at_branch_bottom_below_node_DOWNPASS =
      mr$relative_probs_of_each_state_at_branch_bottom_below_node_DOWNPASS,
    relative_probs_of_each_state_at_branch_top_AT_node_DOWNPASS =
      mr$relative_probs_of_each_state_at_branch_top_AT_node_DOWNPASS,
    condlikes = mr$condlikes, cluster_already_open = NULL)
}

parse_events <- function(txt) {
  if (is.na(txt) || txt == "none") return(NULL)
  rows <- strsplit(txt, ";", fixed = TRUE)[[1]]
  out <- lapply(rows, function(r) {
    kv <- strsplit(r, ",", fixed = TRUE)[[1]]
    kv <- kv[grepl(":", kv, fixed = TRUE)]
    key <- sub(":.*$", "", kv)
    val <- sub("^[^:]*:", "", kv)
    setNames(as.list(val), key)
  })
  do.call(rbind, lapply(out, function(x) as.data.frame(x, stringsAsFactors = FALSE)))
}

# R renumbers internal nodes pruningwise (root = ntips+1); Julia keeps ape's
# numbering.  The descendant-tip signature lets the test map one to the other.
desc_sig <- function(nd, tbl) {
  row <- which(tbl$node == nd)
  kids <- c(tbl$left_desc_nodes[row], tbl$right_desc_nodes[row])
  kids <- kids[!is.na(kids)]
  if (length(kids) == 0) return(as.character(nd))
  paste(sort(unlist(lapply(kids, desc_sig, tbl = tbl))), collapse = "-")
}

cases <- list(c("balanced_dec", "balanced", 0.1, 0.05, 0),
              c("unbalanced_dec", "unbalanced", 0.3, 0.1, 0),
              c("short_dec", "short", 0.1, 0.05, 0),
              c("balanced_decj", "balanced", 0.1, 0.05, 0.1),
              c("unbalanced_decj", "unbalanced", 0.3, 0.1, 0.1))
seeds <- c(1, 7, 12345)

tt_rows <- character(0)
ana_rows <- character(0)

for (cs in cases) {
  tag <- cs[1]; tree <- cs[2]; d <- as.numeric(cs[3]); e <- as.numeric(cs[4]); j <- as.numeric(cs[5])
  res <- make_res(d, e, j, tree)
  smi <- BioGeoBEARS:::get_inputs_for_stochastic_mapping(res, printlevel = 0)
  for (s in seeds) {
    invisible(capture.output(
      out <- BioGeoBEARS:::stochastic_map_given_inputs(
        stochastic_mapping_inputs = smi, maxtries = 100000,
        seedval = s, include_null_range = TRUE)))
    for (nd in 1:nrow(out)) {
      tt_rows <- c(tt_rows, sprintf("%s\t%d\t%d\t%d\t%d\t%d\t%s\t%s\t%s\t%s", tag, s, nd,
        out$sampled_states_AT_nodes[nd],
        ifelse(is.na(out$sampled_states_AT_brbots[nd]), -1, out$sampled_states_AT_brbots[nd]),
        ifelse(is.na(out$samp_LEFT_dcorner[nd]), -1, out$samp_LEFT_dcorner[nd]),
        ifelse(is.na(out$samp_RIGHT_dcorner[nd]), -1, out$samp_RIGHT_dcorner[nd]),
        ifelse(is.na(out$clado_event_type[nd]), "", out$clado_event_type[nd]),
        ifelse(is.na(out$clado_event_txt[nd]), "", out$clado_event_txt[nd]),
        desc_sig(out$node[nd], out)))
      ev <- parse_events(out$anagenetic_events_txt_below_node[nd])
      if (!is.null(ev)) {
        for (k in seq_len(nrow(ev))) {
          ana_rows <- c(ana_rows, sprintf("%s\t%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s", tag, s,
            desc_sig(out$node[nd], out),
            ev$current_rangetxt[k], ev$new_rangetxt[k],
            sprintf("%.17g", as.numeric(ev$event_time[k])),
            sprintf("%.17g", as.numeric(ev$abs_event_time[k])),
            ev$event_type[k], ev$dispersal_to[k]))
        }
      }
    }
  }
}

writeLines(c("case\tseed\tnode\tnode_state\tbrbot_state\tleft_corner\tright_corner\tclado_type\tclado_txt\tdesc", tt_rows),
           file.path(outdir, "bsm_small_trtable.tsv"))
writeLines(c("case\tseed\tdesc\tfrom_range\tto_range\tevent_time\tabs_event_time\tevent_type\tdispersal_to", ana_rows),
           file.path(outdir, "bsm_small_ana.tsv"))
cat("trtable rows:", length(tt_rows), " ana rows:", length(ana_rows), "\n")

