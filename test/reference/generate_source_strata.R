# Run from the repository root with BioGeoBEARS 1.1.4 installed.
suppressMessages(library(BioGeoBEARS))
areas <- c("A", "B", "C")
young <- matrix(1, 3, 3); old <- matrix(1, 3, 3)
young[1, 3] <- 0; old[2, 3] <- 0
times <- c(1, 3); dmats <- list(young, old)
clado <- data.frame(time_bp = c(0.5, 1, 2),
                    clado_event_txt = rep("AB->AB,C", 3),
                    clado_dispersal_to = rep("C", 3))
ana <- data.frame(abs_event_time = c(0.5, 1, 2),
                  event_txt = rep("AB->ABC", 3), dispersal_to = rep("C", 3))
set.seed(42)
clado <- BioGeoBEARS:::simulate_source_area_clado(clado, areas, dmat = dmats, times = times)
ana <- BioGeoBEARS:::simulate_source_area_ana(ana, areas, dmat = dmats, times = times)
write.table(data.frame(kind = rep(c("clado", "ana"), each = 3),
                       time = rep(c(0.5, 1, 2), 2),
                       source = c(clado$clado_dispersal_from, ana$ana_dispersal_from)),
            "test/baselines/source_strata.tsv", sep = "\t", quote = FALSE, row.names = FALSE)
young[1, 3] <- 0.2; young[2, 3] <- 0.8
old[1, 3] <- 0.8; old[2, 3] <- 0.2
dmats <- list(young, old)
clado <- data.frame(time_bp = rep(c(0.5, 1, 2), 4),
                    clado_event_txt = rep("AB->AB,C", 12), clado_dispersal_to = rep("C", 12))
ana <- data.frame(abs_event_time = rep(c(0.5, 1, 2), 4),
                  event_txt = rep("AB->ABC", 12), dispersal_to = rep("C", 12))
set.seed(42)
clado <- BioGeoBEARS:::simulate_source_area_clado(clado, areas, dmat = dmats, times = times)
ana <- BioGeoBEARS:::simulate_source_area_ana(ana, areas, dmat = dmats, times = times)
writeLines(c("source", clado$clado_dispersal_from, ana$ana_dispersal_from),
           "test/baselines/source_strata_weighted.txt")
