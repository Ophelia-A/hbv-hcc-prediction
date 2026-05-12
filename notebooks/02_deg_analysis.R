# 02_deg_analysis.R
# DEG Analysis for GSE14520 using limma
# Comparing Tumor vs Non-Tumor tissue in HBV patients

library(GEOquery)
library(limma)

#  1. Load data from GEO 
cat("Downloading GSE14520...\n")
gset <- getGEO("GSE14520", GSEMatrix = TRUE, getGPL = FALSE)

#  2. Load labels from Python output
labels <- read.csv("C:/Users/vaio/Desktop/DCIT400/hbv-hcc-prediction/data/labels.csv",
                   row.names = 1)

# 3. Process each platform separately
results_list <- list()

for (i in seq_along(gset)) {
  
  platform <- annotation(gset[[i]])
  cat("\nProcessing platform:", platform, "\n")
  
  ex <- exprs(gset[[i]])
  
  # Log2 transform if needed
  qx <- quantile(ex, c(0.25, 0.99), na.rm = TRUE)
  if (qx[2] > 100 || (qx[2] - qx[1]) > 50) {
    ex[ex <= 0] <- NaN
    ex <- log2(ex)
  }
  
  # Match samples to our labels
  common <- intersect(colnames(ex), rownames(labels))
  cat("Matched samples:", length(common), "\n")
  
  if (length(common) < 10) next
  
  ex <- ex[, common]
  group <- factor(labels[common, "label"],
                  levels = c(0, 1),
                  labels = c("NonTumor", "Tumor"))
  
  # limma DEG analysis
  design <- model.matrix(~ 0 + group)
  colnames(design) <- levels(group)
  
  fit <- lmFit(ex, design)
  contrast <- makeContrasts(Tumor - NonTumor, levels = design)
  fit2 <- contrasts.fit(fit, contrast)
  fit2 <- eBayes(fit2)
  
  degs <- topTable(fit2, number = Inf, adjust.method = "BH")
  degs$platform <- platform
  results_list[[platform]] <- degs
}

# ── 4. Combine and save results ────────────────────────────
all_degs <- do.call(rbind, results_list)
output_path <- "C:/Users/vaio/Desktop/DCIT400/hbv-hcc-prediction/data/deg_results.csv"
write.csv(all_degs, output_path)
cat("\nDEG results saved to:", output_path, "\n")
cat("Total probes:", nrow(all_degs), "\n")

