# Generate R SVEMnet reference results for cross-validating the Python package.
# Run with the SVEMnet R package (>= 3.5.0) installed, e.g.:
#   Rscript validation/run_r_reference.R <output_dir>
# Writes designs, responses, groups, RNG uniforms, and fitted results as
# plain CSVs with 17 significant digits.

args <- commandArgs(trailingOnly = TRUE)
out_dir <- if (length(args)) args[[1]] else file.path("validation", "output")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

library(SVEMnet)
stopifnot(packageVersion("SVEMnet") >= "3.5.0")

fmt <- function(x) format(x, digits = 17, scientific = TRUE, trim = TRUE)

write_matrix <- function(m, path) {
  m <- as.matrix(m)
  if (is.null(colnames(m))) colnames(m) <- paste0("V", seq_len(ncol(m)))
  df <- as.data.frame(
    matrix(fmt(as.numeric(m)), nrow = nrow(m), ncol = ncol(m)),
    stringsAsFactors = FALSE
  )
  colnames(df) <- colnames(m)
  utils::write.csv(df, path, row.names = FALSE)
}

write_vector <- function(v, path, name = "value") {
  df <- data.frame(v = fmt(as.numeric(v)), stringsAsFactors = FALSE)
  names(df) <- name
  utils::write.csv(df, path, row.names = FALSE)
}

# Export the design matrix (intercept dropped), response, and whole-term
# groups (R model.matrix 'assign') for one formula/data pair.
export_design <- function(tag, formula, data) {
  mf <- stats::model.frame(formula, data)
  X_full <- stats::model.matrix(formula, mf)
  assign_full <- attr(X_full, "assign")
  term_labels <- attr(attr(mf, "terms"), "term.labels")
  int_idx <- which(colnames(X_full) == "(Intercept)")
  X <- X_full[, -int_idx, drop = FALSE]
  assign_x <- assign_full[-int_idx]
  y <- stats::model.response(mf)

  write_matrix(X, file.path(out_dir, paste0(tag, "_X.csv")))
  write_vector(y, file.path(out_dir, paste0(tag, "_y.csv")), "y")
  groups <- data.frame(
    term = term_labels[assign_x],
    column = seq_len(ncol(X)) - 1L,  # 0-based for Python
    stringsAsFactors = FALSE
  )
  utils::write.csv(groups, file.path(out_dir, paste0(tag, "_groups.csv")),
                   row.names = FALSE)
  list(X = X, y = y, formula = formula, data = data)
}

export_forward_aicc <- function(tag, formula, data) {
  for (crit in c("AICc", "AIC", "BIC")) {
    fit <- forward_aicc(formula, data, criterion = crit)
    utils::write.csv(
      data.frame(name = names(fit$parms), value = fmt(fit$parms),
                 stringsAsFactors = FALSE),
      file.path(out_dir, paste0(tag, "_faicc_", crit, "_coef.csv")),
      row.names = FALSE
    )
    utils::write.csv(
      data.frame(
        selected = if (length(fit$selected_terms)) fit$selected_terms else character(0),
        stringsAsFactors = FALSE
      ),
      file.path(out_dir, paste0(tag, "_faicc_", crit, "_terms.csv")),
      row.names = FALSE
    )
    write_vector(fit$criterion_value,
                 file.path(out_dir, paste0(tag, "_faicc_", crit, "_value.csv")),
                 "criterion_value")
  }
}

export_svem_forward_identity <- function(tag, formula, data) {
  fit <- svem_forward(formula, data, nBoot = 1, weight_scheme = "Identity")
  write_matrix(fit$coef_matrix,
               file.path(out_dir, paste0(tag, "_sf_identity_coef.csv")))
}

export_svem_forward_seeded <- function(tag, formula, data, seed, nBoot) {
  n <- nrow(stats::model.frame(formula, data))
  set.seed(seed)
  U <- matrix(stats::runif(nBoot * n), nrow = nBoot, byrow = TRUE)
  set.seed(seed)
  fit <- svem_forward(formula, data, nBoot = nBoot, weight_scheme = "SVEM",
                      objective = "wAIC")
  write_matrix(U, file.path(out_dir, paste0(tag, "_sf_seeded_U.csv")))
  write_matrix(fit$coef_matrix,
               file.path(out_dir, paste0(tag, "_sf_seeded_coef.csv")))
}

export_svemnet_lasso <- function(tag, formula, data, seed, nBoot) {
  set.seed(seed)
  fit <- SVEMnet(formula, data, nBoot = nBoot, glmnet_alpha = 1,
                 relaxed = FALSE, objective = "wAIC")
  utils::write.csv(
    data.frame(name = names(fit$parms), value = fmt(fit$parms),
               stringsAsFactors = FALSE),
    file.path(out_dir, paste0(tag, "_lasso_coef.csv")),
    row.names = FALSE
  )
  write_vector(fit$y_pred, file.path(out_dir, paste0(tag, "_lasso_pred.csv")),
               "y_pred")
}

## ---- Dataset 1: numeric quadratic-interaction surface -----------------------
set.seed(101)
n <- 40
d1 <- data.frame(X1 = rnorm(n), X2 = rnorm(n), X3 = rnorm(n))
d1$y <- 1 + 2 * d1$X1 - 1.5 * d1$X2 + 1.2 * d1$X1 * d1$X2 +
  0.8 * d1$X1^2 + rnorm(n, 0, 0.3)
f1 <- y ~ (X1 + X2 + X3)^2 + I(X1^2) + I(X2^2) + I(X3^2)
export_design("d1", f1, d1)
export_forward_aicc("d1", f1, d1)
export_svem_forward_identity("d1", f1, d1)
export_svem_forward_seeded("d1", f1, d1, seed = 202, nBoot = 30)
export_svemnet_lasso("d1", f1, d1, seed = 303, nBoot = 100)

## ---- Dataset 2: three-level factor with interaction -------------------------
set.seed(102)
n <- 60
d2 <- data.frame(
  X1 = rnorm(n), X2 = rnorm(n),
  F = factor(sample(c("a", "b", "c"), n, TRUE))
)
d2$y <- 1 + 2 * d2$X1 - d2$X2 + 1.5 * d2$X1 * d2$X2 +
  2 * (d2$F == "b") - 1.5 * (d2$F == "c") + rnorm(n, 0, 0.3)
f2 <- y ~ X1 + X2 + F + X1:X2
export_design("d2", f2, d2)
export_forward_aicc("d2", f2, d2)
export_svem_forward_identity("d2", f2, d2)
export_svem_forward_seeded("d2", f2, d2, seed = 404, nBoot = 30)

## ---- Dataset 3: small n, many candidates (ceiling exercised) ----------------
set.seed(103)
n <- 15
d3 <- as.data.frame(matrix(rnorm(n * 6), n, 6))
names(d3) <- paste0("X", 1:6)
d3$y <- 1 + 2 * d3$X1 - 1.5 * d3$X2 + rnorm(n, 0, 0.3)
f3 <- y ~ X1 + X2 + X3 + X4 + X5 + X6
export_design("d3", f3, d3)
export_forward_aicc("d3", f3, d3)
export_svem_forward_identity("d3", f3, d3)
export_svem_forward_seeded("d3", f3, d3, seed = 505, nBoot = 30)

cat("R reference export complete:", out_dir, "\n")
cat("SVEMnet version:", as.character(packageVersion("SVEMnet")), "\n")
cat("R version:", R.version.string, "\n")
