
# This simulates a model based on dat and a pseudo-structure of proxies
setwd('~/Documents/UCL')
library(ppcor)
library(xgboost)
library(mclust)

data <- read.csv("econ_x_gen.csv", sep = ",", header = TRUE)    # Data 
dat<-data[ , c(3:11,13:92)]
dat_ir<-data[ ,  c(3:10,12:92)]
A_name <- "X"                                       # Treatment name
Y_name <- "Y"                                       # Outcome name
Z_name <- c("Z0", "Z1")
N_name <- c("N")
M_name <- c("M")
covar_name <- c("Y_covariate0", "Y_covariate1", "Y_covariate2", "Y_covariate3", "Y_covariate4")
U_name <- c("U0", "U1")

#dat_names <- read.csv("/Users/afsaneh/Documents/UCL/Thesis/KPV/Education_data/edu_IM_name.csv", header = FALSE)       # Column names (for reference)
#sel_rows <- which(dat[, which(dat_names == A_name)] == 0) # Only choose children with non-zero dose, simplifies fit
#dat <- dat[-sel_rows, ]
p <- ncol(dat)
pcor_dat <- abs(pcor(dat)$estimate)                       # Matrix of partial correlations

is_binary <- rep(0, p)
for (i in 1:p) {
  is_binary[i] <- length(unique(dat[, i])) == 2
}

A <- which(colnames(dat) == A_name)
  #which(dat_names == A_name)                      # Treatment
Y <- which(colnames(dat) == Y_name) #which(dat_names == Y_name)                      # Outcome
Z <- which(colnames(dat) %in% Z_name) 
  #setdiff(which(pcor_dat[, A] > 0.05), c(A,Y))    # Treatment-inducing confounding proxies

W <- which(colnames(dat) %in% W_name) #<- setdiff(which(pcor_dat[, Y] > 0.05), c(A, Y, Z)) # Outcome-inducing confounding proxies
#XU <- setdiff(1:ncol(dat), c(A, Y, Z, W))            # Common causes
#n_X <- round(0.5 * length(XU))                       # Number of common causes which are observed
#XU_c <- pcor_dat[A, XU] +  pcor_dat[Y, XU]           # Criterion to distinguish X from U
X <- 0#XU[sort(XU_c)[1:n_X]]    #Null                       # Observed common causes are the weakest
U <- which(grepl('x', colnames(dat))) #setdiff(1:ncol(dat), c(A, Y, Z, W)) #setdiff(XU, X)                                  # Hidden common causes

# Generative model: from the ordering, U -> X -> {Z, W} -> A -> Y, pick an arbitrary ordering within
# each set to provide a DAG, then fit a model from nonlinear regression/binary classifier, with
# Gaussian error for the continuous variables

# Structure

DAG_order <- c(U, Z, W, A, Y)
rev_DAG_order <- rep(0, p)
for (i in 1:p) {
  rev_DAG_order[i] <- which(DAG_order == i)
}
DAG_parents <- list()
is_U <- rep(FALSE, p); is_U[rev_DAG_order[U]] <- TRUE
is_Z <- rep(FALSE, p); is_Z[rev_DAG_order[Z]] <- TRUE
is_W <- rep(FALSE, p); is_W[rev_DAG_order[W]] <- TRUE
is_A <- rep(FALSE, p); is_A[rev_DAG_order[A]] <- TRUE
is_Y <- rep(FALSE, p); is_Y[rev_DAG_order[Y]] <- TRUE
for (i in 2:p) {
  if (is_U[i]) {
    DAG_parents[[i]] <- intersect(DAG_order[1:(i - 1)], U)
  }  else if (is_Z[i]) {
    DAG_parents[[i]] <- intersect(DAG_order[1:(i - 1)], c(U, X, Z))
  } else if (is_W[i]) {
    DAG_parents[[i]] <- intersect(DAG_order[1:(i - 1)], c(U, X, W))
  } else if (is_A[i]) {
    DAG_parents[[i]] <- intersect(DAG_order[1:(i - 1)], c(U, X, Z, W, A))
  } else {
    DAG_parents[[i]] <- intersect(DAG_order[1:(i - 1)], c(U, X, Z, W, A, Y))
  }
}

# Parameters

f <- list() # Regression function
v <- list() # Error model (for continuous variables)
K <- 2      # Number of mixture components

f[[1]] <- mean(dat[, DAG_order[[1]]]) 
if (!is_binary[DAG_order[1]]) {
  v[[1]] <- Mclust(dat[, DAG_order[1]], G = K, model = "V")
}
for (i in 2:p) {
  cat("Fitting model for variable", i, "\n")
  in_i <- as.matrix(dat[, DAG_parents[[i]]])
  out_i <- dat[, DAG_order[i]]
  if (is_binary[DAG_order[i]]) {
    obj_i <- "reg:logistic"
  } else {
    obj_i <- "reg:squarederror"
  }
  f[[i]] <- xgboost(data = in_i, label = out_i, nrounds = 100, objective = obj_i, verbose = FALSE)
  if (!is_binary[DAG_order[i]]) {
    f_hat <- predict(f[[i]], in_i)
    v[[i]] <- Mclust(out_i - f_hat, G = K, model = "V")
  }
}

# Simulate

mog_sample <- function(n, model) {
  # Sample from a mixture of Gaussians
  probs <- model$parameters$pro
  means <- as.numeric(model$parameters$mean)
  sds <- sqrt(model$parameters$variance$sigmasq)
  K <- length(probs)
  z <- sample(1:K, n, prob = probs, replace = TRUE)
  s <- rnorm(n) * sds[z] + means[z]
}

n_out <- 1000

confound_me_more_please <- TRUE         # What further unmeasured confounding
noise_level <- rep(10, p)                # How much to confound in each continuous variable
noise_level[is_binary == TRUE] <- 1      # How much to confound in each discrete variable
nosy_noise <- rnorm(n_out)               # The extra hidden factor



sample_from_model <- function(n_out, is_binary, DAG_order, DAG_parents,
                              f, v, confound_me_more_please, noise_level, nosy_noise)
{
  p <- length(DAG_order)
  dat_out <- matrix(rep(0, n_out * p), ncol = p)
  
  if (is_binary[DAG_order[1]]) {
    dat_out[, DAG_order[1]] <- as.numeric(runif(n_out) < f[[1]])
  } else {
    dat_out[, DAG_order[1]] <- mog_sample(n_out, v[[1]])
  }

  for (i in 2:p) {
    
    in_i <- as.matrix(dat_out[, DAG_parents[[i]]])
    pred_i <- predict(f[[i]], in_i)
    if (is_binary[DAG_order[i]]) {
      if (confound_me_more_please) {
        pred_i <- 1 / (1 + exp(-((log(pred_i / (1 - pred_i)) + nosy_noise * noise_level[DAG_order[i]]))))
      }
      dat_out[, DAG_order[i]] <- as.numeric(runif(n_out) < pred_i)
      print(paste('succes:', i))
    } else {
      if (confound_me_more_please) {
        pred_i <- pred_i + nosy_noise * noise_level[DAG_order[i]]
      }
      dat_out[, DAG_order[i]] <- pred_i + mog_sample(n_out, v[[i]])
      print(paste('succes:', i))
    }
  }
  
  return(dat_out)
}

dat_out <- sample_from_model(n_out, is_binary, DAG_order, DAG_parents,
                             f, v, confound_me_more_please, noise_level, nosy_noise)

# Diagnostic plot

j <- Y
quartz() # For Mac computers, call X11() in general 
par(mfrow = c(1, 3)) 
hist(dat[, j], main = paste("True data:", as.character(dat_names[j, 1])))
hist(dat_out[, j], main = paste("Fake data:", as.character(dat_names[j, 1])))
qqplot(dat[, j], dat_out[, j])
par(mfrow = c(1, 1))

# Construct Monte Carlo representation of ATE curve

sample_from_intervened_model <- function(n_out, is_binary, 
                                 DAG_order, DAG_parents, rev_DAG_order,
                                 f, v, confound_me_more_please, noise_level, nosy_noise,
                                 A, Y, A_levels)
{
  p <- length(DAG_order)
  dat_out <- matrix(rep(0, n_out * p), ncol = p)
  num_levels <- length(A_levels)
  ATE_curve <- rep(0, num_levels)
  
  if (is_binary[DAG_order[1]]) {
    dat_out[, DAG_order[1]] <- as.numeric(runif(n_out) < f[[1]])
  } else {
    dat_out[, DAG_order[1]] <- mog_sample(n_out, v[[1]])
  }
  
  for (i in 2:p) {
    
    if (DAG_order[i] == A || DAG_order[i] == Y) {
      next
    }
    
    in_i <- as.matrix(dat_out[, DAG_parents[[i]]])
    pred_i <- predict(f[[i]], in_i)
    if (is_binary[DAG_order[i]]) {
      if (confound_me_more_please) {
        pred_i <- 1 / (1 + exp(-((log(pred_i / (1 - pred_i)) + nosy_noise * noise_level[DAG_order[i]]))))
      }
      dat_out[, DAG_order[i]] <- as.numeric(runif(n_out) < pred_i)
    } else {
      if (confound_me_more_please) {
        pred_i <- pred_i + nosy_noise * noise_level[DAG_order[i]]
      }
      dat_out[, DAG_order[i]] <- pred_i + mog_sample(n_out, v[[i]])
    }
  }

  for (do_A in 1:num_levels) {
    dat_out[, A] <- A_levels[do_A]
    i <- rev_DAG_order[Y]
    in_i <- as.matrix(dat_out[, DAG_parents[[i]]])
    pred_i <- predict(f[[i]], in_i)
    if (is_binary[Y]) {
      if (confound_me_more_please) {
        pred_i <- 1 / (1 + exp(-((log(pred_i / (1 - pred_i)) + nosy_noise * noise_level[DAG_order[i]]))))
      }
      dat_out[, Y] <- as.numeric(runif(n_out) < pred_i)
    } else {
      if (confound_me_more_please) {
        pred_i <- pred_i + nosy_noise * noise_level[Y]
      }
      dat_out[, Y] <- pred_i + mog_sample(n_out, v[[i]])
    }
    ATE_curve[do_A] <- mean(dat_out[, Y])
  } 
  
  return(ATE_curve)
}

num_mc <- 100000                   # Number of Monte Carlo samples
A_levels <- sort(unique(dat[, A])) # Intervention levels evaluated
num_levels <- length(A_levels)
ATE_curve <- sample_from_intervened_model(num_mc, is_binary, 
                                         DAG_order, DAG_parents, rev_DAG_order,
                                         f, v, confound_me_more_please, noise_level, nosy_noise,
                                         A, Y, A_levels)
quartz()
plot(A_levels, ATE_curve, type = "l", main = "ATE curve", xlab = "A", ylab = "Y")

#saveRDS(f, ‘structural_equations.rds’)
#saveRDS(f, ‘structural_equations.rds’)
#class(dat_out)
#library(data.table)

#fwrite(as.data.table(dat_out), '/Users/afsaneh/Documents/UCL/Thesis/KPV/ihdp/simulation_output.csv')
#fwrite(as.data.table(ATE_curve), '/Users/afsaneh/Documents/UCL/Thesis/KPV/ihdp/ATE_curve.csv')
#fwrite(as.data.table(A_levels), '/Users/afsaneh/Documents/UCL/Thesis/KPV/ihdp/do_A.csv')