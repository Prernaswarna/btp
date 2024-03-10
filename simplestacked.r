
library(h2o)
library(readr)
library(mefa)
library(lme4)
library(lmerTest)
library(Hmisc)
library(car)
library(sjPlot)
library(optimx)
library(MuMIn)
library(boot)
library(plyr)
library(doParallel)
library(caret)
library(ranger)
library(ROCR)
library(data.table)

registerDoParallel(cores = detectCores())

# Data preparation
context <- read.csv(file.path("data","context.csv", fsep = .Platform$file.sep), row.names = 1)
context$TLOC <- factor(cut(context$LOC, quantile(context$LOC), include.lowest = TRUE), labels = c("Least", "Less", "More", "Most"))
context$NFILES <- factor(cut(context$nfile, quantile(context$nfile), include.lowest = TRUE), labels = c("Least", "Less", "More", "Most"))
context$NCOMMIT <- factor(cut(context$ncommit, quantile(context$ncommit), include.lowest = TRUE), labels = c("Least", "Less", "More", "Most"))
context$NDEV <- factor(cut(context$ndev, quantile(context$ndev), include.lowest = TRUE), labels = c("Least", "Less", "More", "Most"))
context$nlanguage <- factor(cut(context$nlanguage, c(1,1.9,3), include.lowest = TRUE), labels = c("Least", "Most"))  # There are only 3 values (1,2,3) for this vector, hence we only distinguish 1 and >1
context$git <- NULL
context$LOC <- NULL
context$nfile <- NULL
context$ncommit <- NULL
context$ndev <- NULL

project_names <- row.names(context)
print(length(project_names))
metrics <- c("ns","nd","nf","la","ld","lt","norm_entropy")

projects <- list()
for (i in 1:length(project_names)) {
  projects[[i]] <- read_csv(file.path("data",paste(project_names[i], ".csv", sep = ''), 
                                      fsep = .Platform$file.sep),
                            col_types = cols(contains_bug = col_logical(), 
                                             fix = col_logical(), author_date = col_skip(), 
                                             author_date_unix_timestamp = col_skip(), 
                                             author_email = col_skip(), author_name = col_skip(), 
                                             classification = col_skip(), commit_hash = col_skip(), 
                                             commit_message = col_skip(), fileschanged = col_skip(), 
                                             fixes = col_skip(), glm_probability = col_skip(), 
                                             linked = col_skip(), repository_id = col_skip(),
                                             ndev = col_skip(), age = col_skip(), nuc = col_skip(),
                                             exp = col_skip(), rexp = col_skip(), sexp = col_skip()))
  
  projects[[i]]$loc <- projects[[i]]$la + projects[[i]]$ld
  
  projects[[i]]$norm_entropy <- 0
  tmp_norm_entropy <- projects[[i]]$entrophy / sapply(projects[[i]]$nf, log2) # Normalize entropy
  projects[[i]][projects[[i]]$nf >= 2, "norm_entropy"] <- tmp_norm_entropy[projects[[i]]$nf >= 2]
  
  projects[[i]]$project <- project_names[i]
  
  projects[[i]] <- cbind(projects[[i]], rep(context[project_names[i],], times = nrow(projects[[i]])))
} 

# Correlation and redundancy
#vcobj <- varclus(~., data = all_projects[,c("fix", metrics)], similarity = "spearman", trans = "abs")
#plot(vcobj)
#threshold <- 0.7
#abline(h = 1 - threshold, col = "red", lty = 2, lwd = 2)

#redun_obj <- redun(~ relative_churn + ns + norm_entropy + nf + fix + lt, data = all_projects, nk=5)
#paste(redun_obj$Out, collapse =", ")

scale_metrics <- c("ns","nf","lt","norm_entropy","relative_churn")

for (i in 1:length(project_names)) {
  projects[[i]]$relative_churn <- 0
  tmp_relative_churn <- (projects[[i]]$la + projects[[i]]$ld) / projects[[i]]$lt # (la+ld)/lt
  projects[[i]][projects[[i]]$lt >= 1, "relative_churn"] <- tmp_relative_churn[projects[[i]]$lt >= 1]
  projects[[i]][is.na(projects[[i]])] <- 0
  projects[[i]][,scale_metrics] <- lapply(projects[[i]][,scale_metrics], scale) # Data scaling
}

all_projects <- projects[[1]]
for (i in 2:length(project_names)) {
  all_projects <- rbind(all_projects, projects[[i]])
} 
all_projects$project <- as.factor(all_projects$project) # Merged dataset
# Split the data frame into Train and Test dataset
## 75% of the sample size
smp_size <- floor(0.75 * nrow(all_projects))
## set the seed to make your partition reproducible
set.seed(10)
train_ind <- sample(seq_len(nrow(all_projects)), size = smp_size)
all_projects$relative_churn <- as.vector(all_projects$relative_churn)
all_projects$ns <- as.vector(all_projects$ns)
all_projects$nf <- as.vector(all_projects$nf)
all_projects$lt <- as.vector(all_projects$lt)
all_projects$norm_entropy <- as.vector(all_projects$norm_entropy)
train_df <- all_projects[train_ind, ]
test_df <- all_projects[-train_ind, ]

str(train_df)
# initialize the h2o
h2o.init()
# create the train and test h2o data frames
train_df_h2o<-as.h2o(train_df)
test_df_h2o<-as.h2o(test_df)
# Identify predictors and response
y <- "contains_bug"
x <- c("ns","nf","lt","norm_entropy","relative_churn", "fix")
# Number of CV folds (to generate level-one data for stacking)
nfolds <- 5
# 1. Generate a 3-model ensemble (GBM + RF + Logistic)
# Train & Cross-validate a GBM
my_gbm <- h2o.gbm(x = x,
                  y = y,
                  training_frame = train_df_h2o,
                  nfolds = nfolds,
                  keep_cross_validation_predictions = TRUE,
                  seed = 10)
# Train & Cross-validate a RF
my_rf <- h2o.randomForest(x = x,
                          y = y,
                          training_frame = train_df_h2o,
                          nfolds = nfolds,
                          keep_cross_validation_predictions = TRUE,
                          seed = 10)
# Train & Cross-validate a LR
my_lr <- h2o.glm(x = x,
                 y = y,
                 training_frame = train_df_h2o,
                 family = c("binomial"),
                 nfolds = nfolds,
                 keep_cross_validation_predictions = TRUE,
                 seed = 10)

typeof(my_lr)
# Train a stacked random forest ensemble using the GBM, RF and LR above
ensemble <- h2o.stackedEnsemble(x = x,
                                y = y,
                                metalearner_algorithm="drf",
                                training_frame = train_df_h2o,
                                base_models = list(my_gbm, my_rf, my_lr))
# Eval ensemble performance on a test set
perf <- h2o.performance(ensemble, newdata = test_df_h2o)
# Compare to base learner performance on the test set
perf_gbm_test <- h2o.performance(my_gbm, newdata = test_df_h2o)
perf_rf_test <- h2o.performance(my_rf, newdata = test_df_h2o)
perf_lr_test <- h2o.performance(my_lr, newdata = test_df_h2o)
baselearner_best_auc_test <- max(h2o.auc(perf_gbm_test), h2o.auc(perf_rf_test), h2o.auc(perf_lr_test))
ensemble_auc_test <- h2o.auc(perf)
print(sprintf("GBM Test AUC: %s", h2o.auc(perf_gbm_test)))
print(sprintf("Random Forest AUC: %s", h2o.auc(perf_rf_test)))
print(sprintf("Logistic regression AUC: %s", h2o.auc(perf_lr_test)))
print(sprintf("Best Base-learner Test AUC:  %s", baselearner_best_auc_test))
print(sprintf("Ensemble Test AUC:  %s", ensemble_auc_test))