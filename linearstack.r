source(file.path("evalUtils.r", fsep = .Platform$file.sep))
source(file.path("MixRF.R", fsep = .Platform$file.sep))

# Environment preparation
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
library(h2o)
library(deepnet)

h2o.init()

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

# # Correlation and redundancy
# vcobj <- varclus(~., data = all_projects[,c("fix", metrics)], similarity = "spearman", trans = "abs")
# plot(vcobj)
# threshold <- 0.7
# abline(h = 1 - threshold, col = "red", lty = 2, lwd = 2)

# redun_obj <- redun(~ relative_churn + ns + norm_entropy + nf + fix + lt, data = all_projects, nk=5)
# paste(redun_obj$Out, collapse =", ")

scale_metrics <- c("ns","nf","lt","norm_entropy","relative_churn")

for (i in 1:length(project_names)) {
  projects[[i]]$relative_churn <- 0
  tmp_relative_churn <- (projects[[i]]$la + projects[[i]]$ld) / projects[[i]]$lt # (la+ld)/lt
  projects[[i]][projects[[i]]$lt >= 1, "relative_churn"] <- tmp_relative_churn[projects[[i]]$lt >= 1]
  projects[[i]][is.na(projects[[i]])] <- 0
  # projects[[i]][,scale_metrics] <- lapply(projects[[i]][,scale_metrics], scale) # Data scaling
}

all_projects <- projects[[1]]
for (i in 2:length(project_names)) {
  all_projects <- rbind(all_projects, projects[[i]])
} 
all_projects$project <- as.factor(all_projects$project) # Merged dataset

#############################
i=20;

print("PROJECT NUMBER")        
print(i)
print("inside1")

training_set <- subset(all_projects, all_projects$project != project_names[i])
testing_set <- subset(all_projects, all_projects$project == project_names[i])

training_df <- as.h2o(training_set)
test_df <- as.h2o(testing_set)

print("inside2")

# Random forest
print("random forest")

y <- "contains_bug"
x <- c("fix", "ns", "nf", "norm_entropy", "relative_churn", "lt")

rf_model <- h2o.randomForest(x = x,
                y = y,
                training_frame = training_df,
                nfolds = 5,
                keep_cross_validation_predictions = TRUE,
                seed = 5)

predict_rf <- predict(rf_model, training_df);
predict_rf1 <- predict(rf_model, test_df);

# print(h2o.auc(h2o.performance(rf_model, newdata=test_df)))

# # xgboost
# print("xgboost")

# y <- "contains_bug"
# x <- c("fix", "ns", "nf", "norm_entropy", "relative_churn", "lt")

# xgb_model <- h2o.xgboost(x = x,
#                 y = y,
#                 training_frame = training_df,
#                 nfolds = 5,
#                 keep_cross_validation_predictions = TRUE,
#                 seed = 5)

# predict_xgb <- predict(xgb_model, training_df);
# predict_xgb1 <- predict(xgb_model, test_df);
# print(h2o.auc(h2o.performance(xgb_model, newdata=test_df)))


# Project Aware
print("Train project aware")
tmp_project_aware_lr_model <- glmer(contains_bug ~ (norm_entropy | project) + fix + ns + nf + relative_churn + lt, 
                                    data = training_set, family = "binomial")
print("Trained project aware model")

project_aware_trained <- predict(tmp_project_aware_lr_model, training_set, allow.new.levels = TRUE, type = "response")
project_aware_trained_corrected <- predict(tmp_project_aware_lr_model, training_set, allow.new.levels = TRUE, type = "link")
project_aware_trained_corrected <- project_aware_trained_corrected - coef(summary(tmp_project_aware_lr_model))[1,1] + median(coef(tmp_project_aware_lr_model)[[1]][,2]) + training_set$norm_entropy * median(coef(tmp_project_aware_lr_model)[[1]][,1])
project_aware_trained_corrected <- inv.logit(project_aware_trained_corrected)
  
tmp_project_aware_lr_pred <- predict(tmp_project_aware_lr_model, testing_set, allow.new.levels = TRUE, type = "response")
tmp_project_aware_lr_pred_corrected <- predict(tmp_project_aware_lr_model, testing_set, allow.new.levels = TRUE, type = "link")
tmp_project_aware_lr_pred_corrected <- tmp_project_aware_lr_pred_corrected - coef(summary(tmp_project_aware_lr_model))[1,1] + median(coef(tmp_project_aware_lr_model)[[1]][,2]) + testing_set$norm_entropy * median(coef(tmp_project_aware_lr_model)[[1]][,1])
tmp_project_aware_lr_pred_corrected <- inv.logit(tmp_project_aware_lr_pred_corrected)

print("completed project aware")

# evalPredict(testing_set$contains_bug, tmp_project_aware_lr_pred_corrected, testing_set$loc)


# ## Ensemble
# pred_comb <- matrix(as.vector(as.numeric(predict_rf$TRUE)), ncol=1)
# pred_comb <- cbind(pred_comb, as.vector(as.numeric(predict_xgb$TRUE)))
# pred_comb <- cbind(pred_comb, as.vector(as.numeric(project_aware_trained_corrected)))
# pred_comb <- cbind(pred_comb, as.numeric(training_set$contains_bug))

# colnames(pred_comb) <- c("predict_rf", "predict_xgb", "predict_project_aware", "contains_bug")

# head(pred_comb)

# ens_model <- glm(contains_bug ~ predict_rf + predict_xgb + project_aware , data = training_set, family="binomial")

# yy1 <- predict(ens_model, newdata = testing_set, type = "response")
# print(evalPredict(testing_set$contains_bug, yy1, testing_set$loc))

## Ensemble
pred_comb <- matrix(as.vector(as.numeric(predict_rf[,3])), ncol=1)
# pred_comb <- cbind(pred_comb, as.vector(as.numeric(predict_xgb[,3])))
pred_comb <- cbind(pred_comb, as.vector(as.numeric(project_aware_trained_corrected)))


head(pred_comb)

x <- pred_comb
y <- as.numeric(training_set$contains_bug)

dnn <- dbn.dnn.train(x, y, hidden = c(1), activationfun = "sigm")
print("TRAINED NN")


pred_comb1 <- matrix(as.vector(as.numeric(predict_rf1[,3])), ncol=1)
# pred_comb1 <- cbind(pred_comb1, as.vector(as.numeric(predict_xgb1[,3])))
pred_comb1 <- cbind(pred_comb1, as.vector(as.numeric(tmp_project_aware_lr_pred_corrected)))

print("NN Ensemble Output on Test Set")
yy1 <- nn.predict(dnn, pred_comb1)

## Performace
print(evalPredict(testing_set$contains_bug, yy1, testing_set$loc))