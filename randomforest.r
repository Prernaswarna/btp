# Set working directory to that of the current file
#setwd(dirname(rstudioapi::getActiveDocumentContext()$path))  # Only when using RStudio
source(file.path("evalUtils.r", fsep = .Platform$file.sep))
source(file.path("MixRF.R", fsep = .Platform$file.sep))

# Environment preparation
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
library(randomForest)

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

all_projects$relative_churn <- as.vector(all_projects$relative_churn)
all_projects$ns <- as.vector(all_projects$ns)
all_projects$nf <- as.vector(all_projects$nf)
all_projects$lt <- as.vector(all_projects$lt)
all_projects$norm_entropy <- as.vector(all_projects$norm_entropy)

h2o.init()
# for (i in 19:length(project_names)) {
#   training_set <- subset(all_projects, all_projects$project != project_names[i])
#   testing_set <- subset(all_projects, all_projects$project == project_names[i])
#   train_df_h2o<-as.h2o(training_set)
#   test_df_h2o<-as.h2o(testing_set)
#   print("first break")
#   y <- "contains_bug"
#   x <- c("ns","nf","lt","norm_entropy","relative_churn", "fix")
  
#   my_rf <- h2o.randomForest(x = x,
#                           y = y,
#                           training_frame = train_df_h2o,
#                           )
#   print("trained model")
#   print(h2o.predict(my_rf, newdata=test_df_h2o))
#   perf_rf_test <- h2o.performance(my_rf, newdata = test_df_h2o)
#   print(h2o.auc(perf_rf_test))
# }

  training_set <- subset(all_projects, all_projects$project != project_names[20])
  testing_set <- subset(all_projects, all_projects$project == project_names[20])
  train_df_h2o<-as.h2o(training_set)
  test_df_h2o<-as.h2o(testing_set)
  print("first break")
  y <- "contains_bug"
  x <- c("ns","nf","lt","norm_entropy","relative_churn", "fix")
  
  my_rf <- h2o.randomForest(x = x,
                          y = y,
                          training_frame = train_df_h2o,
                          nfolds = 5,
                  keep_cross_validation_predictions = TRUE,
                  seed = 5
                          )

rf_pred<- h2o.predict(my_rf, newdata=test_df_h2o)


training_set_euc <- subset(all_projects, all_projects$project==project_names[15]|all_projects$project==project_names[19])
testing_set_euc <- subset(all_projects, all_projects$project == project_names[20])
print("got euc sets")
print(nrow(testing_set_euc))
tmp_project_aware_lr_model_euc <- glmer(contains_bug ~ (norm_entropy | project) + fix + ns + nf + relative_churn + lt, 
                                     data = training_set_euc, nAGQ=0, family = "binomial")
#tmp_project_aware_lr_model_euc <- glm(contains_bug ~ fix + ns + nf + norm_entropy + relative_churn + lt, data = training_set_euc, family="binomial")
# tmp_project_aware_lr_pred_euc <- predict(tmp_project_aware_lr_model_euc, testing_set_euc, allow.new.levels = TRUE, type = "response")
# print(length(tmp_project_aware_lr_pred_euc))

tmp_project_aware_lr_pred_corrected_euc <- predict(tmp_project_aware_lr_model_euc, testing_set_euc, allow.new.levels = TRUE, type = "link")
  tmp_project_aware_lr_pred_corrected_euc <- tmp_project_aware_lr_pred_corrected_euc - coef(summary(tmp_project_aware_lr_model_euc))[1,1] + median(coef(tmp_project_aware_lr_model_euc)[[1]][,2]) + testing_set_euc$norm_entropy * median(coef(tmp_project_aware_lr_model_euc)[[1]][,1])
  tmp_project_aware_lr_pred_corrected_euc <- inv.logit(tmp_project_aware_lr_pred_corrected_euc)

pred1 <- as.numeric(rf_pred[,3])
pred1_vector <- as.vector(pred1)
pred2_vector <- as.numeric(tmp_project_aware_lr_pred_corrected_euc)
# pred2 <- data.frame(tmp_project_aware_lr_pred_corrected_euc)
# sortedId <- order(tmp_project_aware_lr_pred_corrected_euc, increasing=TRUE)
# print(head(tmp_project_aware_lr_pred_corrected_euc))
# result = c()
# for(i in 1:nrow(testing_set_euc)) {
#   print(tmp_project_aware_lr_pred_corrected_euc[i])
#   print(pred1[i])
#   print(tmp_project_aware_lr_pred_corrected_euc[i]+pred1[i])
#   val<-0.5*tmp_project_aware_lr_pred_corrected_euc[i]+0.5*pred1[i]
#   result <- append(result, val)
# }
# print(head(sortedId))
# new_pred <- 0.5*pred1+0.5*tmp_project_aware_lr_pred_corrected_euc
# print(pred1)
# # print(head(tmp_project_aware_lr_pred_euc, 5))
print(head(pred1_vector))
print(head(pred2_vector))
result <- 0.5*(pred1_vector+pred2_vector)
print(head(result))

print(evalPredict(testing_set$contains_bug, result, testing_set$loc))
