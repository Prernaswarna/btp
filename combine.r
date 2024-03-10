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
library(bestglm)
library(dplyr)
library(brms)
library(rstan)
library(pROC)
library(gbm)

rstan_options(auto_write=TRUE)
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
  projects[[i]]$projectid <- i
  if(isTRUE(projects[[i]]$contains_bug)) {
    projects[[i]]$result=2
  }
  else {
    projects[[i]]$result=1
  }
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
all_projects$project <- as.factor(all_projects$project)

# head(all_projects)

get_project_aware_lr_perf <- function(i, correct = TRUE) {
  training_set <- subset(all_projects, all_projects$project != project_names[i])
  testing_set <- subset(all_projects, all_projects$project == project_names[i])
  print("got sets")
  tmp_project_aware_lr_model <- glmer(contains_bug ~ (norm_entropy | project) + fix + ns + nf + relative_churn + lt, 
                                      data = training_set, nAGQ=0, family = "binomial")
  print("trained model")
  tmp_project_aware_lr_pred <- predict(tmp_project_aware_lr_model, testing_set, allow.new.levels = TRUE, type = "response")
  tmp_project_aware_lr_pred_corrected <- predict(tmp_project_aware_lr_model, testing_set, allow.new.levels = TRUE, type = "link")
  tmp_project_aware_lr_pred_corrected <- tmp_project_aware_lr_pred_corrected - coef(summary(tmp_project_aware_lr_model))[1,1] + median(coef(tmp_project_aware_lr_model)[[1]][,2]) + testing_set$norm_entropy * median(coef(tmp_project_aware_lr_model)[[1]][,1])
  tmp_project_aware_lr_pred_corrected <- inv.logit(tmp_project_aware_lr_pred_corrected)
  return(tmp_project_aware_lr_pred_corrected)
#   if(correct) {
#     return(evalPredict(testing_set$contains_bug, tmp_project_aware_lr_pred_corrected, testing_set$loc))
#   } else {
#     return(evalPredict(testing_set$contains_bug, tmp_project_aware_lr_pred, testing_set$loc))
#   }
}
print("get_lr_perf")
# project_aware_lr_perf_correct <- llply(seq(1, length(project_names), 1), get_project_aware_lr_perf)
project_result<- get_project_aware_lr_perf(1)
project_result_list <- project_result
for (i in 2:length(project_names)) {
    tmp_result<- get_project_aware_lr_perf(i)
    project_result <- rbind(project_result, tmp_result)
    project_result_list <- c(project_result_list, tmp_result)
}

# str(df)
# head(project_result)
length(project_result)
# project_aware_lr_perf_correct <- rbindlist(project_aware_lr_perf_correct, fill=TRUE)
# project_result_df <- as.data.frame(lapply(project_result, unlist))
project_roc <- calcROC(project_result, all_projects$contains_bug)
print(project_roc)

project_confMatrix <- evalConfusionMatrix(all_projects$contains_bug, project_result, 0.5)
print(project_confMatrix$precision)
print(project_confMatrix$recall)
print(project_confMatrix$f)
print(project_confMatrix$gscore)

# head(project_aware_lr_perf_correct)
print("project aware")
# Performance of Context Aware JIT Model (Logistic Regression)
get_context_aware_lr_perf <- function(i, correct = TRUE) {
  training_set <- subset(all_projects, all_projects$project != project_names[i])
  testing_set <- subset(all_projects, all_projects$project == project_names[i])
  print("got sets")
  tmp_context_aware_lr_model <- glmer(contains_bug ~  (1|project) + fix + ns + nf + relative_churn + lt + (1 | language) + (1 | nlanguage) + (1 | TLOC)+ (1 | NFILES) + (1 | NCOMMIT) + (1 | NDEV) + (1 | audience) + (1 | ui) + (1 | database), 
                                      data = training_set,  family = "binomial")
  print("trained model")
  tmp_context_aware_lr_pred <- predict(tmp_context_aware_lr_model, testing_set, allow.new.levels = TRUE, type = "response")
  tmp_context_aware_lr_pred_corrected <- predict(tmp_context_aware_lr_model, testing_set, allow.new.levels = TRUE, type = "link")
  tmp_context_aware_lr_pred_corrected <- tmp_context_aware_lr_pred_corrected - coef(summary(tmp_context_aware_lr_model))[1,1] + median(coef(tmp_context_aware_lr_model)$project[,2]) + testing_set$norm_entropy * median(coef(tmp_context_aware_lr_model)$project[,1])
  if (testing_set$language[1] == 'PHP' || testing_set$language[1] == 'C' || testing_set$language[1] == 'C++' || testing_set$language[1] == 'Perl') {
    tmp_context_aware_lr_pred_corrected <- tmp_context_aware_lr_pred_corrected + median(coef(tmp_context_aware_lr_model)$language[,2])
  }
  tmp_context_aware_lr_pred_corrected <- inv.logit(tmp_context_aware_lr_pred_corrected)
  # print(tmp_context_aware_lr_pred_corrected)
  # print(evalPredict(testing_set$contains_bug, tmp_context_aware_lr_pred_corrected, testing_set$loc))
  # print(length(tmp_context_aware_lr_pred))
  return(tmp_context_aware_lr_pred_corrected)
#   if(correct) {
#     return(evalPredict(testing_set$contains_bug, tmp_context_aware_lr_pred_corrected, testing_set$loc))
#   } else {
#     return(evalPredict(testing_set$contains_bug, tmp_context_aware_lr_pred, testing_set$loc))
#   }
}
print("get_lr_perf")
# context_aware_lr_perf_correct <- llply(seq(1, length(project_names), 1), get_context_aware_lr_perf)
# context_aware_lr_perf_correct <- rbindlist(context_aware_lr_perf_correct, fill=TRUE)

# head(context_aware_lr_perf_correct)

context_result<- get_context_aware_lr_perf(1)
context_result_list <- context_result
for (i in 2:length(project_names)) {
    tmp_result<- get_context_aware_lr_perf(i)
    # print(tmp_result)
    
      context_result <- rbind(context_result, tmp_result)
      context_result_list <- c(context_result_list, tmp_result)
    
}
# head(context_result)
length(context_result)

context_roc <- calcROC(context_result, all_projects$contains_bug)
print(context_roc)

context_confMatrix <- evalConfusionMatrix(all_projects$contains_bug, context_result, 0.5)
print(context_confMatrix$precision)
print(context_confMatrix$recall)
print(context_confMatrix$f)
print(context_confMatrix$gscore)
print("context aware")


combined_model <- cbind(all_projects$contains_bug, all_projects$project)
combined_model <- cbind(combined_model, project_result)
combined_model <- cbind(combined_model, context_result)
# combined_model_data <- data.frame(matrix(unlist(combined_model), nrow=length(combined_model), byrow=TRUE))
combined_model_data <- data.frame(all_projects$contains_bug, all_projects$project, project_result_list, context_result_list)
# head(combined_model_data)
colnames(combined_model_data) <- c('contains_bug', 'project', 'project_result', 'context_result')
# head(combined_model_data)
str(combined_model_data)
print(length(combined_model_data))
# combined_gbm <- gbm(
#   formula = contains_bug ~ project_result + context_result,
#   data = combined_model_data,
#   n.cores = NULL, # will use all cores by default
#   ) 

# print("obtained combined model")

# head(combined_model_data)

# Performance of Context Aware JIT Model (Logistic Regression)
get_combined_lr_perf <- function(i, correct = TRUE) {
  training_set <- subset(combined_model_data, combined_model_data$project != project_names[i])
  testing_set <- subset(combined_model_data, combined_model_data$project == project_names[i])
  # print(str(training_set))
  # print(str(testing_set))
  print("got sets")
  combined_gbm <- gbm(
  formula = contains_bug ~ project_result + context_result,
  data = training_set,
  n.cores = 1, # will use all cores by default
  ) 
  print("trained model")
  print(i)
  tmp_combined_lr_pred <- predict(combined_gbm, testing_set, allow.new.levels = TRUE, type = "response")
  tmp_combined_lr_pred_corrected <- predict(combined_gbm, testing_set, allow.new.levels = TRUE, type = "link")
  tmp_combined_lr_pred_corrected <- tmp_combined_lr_pred_corrected - coef(summary(combined_gbm))[1,1]
  # if (testing_set$language[1] == 'PHP' || testing_set$language[1] == 'C' || testing_set$language[1] == 'C++' || testing_set$language[1] == 'Perl') {
  #   tmp_combined_lr_pred_corrected <- tmp_combined_lr_pred_corrected + median(coef(tmp_combined_lr_model)$language[,2])
  # }
  tmp_combined_lr_pred_corrected <- inv.logit(tmp_combined_lr_pred_corrected)
  # print(tmp_combined_lr_pred)
  print(calcROC(tmp_combined_lr_pred, testing_set$contains_bug))
  # print(head(tmp_combined_lr_pred))
  # print("confusion matrix")
  print(evalConfusionMatrix(testing_set$contains_bug, tmp_combined_lr_pred, 0.5))
  # print(length(tmp_combined_lr_pred_corrected))
  # print("not corrected")
  # print(length(tmp_combined_lr_pred))
  # print(str(tmp_combined_lr_pred))
  return(tmp_combined_lr_pred)
#   if(correct) {
#     return(evalPredict(testing_set$contains_bug, tmp_context_aware_lr_pred_corrected, testing_set$loc))
#   } else {
#     return(evalPredict(testing_set$contains_bug, tmp_context_aware_lr_pred, testing_set$loc))
#   }
}

combined_result<- get_combined_lr_perf(1)
combined_result_list <- context_result
for (i in 2:length(project_names)) {
    tmp_result<- get_combined_lr_perf(i)
    # print("tmp_result length")
    # length(tmp_result)
    # str(tmp_result)
      combined_result <- rbind(combined_result, tmp_result)
      combined_result_list <- c(combined_result_list, tmp_result)    
}
# head(combined_result)
# length(combined_result)
# length(combined_result_list)

# combined_roc <- calcROC(combined_result, combined_model_data$contains_bug)
# print(combined_roc)

# combined_confMatrix <- evalConfusionMatrix(combined_model_data$contains_bug, combined_result, 0.5)
# print(combined_confMatrix$precision)
# print(combined_confMatrix$recall)
# print(combined_confMatrix$f)
# print(combined_confMatrix$gscore)
print("combined evaluation")