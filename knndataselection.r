# Set working directory to that of the current file
#setwd(dirname(rstudioapi::getActiveDocumentContext()$path))  # Only when using RStudio
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

registerDoParallel(cores = detectCores())

# Data preparation
context <- read.csv(file.path("data","context.csv", fsep = .Platform$file.sep), row.names = 1)
context$TLOC <- factor(cut(context$LOC, quantile(context$LOC), include.lowest = TRUE), labels = c("Least", "Less", "More", "Most"))
context$NFILES <- factor(cut(context$nfile, quantile(context$nfile), include.lowest = TRUE), labels = c("Least", "Less", "More", "Most"))
context$NCOMMIT <- factor(cut(context$ncommit, quantile(context$ncommit), include.lowest = TRUE), labels = c("Least", "Less", "More", "Most"))
context$NDEV <- factor(cut(context$ndev, quantile(context$ndev), include.lowest = TRUE), labels = c("Least", "Less", "More", "Most"))
context$nlanguage <- factor(cut(context$nlanguage, c(1,1.9,3), include.lowest = TRUE), labels = c("Least", "Most"))  # There are only 3 values (1,2,3) for this vector, hence we only distinguish 1 and >1
# context$git <- NULL
# context$LOC <- NULL
# context$nfile <- NULL
# context$ncommit <- NULL
# context$ndev <- NULL

project_names <- row.names(context)
print(length(project_names))
metrics <- c("ns","nd","nf","la","ld","lt","norm_entropy")

projects <- list()
for (i in 1:length(project_names)) {
  projects[[i]] <- read_csv(file.path("data",paste(project_names[i], ".csv", sep = ''), 
                                      fsep = .Platform$file.sep),
                            col_types = cols(contains_bug = col_logical(), 
                                             fix = col_logical(),commit_hash = col_skip(), commit_message = col_skip(), fileschanged = col_skip(), 
                                             fixes = col_skip(),ndev = col_skip(),))
  
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

scale_metrics <- c("ns","nf","lt","norm_entropy","relative_churn","rexp","loc","sexp","exp","age","nuc")

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

# for(i in 1:length(project_names)) {
#     print(i)
#     mean_norm_entropy = mean(projects[[i]]$norm_entropy)
#     # mean_fix = mean(projects[[i]]$fix)
#     mean_ns = mean(projects[[i]]$ns)
#     mean_nf = mean(projects[[i]]$nf)
#     mean_relative_churn = mean(projects[[i]]$relative_churn)
#     mean_lt = mean(projects[[i]]$lt)
#     std_norm_entropy = sd(projects[[i]]$norm_entropy)
#     # std_fix = sd(projects[[i]]$fix)
#     std_ns = sd(projects[[i]]$ns)
#     std_nf = sd(projects[[i]]$nf)
#     std_relative_churn = sd(projects[[i]]$relative_churn)
#     std_lt = sd(projects[[i]]$lt)
#     print(mean_norm_entropy)
#     # print(mean_fix)
#     print(mean_ns)
#     print(mean_nf)
#     print(mean_relative_churn)
#     print(mean_lt)
#     print(std_norm_entropy)
#     # print(std_fix)
#     print(std_ns)
#     print(std_nf)
#     print(std_relative_churn)
#     print(std_lt)
# }

# for(i in 1:length(project_names)) {
#   print("Project")
#   print(i)
#     mean_norm_entropy_i = mean(projects[[i]]$norm_entropy)
#     # mean_fix = mean(projects[[i]]$fix)
#     mean_ns_i = mean(projects[[i]]$ns)
#     mean_nf_i = mean(projects[[i]]$nf)
#     mean_relative_churn_i = mean(projects[[i]]$relative_churn)
#     mean_lt_i = mean(projects[[i]]$lt)
#     std_norm_entropy_i = sd(projects[[i]]$norm_entropy)
#     # std_fix = sd(projects[[i]]$fix)
#     std_ns_i = sd(projects[[i]]$ns)
#     std_nf_i = sd(projects[[i]]$nf)
#     std_relative_churn_i = sd(projects[[i]]$relative_churn)
#     std_lt_i = sd(projects[[i]]$lt)
#   for(j in 1:length(project_names)) {
#     mean_norm_entropy_j = mean(projects[[j]]$norm_entropy)
#     # mean_fix = mean(projects[[i]]$fix)
#     mean_ns_j = mean(projects[[j]]$ns)
#     mean_nf_j = mean(projects[[j]]$nf)
#     mean_relative_churn_j = mean(projects[[j]]$relative_churn)
#     mean_lt_j = mean(projects[[j]]$lt)
#     std_norm_entropy_j = sd(projects[[j]]$norm_entropy)
#     # std_fix = sd(projects[[i]]$fix)
#     std_ns_j = sd(projects[[j]]$ns)
#     std_nf_j = sd(projects[[j]]$nf)
#     std_relative_churn_j = sd(projects[[j]]$relative_churn)
#     std_lt_j = sd(projects[[j]]$lt)
#     print(j)
#     euc_distance = sqrt((mean_norm_entropy_i-mean_norm_entropy_j)^2+(mean_ns_i-mean_ns_j)^2+(mean_nf_i-mean_nf_j)^2+(mean_lt_i-mean_lt_j)^2+(mean_relative_churn_i-mean_relative_churn_j)^2)
#     print(euc_distance)
#   }
#   print("")
#   print("")
# }

training_set_euc <- subset(all_projects, all_projects$project==project_names[1]|all_projects$project==project_names[17])
testing_set_euc <- subset(all_projects, all_projects$project == project_names[2])
print("got euc sets")
tmp_project_aware_lr_model_euc <- glmer(contains_bug ~ (norm_entropy | project)+ns +nf+rexp+loc+sexp+exp+age+nuc, 
                                     data = training_set_euc,nAGQ=0,family = "binomial",)
#tmp_project_aware_lr_model_euc <- glm(contains_bug ~ fix + ns + nf + norm_entropy + relative_churn + lt, data = training_set_euc, family="binomial")
# tmp_project_aware_lr_pred_euc <- predict(tmp_project_aware_lr_model_euc, testing_set_euc, allow.new.levels = TRUE, type = "response")
# # print(evalPredict(testing_set_euc$contains_bug, tmp_project_aware_lr_pred_euc, testing_set_euc$loc))
  tmp_project_aware_lr_pred_corrected_euc <- predict(tmp_project_aware_lr_model_euc, testing_set_euc, allow.new.levels = TRUE, type = "link")
  # print("coef")
  # print(coef(tmp_project_aware_lr_model_euc))
  # print(coef(tmp_project_aware_lr_model_euc)[[1]][,1])
#                                                                                      intercept estimate                                   individual intercept
  tmp_project_aware_lr_pred_corrected_euc <- tmp_project_aware_lr_pred_corrected_euc - coef(summary(tmp_project_aware_lr_model_euc))[1,1] + median(coef(tmp_project_aware_lr_model_euc)[[1]][,2]) + testing_set_euc$ns * median(coef(tmp_project_aware_lr_model_euc)[[1]][,1])
  tmp_project_aware_lr_pred_corrected_euc <- inv.logit(tmp_project_aware_lr_pred_corrected_euc)
  print(evalPredict(testing_set_euc$contains_bug, tmp_project_aware_lr_pred_corrected_euc, testing_set_euc$loc))
# print(coef(tmp_project_aware_lr_model_euc)[[1]])
print(r.squaredGLMM(tmp_project_aware_lr_model_euc))
print("Done data selection")