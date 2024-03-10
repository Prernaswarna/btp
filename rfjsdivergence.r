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
                                             ndev = col_skip()))
  
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

scale_metrics <- c("ns","nf","lt","norm_entropy","relative_churn","rexp","sexp","loc")

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

print("RQ6")
project_aware_rf_perf <- as.data.frame(c())

# #spearman closest projects
# p1 <- c(3, 4, 1, 7, 10, 16, 4, 12, 13, 5, 15, 8, 14, 13, 11, 6, 13, 19, 20, 19)
# p2 <- c(16, 7, 16, 2, 11, 1, 2, 10, 14, 15, 19, 4, 9, 15, 19, 3, 1, 11, 18, 15)
# p3 <- c(6, 12, 6, 12, 15, 3, 15, 4, 17, 11, 5, 7, 17, 9, 20, 1, 3, 15, 11, 11 )
# p4 <- c(17, 20, 17, 8, 18, 17, 12, 7, 12, 18, 18, 9, 3, 17, 10, 17, 14, 20, 15, 18)

#js divergence closest projects
p1 <- c(3, 17, 1, 17, 1, 1, 4,17,15, 17,17,15,17,17,8,3,8,17,17,3)
p2 <- c(17, 1, 16, 1, 17, 3, 6,15,17,8,8,17,6,12,12,1,11,3,6,1)
p3 <- c(6, 4, 20, 3, 15, 17, 3,10,12,12,12,8,3,4,9,6,10,12,3,4)
p4 <- c(5, 3, 6, 8, 4, 4, 12, 12,8,15,15,10,15,10,10,4,1,13,15,6)


for(i in 8:length(project_names)) {
  print(i)
  proj1 = p1[i]
  proj2 = p2[i]
  proj3 = p3[i]
  proj4 = p4[i]
  training_set <- subset(all_projects, all_projects$project==project_names[proj1]|all_projects$project==project_names[proj2]|all_projects$project==project_names[proj3]|all_projects$project==project_names[proj4])
  testing_set <- subset(all_projects, all_projects$project == project_names[i])

# # without feature selection 
#   tmp_project_aware_rf_model <- MixRFb(training_set$contains_bug, x = 'fix + ns + nf + relative_churn + lt', random = '(norm_entropy | project)', data = training_set, verbose=T, ErrorTolerance = 1, ErrorTolerance0 = 0.3, MaxIterations=20)


# feature selection 6
  tmp_project_aware_rf_model <- MixRFb(training_set$contains_bug, x = 'ns+nf+rexp+loc+sexp', random = '(norm_entropy | project)', data = training_set, verbose=T, ErrorTolerance = 1, ErrorTolerance0 = 0.3, MaxIterations=20)
  tmp_project_aware_rf_pred <- predict.MixRF(tmp_project_aware_rf_model, testing_set, EstimateRE = TRUE)
  tmp_project_aware_rf_pred_corrected <- tmp_project_aware_rf_pred + median(coef(tmp_project_aware_rf_model$MixedModel)$project[,'(Intercept)']) + median(coef(tmp_project_aware_rf_model$MixedModel)$project[,'norm_entropy']) * testing_set$norm_entropy
  tmp_project_aware_rf_pred_corrected <- inv.logit(tmp_project_aware_rf_pred_corrected)
  print("model built")
  VarF <- var(tmp_project_aware_rf_model$forest$predictions)
  print("VarF")
  VarRand <- var(predict(tmp_project_aware_rf_model$MixedModel, newdata=training_set))
  print("VarRand")
  pred_err <- training_set$contains_bug - inv.logit(predict.MixRF(tmp_project_aware_rf_model, training_set, EstimateRE = TRUE))
  print("pred_err")
  VarDisp <- var(logit(subset(abs(pred_err), abs(pred_err) > 0 & abs(pred_err) < 1)))
  print("VarDisp")
  Rc_project_aware_rf <- (VarF+VarRand)/(VarF+VarRand+VarDisp)

  print(i)
  print(proj1)
  print(proj2)
  print(proj3)
  print(proj4)

  print(Rc_project_aware_rf)
  print("R2")
  print(evalPredict(testing_set$contains_bug, tmp_project_aware_rf_pred_corrected, testing_set$loc))
  print("evalPredict")
    # project_aware_rf_perf <- rbind(project_aware_rf_perf, evalPredict(testing_set$contains_bug, tmp_project_aware_rf_pred_corrected, testing_set$loc))
    #   print(project_aware_rf_perf)
    #   print(i)
}