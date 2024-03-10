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

# ===================== RQ1 ========================
# local.jit <- c()
# local.intercept <- c()
# local.entropy <- c()

# for (i in 1:length(project_names)) {
#   # Model training
#   trained_model <- glm(contains_bug ~ fix + ns + nf + norm_entropy + relative_churn + lt, 
#                        data = projects[[i]], family = "binomial")
#   local.jit[[length(local.jit)+1]] <- trained_model
  
#   # Model summary
#   print(project_names[i])
#   local.intercept[[length(local.intercept)+1]] <- trained_model$coefficients[1]
#   local.entropy[[length(local.entropy)+1]] <- trained_model$coefficients[5]
#   print(trained_model$coefficients)
  
#   # Chisq
#   print(Anova(trained_model, Type = 2))
  
#   # Goodness-of-fit
#   print(r.squaredGLMM(trained_model))
# }

# # ===================== RQ2 ========================
# # Model training
# global.jit <- glm(contains_bug ~ fix + ns + nf + norm_entropy + relative_churn + lt, data = all_projects, family = "binomial")

# # Median Absolute Error for entropy and intercepts
# global.jit.entropy <- coef(global.jit)[5]
# global.jit.intercept <- coef(global.jit)[1]
# diff<- list()
# #for(i in 1:length(local.entropy)) {
# #  diff<-append(diff, abs(local.entropy[i]-global.jit.entropy))
# #}
# #median(abs(diff)) # MAE for Entropy
# #median(abs(local.intercept - global.jit.intercept)) # MAE for intercepts

# # Chisq
# Anova(global.jit, Type = 2)

# # Goodness-of-fit
# print(r.squaredGLMM(global.jit))

# # ===================== RQ3 ========================
# # Performance of Global JIT Model (Logistic Regression)
# print("RQ3")
# get_lr_perf <- function(i) {
#   training_set <- subset(all_projects, all_projects$project != project_names[i])
#   testing_set <- subset(all_projects, all_projects$project == project_names[i])
  
#   tmp_lr_model <- glm(contains_bug ~ fix + ns + nf + norm_entropy + relative_churn + lt, data = training_set, family="binomial")
#   tmp_lr_pred <- predict(tmp_lr_model, newdata = testing_set, type = "response")
#   print(i)
#   print(evalPredict(testing_set$contains_bug, tmp_lr_pred, testing_set$loc))
#   return(evalPredict(testing_set$contains_bug, tmp_lr_pred, testing_set$loc))
# }
# print("get_lr_perf")
# lr_perf <- llply(seq(1, length(project_names), 1), get_lr_perf)
# lr_perf <- rbindlist(lr_perf, fill=TRUE)
# print("global")
# # Performance of Project Aware JIT Model (Logistic Regression)
# get_project_aware_lr_perf <- function(i, correct = TRUE) {
#   training_set <- subset(all_projects, all_projects$project != project_names[i])
#   testing_set <- subset(all_projects, all_projects$project == project_names[i])
#   print("got sets")
#   tmp_project_aware_lr_model <- glmer(contains_bug ~ (norm_entropy | project) + fix + ns + nf + relative_churn + lt, 
#                                       data = training_set, nAGQ=0, family = "binomial")
#   print("trained model")
#   tmp_project_aware_lr_pred <- predict(tmp_project_aware_lr_model, testing_set, allow.new.levels = TRUE, type = "response")
#   tmp_project_aware_lr_pred_corrected <- predict(tmp_project_aware_lr_model, testing_set, allow.new.levels = TRUE, type = "link")
#   tmp_project_aware_lr_pred_corrected <- tmp_project_aware_lr_pred_corrected - coef(summary(tmp_project_aware_lr_model))[1,1] + median(coef(tmp_project_aware_lr_model)[[1]][,2]) + testing_set$norm_entropy * median(coef(tmp_project_aware_lr_model)[[1]][,1])
#   tmp_project_aware_lr_pred_corrected <- inv.logit(tmp_project_aware_lr_pred_corrected)
  
#   if(correct) {
#     print(evalPredict(testing_set$contains_bug, tmp_project_aware_lr_pred, testing_set$loc))
#     return(evalPredict(testing_set$contains_bug, tmp_project_aware_lr_pred_corrected, testing_set$loc))
#   } else {
#     return(evalPredict(testing_set$contains_bug, tmp_project_aware_lr_pred, testing_set$loc))
#   }
# }
# print("get_lr_perf")
# project_aware_lr_perf_correct <- llply(seq(1, length(project_names), 1), get_project_aware_lr_perf)
# project_aware_lr_perf_correct <- rbindlist(project_aware_lr_perf_correct, fill=TRUE)
# print("project aware")
# # Performance of Context Aware JIT Model (Logistic Regression)
# get_context_aware_lr_perf <- function(i, correct = TRUE) {
#   training_set <- subset(all_projects, all_projects$project != project_names[i])
#   testing_set <- subset(all_projects, all_projects$project == project_names[i])
#   print("got sets")
#   tmp_context_aware_lr_model <- glmer(contains_bug ~ (0 + norm_entropy | project) + fix + ns + nf + relative_churn + lt + (1 | language) + (1 | nlanguage) + (1 | TLOC)+ (1 | NFILES) + (1 | NCOMMIT) + (1 | NDEV) + (1 | audience) + (1 | ui) + (1 | database), 
#                                       data = training_set,  family = "binomial")
#   print("trained model")
#   tmp_context_aware_lr_pred <- predict(tmp_context_aware_lr_model, testing_set, allow.new.levels = TRUE, type = "response")
#   tmp_context_aware_lr_pred_corrected <- predict(tmp_context_aware_lr_model, testing_set, allow.new.levels = TRUE, type = "link")
#   tmp_context_aware_lr_pred_corrected <- tmp_context_aware_lr_pred_corrected - coef(summary(tmp_context_aware_lr_model))[1,1] + median(coef(tmp_context_aware_lr_model)$project[,2]) + testing_set$norm_entropy * median(coef(tmp_context_aware_lr_model)$project[,1])
#   if (testing_set$language[1] == 'PHP' || testing_set$language[1] == 'C' || testing_set$language[1] == 'C++' || testing_set$language[1] == 'Perl') {
#     tmp_context_aware_lr_pred_corrected <- tmp_context_aware_lr_pred_corrected + median(coef(tmp_context_aware_lr_model)$language[,2])
#   }
#   tmp_context_aware_lr_pred_corrected <- inv.logit(tmp_context_aware_lr_pred_corrected)
  
#   if(correct) {
#     print(evalPredict(testing_set$contains_bug, tmp_context_aware_lr_pred, testing_set$loc))
#     return(evalPredict(testing_set$contains_bug, tmp_context_aware_lr_pred_corrected, testing_set$loc))
#   } else {
#     return(evalPredict(testing_set$contains_bug, tmp_context_aware_lr_pred, testing_set$loc))
#   }
# }
# print("get_lr_perf")
# context_aware_lr_perf_correct <- llply(seq(1, length(project_names), 1), get_context_aware_lr_perf)
# context_aware_lr_perf_correct <- rbindlist(context_aware_lr_perf_correct, fill=TRUE)
# print("context aware")

# # ===================== RQ4 ========================
# # Model training
print("RQ4")
project.aware <- glmer(contains_bug ~ (norm_entropy | project) +(1|cluster)+ fix + ns + nf + relative_churn + lt, 
                       data = all_projects, nAGQ=0, family = "binomial")
print("trained model")
# # Median Absolute Error for entropy and intercepts
# project.aware.entropy <- coef(project.aware)[[1]][,1]
# project.aware.intercept <- coef(project.aware)[[1]][,2]
# #median(abs(local.entropy - project.aware.entropy)) # MAE for Entropy
# #median(abs(local.intercept - project.aware.intercept)) # MAE for intercepts

# Goodness-of-fit
print(r.squaredGLMM(project.aware))

# # Chisq
# Anova(project.aware, Type = 2)

# # Likelihood Ratio Test
# project.aware.noslope <- glmer(contains_bug ~ (1 | project) + fix + ns + nf + relative_churn + lt, 
#                                data = all_projects, family = "binomial")
# mixed.effect.null <- glm(contains_bug ~ fix + ns + nf + relative_churn + lt, 
#                          data = all_projects, family = "binomial")
# anova(project.aware, project.aware.noslope, mixed.effect.null)

# # ===================== RQ5 ========================
# Model training
# print("RQ5")
# context.aware <- glmer(contains_bug ~ (0 + norm_entropy | project) + fix + ns + nf + relative_churn + lt + (1 | language) + (1 | nlanguage) + (1 | TLOC)+ (1 | NFILES) + (1 | NCOMMIT) + (1 | NDEV) + (1 | audience) + (1 | ui) + (1 | database)+(1|cluster), 
#                        data = all_projects, family = "binomial", 
#                        control=glmerControl(optimizer="bobyqa",optCtrl=list(maxfun=2e5)),nAGQ=0)

# # Median Absolute Error for entropy and intercepts
# context.aware.entropy <- coef(context.aware)[[1]][,1]
# context.aware.intercept <- c(-0.412809322, -0.403441522, -0.368979092, -0.019905339, -0.548783572, 
#                               -0.669703372, 0.387229798, -0.295431722, 0.301578719, -0.049754352, 
#                               0.136789488, 0.221394998, -0.145870962, -0.161213922, 0.412493888, 
#                               -0.431692062, -0.528704192, 0.132923678, -0.112635552, 0.714081312) # sum of intercepts of contextual factors for each project
# #median(abs(local.entropy - context.aware.entropy)) # MAE for Entropy
# #median(abs(local.intercept - context.aware.intercept)) # MAE for intercepts

# # Goodness-of-fit
# print(r.squaredGLMM(context.aware))

# Chisq
Anova(project.aware, Type = 2)

# # Likelihood Ratio Test (using multicores to train models parallelly)
# get_randomeff_models <- function(id) {
#   print(i)
#   random.eff.formula <- c()
#   random.eff.formula[[1]] <- contains_bug ~ (0 + norm_entropy | project) + fix + ns + nf + relative_churn + lt + (1 | language) + (1 | nlanguage) + (1 | TLOC) + (1 | NFILES) + (1 | NCOMMIT) + (1 | NDEV) + (1 | audience) + (1 | ui)
#   random.eff.formula[[2]] <- contains_bug ~ (0 + norm_entropy | project) + fix + ns + nf + relative_churn + lt + (1 | language) + (1 | nlanguage) + (1 | TLOC) + (1 | NFILES) + (1 | NCOMMIT) + (1 | NDEV) + (1 | audience)
#   random.eff.formula[[3]] <- contains_bug ~ (0 + norm_entropy | project) + fix + ns + nf + relative_churn + lt + (1 | language) + (1 | nlanguage) + (1 | TLOC) + (1 | NFILES) + (1 | NCOMMIT) + (1 | NDEV)
#   random.eff.formula[[4]] <- contains_bug ~ (0 + norm_entropy | project) + fix + ns + nf + relative_churn + lt + (1 | language) + (1 | nlanguage) + (1 | TLOC) + (1 | NFILES) + (1 | NCOMMIT)
#   random.eff.formula[[5]] <- contains_bug ~ (0 + norm_entropy | project) + fix + ns + nf + relative_churn + lt + (1 | language) + (1 | nlanguage) + (1 | TLOC) + (1 | NFILES)
#   random.eff.formula[[6]] <- contains_bug ~ (0 + norm_entropy | project) + fix + ns + nf + relative_churn + lt + (1 | language) + (1 | nlanguage) + (1 | TLOC)
#   random.eff.formula[[7]] <- contains_bug ~ (0 + norm_entropy | project) + fix + ns + nf + relative_churn + lt + (1 | language) + (1 | nlanguage)
#   random.eff.formula[[8]] <- contains_bug ~ (0 + norm_entropy | project) + fix + ns + nf + relative_churn + lt + (1 | language)
#   random.eff.formula[[9]] <- contains_bug ~ (0 + norm_entropy | project) + fix + ns + nf + relative_churn + lt
  
#   if (id == 1 || id == 5 || id == 7) {
#     tmp_mixed_model <- glmer(random.eff.formula[[id]], 
#                              data = all_projects, family = "binomial",control=glmerControl(optimizer="bobyqa",optCtrl=list(maxfun=2e5)))
#   }
#   else {
#     tmp_mixed_model <- glmer(random.eff.formula[[id]], 
#                              data = all_projects, family = "binomial", 
#                              control=glmerControl(optimizer="bobyqa",optCtrl=list(maxfun=2e5)))
#   }
#   print("done")
#   return(tmp_mixed_model)
# }
# print("Last part of rq5")
# registerDoParallel(cores=detectCores())
# randomeff_models <- llply(seq(1, 9, 1), get_randomeff_models)
# anova(context.aware, randomeff_models[[1]], randomeff_models[[2]], randomeff_models[[3]], randomeff_models[[4]], randomeff_models[[5]], 
#       randomeff_models[[6]], randomeff_models[[7]], randomeff_models[[8]], randomeff_models[[9]], mixed.effect.null)

# ===================== RQ6 ========================
# Performance of Global JIT Model (Random Forest)
print("RQ6")
rf_perf <- as.data.frame(c())
#for (i in 1:length(project_names)) {
#  training_set <- subset(all_projects, all_projects$project != project_names[i])
#  testing_set <- subset(all_projects, all_projects$project == project_names[i])
#  
#  tmp_rf_model <- ranger(as.factor(contains_bug) ~ fix + ns + nf + norm_entropy + relative_churn + lt, data = training_set, probability = TRUE)
#  tmp_rf_pred <- predict(tmp_rf_model, testing_set)$predictions[,2]
  
#  rf_perf <- rbind(rf_perf, evalPredict(testing_set$contains_bug, tmp_rf_pred, testing_set$loc))
#  print("project number")
#  print(i)
#}
print("global JIT model")
# Performance of Project Aware JIT Model (Random Forest)
project_aware_rf_perf <- as.data.frame(c())
#print(length(project_names))
#for (i in 1:length(project_names)) {
#  training_set <- subset(all_projects, all_projects$project != project_names[i])
#  testing_set <- subset(all_projects, all_projects$project == project_names[i])
#
#  tmp_project_aware_rf_model <- MixRFb(training_set$contains_bug, x = 'fix + ns + nf + relative_churn + lt', random = '(norm_entropy | project)', data = training_set, verbose=T, ErrorTolerance = 1, ErrorTolerance0 = 0.3, MaxIterations=35)
#  tmp_project_aware_rf_pred <- predict.MixRF(tmp_project_aware_rf_model, testing_set, EstimateRE = TRUE)
#  tmp_project_aware_rf_pred_corrected <- tmp_project_aware_rf_pred + median(coef(tmp_project_aware_rf_model$MixedModel)$project[,'(Intercept)']) + median(coef(tmp_project_aware_rf_model$MixedModel)$project[,'norm_entropy']) * testing_set$norm_entropy
#  tmp_project_aware_rf_pred_corrected <- inv.logit(tmp_project_aware_rf_pred_corrected)
#  
# project_aware_rf_perf <- rbind(project_aware_rf_perf, evalPredict(testing_set$contains_bug, tmp_project_aware_rf_pred_corrected, testing_set$loc))
#  print("project number")
#  print(i)
#}
# print("project aware")
# # Performance of Context Aware JIT Model (Random Forest)
# for(i in 1:length(project_names)) {
#   print(i)
#   training_set <- subset(all_projects, all_projects$project != project_names[i])
#   testing_set <- subset(all_projects, all_projects$project == project_names[i])
  
#   tmp_context_aware_rf_model <- MixRFb(training_set$contains_bug, x = 'fix + ns + nf + relative_churn + lt', random = '(0 + norm_entropy | project) + (1 | language) + (1 | nlanguage) + (1 | TLOC) + (1 | NFILES) + (1 | NCOMMIT) + (1 | NDEV) + (1 | audience) + (1 | ui) + (1 | database)', 
#                                        data = training_set, verbose=T, ErrorTolerance = 1, ErrorTolerance0 = 0.3, 
#                                        MaxIterations = 15)

#   tmp_context_aware_rf_pred <- predict.MixRF(tmp_context_aware_rf_model, testing_set, EstimateRE = TRUE)
#   tmp_context_aware_rf_pred_corrected <- tmp_context_aware_rf_pred + median(coef(tmp_context_aware_rf_model$MixedModel)$project[,'norm_entropy']) * testing_set$norm_entropy
#   if (testing_set$language[1] == 'PHP' || testing_set$language[1] == 'C' || testing_set$language[1] == 'C++' || testing_set$language[1] == 'Perl') {
#     tmp_context_aware_rf_pred_corrected <- tmp_context_aware_rf_pred_corrected + median(coef(tmp_context_aware_rf_model$MixedModel)$language[,2])
#   }
#   tmp_context_aware_rf_pred_corrected <- inv.logit(tmp_context_aware_rf_pred_corrected)
#   print(evalPredict(testing_set$contains_bug, tmp_context_aware_rf_pred_corrected, testing_set$loc))
#   VarF <- var(tmp_context_aware_rf_model$forest$predictions)
# VarRand <- var(predict(tmp_context_aware_rf_model$MixedModel, newdata=training_set))
# pred_err <- training_set$contains_bug - inv.logit(predict.MixRF(tmp_context_aware_rf_model, training_set, EstimateRE = TRUE))
# VarDisp <- var(logit(subset(abs(pred_err), abs(pred_err) > 0 & abs(pred_err) < 1)))
# Rc_context_aware_rf <- (VarF+VarRand)/(VarF+VarRand+VarDisp)
# print(Rc_context_aware_rf)
#   if(correct) {
#     return(evalPredict(testing_set$contains_bug, tmp_context_aware_rf_pred_corrected, testing_set$loc))
#   } else {
#     return(evalPredict(testing_set$contains_bug, tmp_context_aware_rf_pred, testing_set$loc))
#   }
# }
# print("done training context aware")
# context_aware_rf_perf_correct <- llply(seq(1, length(project_names), 1), get_context_aware_rf_perf, .parallel=FALSE)
# context_aware_rf_perf_correct <- rbindlist(context_aware_rf_perf_correct, fill=TRUE)

# # Goodness-of-fit of Global JIT Model (Random Forest)
# global_rf_model <- ranger(as.factor(contains_bug) ~ fix + ns + nf + norm_entropy + relative_churn + lt, data = all_projects, probability = TRUE)
# VarF <- var(global_rf_model$predictions[,2])
# VarDisp <- var(all_projects$contains_bug - global_rf_model$predictions[,2])
# Rc_global_rf <- (VarF)/(VarF+VarDisp)

# # Goodness-of-fit of Project Aware JIT Model (Random Forest)
# project_aware_rf_model <- MixRFb(all_projects$contains_bug, x = 'fix + ns + nf + relative_churn + lt', random = '(norm_entropy | project)', data = all_projects, verbose=T, ErrorTolerance = 1, ErrorTolerance0 = 0.3, MaxIterations=20)
# VarF <- var(project_aware_rf_model$forest$predictions)
# VarRand <- var(predict(project_aware_rf_model$MixedModel, newdata=all_projects))
# pred_err <- all_projects$contains_bug - inv.logit(predict.MixRF(project_aware_rf_model, all_projects, EstimateRE = TRUE))
# VarDisp <- var(logit(subset(abs(pred_err), abs(pred_err) > 0 & abs(pred_err) < 1)))
# Rc_project_aware_rf <- (VarF+VarRand)/(VarF+VarRand+VarDisp)

# Goodness-of-fit of Context Aware JIT Model (Random Forest)
# context_aware_rf_model <- MixRFb(all_projects$contains_bug, x = 'fix + ns + nf + relative_churn + lt', random = '(0 + norm_entropy | project) + (1 | language) + (1 | nlanguage) + (1 | TLOC) + (1 | NFILES) + (1 | NCOMMIT) + (1 | NDEV) + (1 | audience) + (1 | ui) + (1 | database)', 
#                                      data = all_projects, verbose=T, ErrorTolerance = 1, ErrorTolerance0 = 0.3, 
#                                      MaxIterations = 10, MaxIterations0 = 10)
# VarF <- var(context_aware_rf_model$forest$predictions)
# VarRand <- var(predict(context_aware_rf_model$MixedModel, newdata=all_projects))
# pred_err <- all_projects$contains_bug - inv.logit(predict.MixRF(context_aware_rf_model, all_projects, EstimateRE = TRUE))
# VarDisp <- var(logit(subset(abs(pred_err), abs(pred_err) > 0 & abs(pred_err) < 1)))
# Rc_context_aware_rf <- (VarF+VarRand)/(VarF+VarRand+VarDisp)
