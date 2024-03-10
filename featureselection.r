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

rstan_options(auto_write=TRUE)
registerDoParallel(cores = detectCores())

# Data preparation
context <- read.csv(file.path("data","context.csv", fsep = .Platform$file.sep), row.names = 1)
# context$TLOC <- factor(cut(context$LOC, quantile(context$LOC), include.lowest = TRUE), labels = c("Least", "Less", "More", "Most"))
# context$NFILES <- factor(cut(context$nfile, quantile(context$nfile), include.lowest = TRUE), labels = c("Least", "Less", "More", "Most"))
# context$NCOMMIT <- factor(cut(context$ncommit, quantile(context$ncommit), include.lowest = TRUE), labels = c("Least", "Less", "More", "Most"))
# context$NDEV <- factor(cut(context$ndev, quantile(context$ndev), include.lowest = TRUE), labels = c("Least", "Less", "More", "Most"))
# context$nlanguage <- factor(cut(context$nlanguage, c(1,1.9,3), include.lowest = TRUE), labels = c("Least", "Most"))  # There are only 3 values (1,2,3) for this vector, hence we only distinguish 1 and >1
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
                                             fix = col_logical(), author_date = col_skip(), 
                                             author_date_unix_timestamp = col_skip(), 
                                             author_email = col_skip(), author_name = col_skip(), 
                                             classification = col_skip(), commit_hash = col_skip(), 
                                             commit_message = col_skip(), fileschanged = col_skip(), 
                                             fixes = col_skip(), glm_probability = col_skip(), 
                                             linked = col_skip(), repository_id = col_skip(),
                                              nuc = col_skip(),
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

head(all_projects)

# all_projects.bglm <- all_projects[, c("fix", "ndev", "age", "ns", "nf", "norm_entropy", "relative_churn", "lt", "nlanguage", "loc", "nfile", "ncommit", "ndev", "contains_bug")]
# rename(all_projects.bglm, y=contains_bug)

# best.logit <- bestglm(all_projects.bglm,
#                 IC = "AIC",                
#                 family=binomial,
#                 method = "exhaustive")

# summary(best.logit$BestModel)
# best.logit$Subsets


# global.jit <- glm(contains_bug ~ fix + ns + nf + norm_entropy + relative_churn + lt, data = all_projects, family = "binomial")
# # Chisq
# Anova(global.jit, Type = 2)

# # Goodness-of-fit
# print(r.squaredGLMM(global.jit))


# featureglobal.jit <- glm(contains_bug ~ fix + ns + nf + norm_entropy + lt, data = all_projects, family = "binomial")
# # Chisq
# Anova(featureglobal.jit, Type = 2)

# # Goodness-of-fit
# print(r.squaredGLMM(featureglobal.jit))

# context.aware <- glmer(contains_bug ~ (0 + norm_entropy | project) + fix + ns + nf + relative_churn + lt + (1 | language) + (1 | nlanguage) + (1 | TLOC)+ (1 | NFILES) + (1 | NCOMMIT) + (1 | NDEV) + (1 | audience) + (1 | ui) + (1 | database), 
#                        data = all_projects, family = "binomial", 
#                        control=glmerControl(optimizer="bobyqa",optCtrl=list(maxfun=2e5)))

# # Goodness-of-fit
# print(r.squaredGLMM(context.aware))

# # Chisq
# Anova(context.aware, Type = 2)


# contextfeature.aware <- glmer(contains_bug ~ (0 + norm_entropy | project) + fix + ns + nf + relative_churn + lt + (1 | language) + (1 | TLOC)+ (1 | NFILES) + (1 | NCOMMIT) + (1 | NDEV), 
#                        data = all_projects, family = "binomial", 
#                        control=glmerControl(optimizer="bobyqa",optCtrl=list(maxfun=2e5)))

# # Goodness-of-fit
# print(r.squaredGLMM(context.aware))

# # Chisq
# Anova(context.aware, Type = 2)

# context.aware <- glm(contains_bug ~ norm_entropy + fix + ns + nf  + lt + language + nlanguage + TLOC+ NFILES + NCOMMIT + NDEV + audience + ui + database, 
#                         data = all_projects, family = "binomial")

# # # Goodness-of-fit
# # print(r.squaredGLMM(context.aware))

# summary(context.aware)

# Set priors
pr = prior(normal(0, 1), class = 'b')
            
# Fit model
context_bayes <- brm(result ~ (0 + norm_entropy | projectid) + ns + nf + relative_churn + lt  + (1 | nlanguage) + (1 | loc)+ (1 | nfile) + (1 | ncommit) + (1 | ndev),
             data = all_projects,
             prior = pr,
             family = binomial(),
             warmup = 1500, # burn-in
             iter = 3000, # number of iterations
             chains = 2,  # number of MCMC chains
             control = list(adapt_delta = 0.8)
             ) # advanced MC settings

# Example summary                     
summary(context_bayes)