# The following implementation is modified from https://github.com/randel/MixRF 

MixRFb <- function(Y, x, random, data, initialRandomEffects=0,
                   ErrorTolerance=0.001, MaxIterations=200,
                   ErrorTolerance0=0.001, MaxIterations0=15, verbose=FALSE) {
  
  # Condition that indicates the loop has not converged or run out of iterations
  ContinueCondition0 <- TRUE
  iterations0 = 0
  
  # Get initial values
  
  mu = rep(mean(Y),length(Y))
  eta = log(mu/(1-mu))
  y = eta + (Y-mu)/(mu*(1-mu))
  weights = mu*(1-mu)
  
  AdjustedTarget <- y - initialRandomEffects
  
  f1 = as.formula(paste0('AdjustedTarget ~ ', x))
  f0 = as.formula(paste0('resi ~ -1 + ', random))
  
  # mimic randomForest's mtry
  ncol = length(strsplit(x,split="[+]")[[1]])
  mtry = if (!is.null(y) && !is.factor(y))
    max(floor(ncol/3), 1) else floor(sqrt(ncol))
  
  oldLogLik = oldEta = -Inf
  
  # PQL
  while(ContinueCondition0) {
    
    iterations0 <- iterations0 + 1
    
    iterations = 0
    ContinueCondition <- TRUE
    
    # random forest + lmer
    while(ContinueCondition) {
      
      iterations <- iterations + 1
      
      # random Forest
      data$AdjustedTarget = AdjustedTarget
      #rf = cforest(f1, data=data, weights = weights, control = cforest_unbiased(mtry = mtry))
      rf = ranger(f1, data=data, case.weights = weights, mtry = mtry)
      
      # y - X*beta (out-of-bag prediction)
      #pred = predict(rf, OOB = TRUE)
      pred = rf$predictions
      resi = y - pred
      
      ## Estimate New Random Effects and Errors using lmer
      lmefit <- lmer(f0, data=data, weights=weights, 
                     control = lmerControl(optimizer = "nloptwrap", calc.derivs = FALSE))
      
      # check convergence
      LogLik <- as.numeric(logLik(lmefit))
      
      ContinueCondition <- (abs(LogLik-oldLogLik)>ErrorTolerance & iterations < MaxIterations)
      oldLogLik <- LogLik
      
      # Extract (the only) random effects part (Zb) to make the new adjusted target
      AllEffects <- predict(lmefit)
      
      #  y-Zb
      AdjustedTarget <- y - AllEffects
      
      # monitor the change the of logLikelihood
      if(verbose) print(c(iterations0,iterations,LogLik))
    }
    
    eta = pred + AllEffects
    mu = 1/(1+exp(-eta))
    y = eta + (Y-mu)/(mu*(1-mu))
    AdjustedTarget <- y - AllEffects
    weights = as.vector(mu*(1-mu))
    
    print(c(iter = iterations0, maxEtaChange=max(abs(eta-oldEta))))
    
    ContinueCondition0 <- (max(abs(eta-oldEta))>ErrorTolerance0 & iterations0 < MaxIterations0)
    oldEta <- eta
  }
  
  result <- list(forest=rf, MixedModel=lmefit, RandomEffects=ranef(lmefit),
                 IterationsUsed=iterations0)
  
  return(result)
}



# predict the link transformed response (eta)

predict.MixRF <- function(object, newdata, EstimateRE=TRUE){
  
  forestPrediction <- predict(object$forest,data=newdata)$predictions
  
  # If not estimate random effects, just use the forest for prediction.
  if(!EstimateRE){
    return(forestPrediction)
  }
  
  RandomEffects <- predict(object$MixedModel, newdata=newdata, allow.new.levels=TRUE)
  
  completePrediction = forestPrediction + RandomEffects
  
  return(completePrediction)
}