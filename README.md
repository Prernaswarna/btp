# Replication package for "The Impact of Data Merging on the Interpretation of Cross-Project Just-In-Time Defect Models"

### Link to pre-print
https://www.researchgate.net/publication/350853761_The_Impact_of_Data_Merging_on_the_Interpretation_of_Cross-Project_Just-In-Time_Defect_Models

### Note on our implementation of mixed-effect modelling

When testing a project-aware or a context-aware model on data from an unseen project during training (i.e., a project whose data is not used to train the mixed-effect model), it is possible that the testing data contains values for random effects whose coefficients were not learnt during model construction. For example, the project that is used for testing may be written in the PHP language, while non of the projects in training data are written in the PHP language. In this case, the constructed context-aware model will not be able to learn the random intercept of the PHP language during training. A commonly used strategy is to ignore the unseen random effect during prediction (i.e., assume a coefficient of 0). However such strategy will significantly impact the performance of the model. 

In our study, instead of assuming a coefficient of 0, we used the median of coefficients of other seen random effects during training as the unseen random effect's coefficient. For example, we use the median random intercept of all other seen project languages during training (Java, JavaScript, Perl, Python, C, and C++) as the random intercept of the PHP language. We observed that our strategy significantly improves the performance of predicting unseen project during training compared to the commonly used strategy of assuming a coefficient of 0.
