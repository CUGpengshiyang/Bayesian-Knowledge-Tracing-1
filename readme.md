# Comparing Different Approaches to Modeling Student Knowledge

This tool is developed to fit and train specific type of Hidden Markov Models used in education called Bayesian Knowledge Tracing (BKT) model. Modeling a student’s knowledge provides a significant predictive measure such as indicating to which extent a student would be able to perform successfully on a certain task and suggesting which task the student should be given next. By comparing several versions of the BKT model, the author attempts to test whether contextualizing a subset of parameters would provide a better representation of knowledge acquisition of students. The utility is written in Python. The project started in September 2016 and is on going as of April 2017.

# Running the Model

Running runModel.py prompts the user to enter the route to the student submission data file, the activity-skill map file, which model to run, and the number of cross validation folds. When the all the data files have been read in, the chosen model is fit (Brute Force for original BKT model) accordingly with a training set and predicts student performance on the test set. auc.py calculates the auc score of each cross validation set and the average auc score.
# References

1. Corbett, A. T., and Anderson, J. R.: Knowledge tracing: Modeling the acquisition of procedural knowledge. User Modeling and User-Adapted Interaction, 4(4), 253-278. (1995)

2. Baker, R.S.J.d., Corbett, A.T., Aleven, V.: More Accurate Student Modeling Through Contextual Estimation of Slip and Guess Probabilities in Bayesian Knowledge Tracing. Proceedings of the 9th International Conference on Intelligent Tutoring Systems, 406-415. (2008)
