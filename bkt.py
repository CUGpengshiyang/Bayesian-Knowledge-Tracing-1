import numpy as np
import math
from sklearn import metrics
import re
import csv

class ClassicBKT():

    def __init__(self):
        self.paramChoices = []
        self.best_params = {} #accessible by -> bestParams[skill]['p_guess']

        self.activities = {} #{activity: [skill1, skill2, skill3, ...]}
        self.activity_d = {} #(activity: index)
        self.n_activities = 0

        self.skill_d = {} #(skill: index)
        self.skills = [] #[skill1, skill2, skill3, ...]
        self.n_skills = 0

    # Generate Parameters brute force: run it once for the entire model
    def generateParams(self):
        gran = 20 # number of paramaters vals to try between 0 and 1
        spread = [x / float(gran) for x in range(1, gran)] # parameter vals to try

        for p_guess in [x for x in spread if x < 0.5]:
            for p_slip in [x for x in spread if x < 0.5]:
                for p_init in spread:
                    for p_transit in spread:
                        self.paramChoices.append(
                            {'p_guess': p_guess,
                            'p_slip': p_slip,
                            'p_init': p_init,
                            'p_transit': p_transit
                            }
                        )
        #print self.paramChoices

    def generateSkillMap(self, map):
        for row in map:
            activity = row[0]
            skill = row[1]

            if not activity in self.activities:
                self.activity_d[activity] = self.n_activities
                #extract from skill-activity mapping
                self.activities[activity] = [skill]
                self.n_activities += 1
            else:
                self.activities[activity].append(skill)

            if not skill in self.skill_d:
                    self.skill_d[skill] = self.n_skills
                    self.skills.append(skill)
                    self.n_skills += 1


    def fit(self, trainingSet, index):
        scores = [ [] for c in range(0, self.n_skills) ] #list of lists (index is equivalent to skill index)
        user_scores = [ [] for c in range(0, self.n_skills) ] #list of lists (index is equivalent to skill index)

        unique = []

        for r, row in enumerate(trainingSet):
            user = row[3]
            score = score = int(round(float(row[6])/float(row[5])))
            activity = row[2]

            if not self.activities.has_key(activity):
                continue

            if not user in unique:
                unique.append(user)
                for s in range(0, self.n_skills):
                    if len(user_scores[s]) > 0:
                        scores[s].append(list(user_scores[s])) # deep copy the scores
                user_scores = [ [] for c in range(0, self.n_skills) ]

            for skill in self.activities[activity]:
                user_scores[self.skill_d[skill]].append(score)


        for s in range(0, self.n_skills):
            if len(user_scores[s]) > 0:
                scores[s].append(list(user_scores[s])) # deep copy the scores

        
        best_params = [self.paramChoices[0] for x in range(0, self.n_skills)]
        best_scores = [float("-inf") for x in range(0, self.n_skills)]
        n_questions = [0 for x in range(0, self.n_skills)]

        # for each skill
        for c in range(0, self.n_skills):
            best_params[c] = self.paramChoices[0]

            # for each set of params
            
            for p, param in enumerate(self.paramChoices):
                total_score = 0
                n_attempts = 0
                # for each user that solved an exercise with this concept
                for u in range(0, len(scores[c])):
                    n_attempts += len(scores[c][u])
                    p_guess = param['p_guess']
                    p_guess_c = 1 - p_guess
                    p_slip = param['p_slip']
                    p_slip_c = 1 - p_slip
                    p_transit = param['p_transit']
                    # update these for each score:
                    user_score = 0
                    p_mastered = param['p_init']
                    p_mastered_c = 1 - p_mastered
                    # for each score
                    for s in scores[c][u]:
                        sm1 = s - 1
                        p_correct = (p_mastered * p_slip_c) + (p_mastered_c * p_guess)
                        #optimization: the signifance of this parameter (not the user's score)/ measure of how fit is 
                        user_score += math.log( (s * p_correct) - (sm1 * (1 - p_correct)) )
                        p_learn = ( s * p_mastered * p_slip_c / (p_mastered * p_slip_c + p_mastered_c * p_guess) ) - ( sm1 * p_mastered * p_slip / (p_mastered * p_slip + p_mastered_c * p_guess_c) )
                        p_mastered = p_learn + (1 - p_learn) * p_transit
                        p_mastered_c = 1 - p_mastered

                    total_score += user_score

                print "---"
                print str(c) + "/" + str(self.n_skills)
                print "---"
                print str(p) + "/" + str(len(self.paramChoices))
                print "---"

                if total_score > best_scores[c]:
                    best_scores[c] = total_score
                    best_params[c] = self.paramChoices[p]
                    n_questions[c] = n_attempts
                    print self.paramChoices[p]
                    print str(total_score) + "/" + str(n_attempts)
        
        # Report Results
        for c, skill in enumerate(self.skills):
            print str(best_scores[c]) + "/" + str(n_questions[c])
            self.best_params[skill] = best_params[c]


        parameter_writer = csv.writer(open('parameters' +  str(index) + '.csv', 'w'))
        parameter_writer.writerow(['skill', 'p_init', 'p_transit', 'p_guess', 'p_slip'])
        for key in self.best_params.keys():
            row = [key, str(self.best_params[key]['p_init']), str(self.best_params[key]['p_transit']), str(self.best_params[key]['p_guess']), str(self.best_params[key]['p_slip'])]
            parameter_writer.writerow(row)
            
        
        '''
        #read parameters from file
        with open("parameters0.csv", 'rU') as f:
            reader = csv.reader((line for line in f if not line.isspace()))
            for i, row in enumerate(reader):
                if i == 0:
                    continue
                skill = row[0]
                p_init = float(row[1])
                p_transit = float(row[2])
                p_guess = float(row[3])
                p_slip = float(row[4])
                self.best_params[skill] = {'p_guess': p_guess,
                            'p_slip': p_slip,
                            'p_init': p_init,
                            'p_transit': p_transit
                            }
        '''
                        
        
    def writePrediction(self, testSet, index):
        y, scores = self.predict(testSet)

        writer = csv.writer(open('results' +  str(index) + '.csv', 'w'))
        writer.writerow(['y', 'prediction'])
        for user in y.keys(): #number of users
            for i in range(0, len(y[user])):
                line = [str(y[user][i]), str(scores[user][i])]
                writer.writerow(line)
                if y[user][i] == 0:
                    print 'y: ' + str(y[user][i])
                    print 'score: ' + str(scores[user][i])
            writer.writerow([])
            
    def predict(self, testSet):
        test_scores = [ {} for c in range(0, self.n_skills) ] #index = skill number in skills [{user1: [score1, score2], user2: [score1, score2]}, {user2: [score1, score2]} ...]
        user_scores = [ [] for c in range(0, self.n_skills) ]
        
        p_correct_d = {} #{user1: {activity: [skill1_index in test score, skill2_index, skill3_index, ...], activity2: [skill1_index, skill2_index, skill3_index, ...]} , }

        unique = []

        for r, row in enumerate(testSet):
            user = row[3]
            score = score = int(round(float(row[6])/float(row[5])))
            activity = row[2]

            if not self.activities.has_key(activity):
                continue

            if not p_correct_d.has_key(user):
                p_correct_d[user] = {}

            if not p_correct_d[user].has_key(activity):
                p_correct_d[user][activity] = []

            skill_indicies = []

            for skill in self.activities[activity]:
                #user_scores[self.skill_d[skill]].append(score)
                if not test_scores[self.skill_d[skill]].has_key(user):
                    test_scores[self.skill_d[skill]][user] = []
                test_scores[self.skill_d[skill]][user].append(score)
                skill_indicies.append(len(test_scores[self.skill_d[skill]][user])-1)


            p_correct_d[user][activity].append(skill_indicies) #for this attempt

            if r == len(testSet)-1:
                for s in range(0, self.n_skills):
                    if len(user_scores[s]) > 0:
                        test_scores[s][user] = user_scores[s]

        #p_correct_d = {user1: {activity: [[skill1_index in test score, skill2_index, skill3_index, ...], [skill1_index in test score, skill2_index, skill3_index, ...]], activity2: [skill1_index, skill2_index, skill3_index, ...]} , }

        p_scores = [ {} for c in range(0, self.n_skills) ] #[{user1: [score1, score2], user2: [score1, score2]}, {user2: [score1, score2]} ...]
        #compute p_correct for each skill attempt and store it in y_scores

        # for each skill
        for c, concept in enumerate(self.skills):
            # for each user
            for user in test_scores[c].keys():
                if not self.best_params.has_key(concept):
                    self.best_params[concept] = self.paramChoices[0]

                p_guess = self.best_params[concept]['p_guess']
                p_guess_c = 1 - p_guess
                p_slip = self.best_params[concept]['p_slip']
                p_slip_c = 1 - p_slip
                p_transit = self.best_params[concept]['p_transit']
                # update these for each score:
                p_mastered = self.best_params[concept]['p_init']
                p_mastered_c = 1 - p_mastered

                p_scores[c][user] = []

                # for each score
                for s in test_scores[c][user]:
                    sm1 = s - 1
                    p_correct = (p_mastered * p_slip_c) + (p_mastered_c * p_guess)

                    p_learn = ( s * p_mastered * p_slip_c / (p_mastered * p_slip_c + p_mastered_c * p_guess) ) - ( sm1 * p_mastered * p_slip / (p_mastered * p_slip + p_mastered_c * p_guess_c) )
                    p_mastered = p_learn + (1 - p_learn) * p_transit
                    p_mastered_c = 1 - p_mastered

                    p_scores[c][user].append(p_correct)

                    #result = {'prediction': p_correct, 'actual': s, 'concept': concepts[c]}
                    #results.append(result)

        y = {} # user: [] 
        scores = {} # user: []

        #iterate through the activities in the test set again and average the p_correct scores for each activity
        #p_correct_d = {user1: {activity: [[skill1_index in test score, skill2_index, skill3_index, ...], [skill1_index in test score, skill2_index, skill3_index, ...]], activity2: [skill1_index, skill2_index, skill3_index, ...]} , }
        for user in p_correct_d:
            y[user] = []
            scores[user] = []
            for activity in p_correct_d[user]:
                for attempt in p_correct_d[user][activity]: #attempt = [skill1_index in test score, skill2_index, skill3_index, ...]
                    acc_score = 0
                    y_score = 0
                    for i in range(0, len(attempt)):
                        index = attempt[i] #index for test_score  
                        skill_i = self.skill_d[self.activities[activity][i]]

                        acc_score += p_scores[skill_i][user][index]
                        if i == 0:
                            y_score = test_scores[skill_i][user][index]                 

                    y[user].append(y_score)
                    scores[user].append(acc_score/len(self.activities[activity]))

        return y, scores



