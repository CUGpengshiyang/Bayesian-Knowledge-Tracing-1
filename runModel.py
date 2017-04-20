# Imports
import csv
import sys
import argparse
import math
import numpy as np
from bkt import ClassicBKT


def read_data(fileName):
    unique = []
    filteredData = [] #[[row1, row2], [row1, row2]]
    currUserData = []

    with open(fileName, 'rU') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == 0:
                continue
            s_id = row[3]
            if not s_id in unique:
                unique.append(s_id)
                if len(currUserData) > 0:
                    filteredData.append(currUserData)
                    currUserData = []
            currUserData.append(row)

        if len(currUserData)> 0:
            filteredData.append(currUserData)
    print "Number of Valid activities: " + str(len(filteredData))
    return filteredData



def read_map(fileName):
    mycvs = []
    filteredData = []

    with open(fileName, 'rU') as f:
        reader = csv.reader((line for line in f if not line.isspace()))
        mycsv = list(reader)

    for row in mycsv:
        activity_id = row[0]
        skill_id = row[1]
        if not activity_id == "" and not skill_id == "":
            twoCol = [activity_id, skill_id]
            filteredData.append(twoCol)

    return filteredData

def crossValidate(model, interactions, n_folds): # dic [[row1, row2], [row1, row2]]
    #make the folds 
    bin_size = float(len(interactions)) / n_folds
    print "Bin size(number of students): " + str(bin_size)
    bins = []

    for i in range(n_folds):
        start = int(math.floor(i * bin_size))
        end = int(math.floor((i + 1) * bin_size))
        bin = []
        for index in range(start, end):
            bin = bin + interactions[index]

        bins.append(bin)
        print "Submissions in bin" + str(i) +": "+ str(len(bin))

    # Do crossvalidaion.
    for i in range(n_folds):
        training_set = []
        test_set = []

        for j in range(n_folds):
            if j != i:
                for bin in bins[j]:
                    training_set.append(bin)
            else:
                test_set = bins[j]
        print "CrossValidation # is " + str(i)
        print "Length of Training Set is : " + str(len(training_set))
        model.fit(training_set, i)
        model.writePrediction(test_set, i)

if __name__ == "__main__":
    student_data = raw_input("Enter the student data file in csv format: ")
    skill_map = raw_input("Enter the activity-skill map file in csv format: ")
    m = raw_input("Enter the model to use: {bkt, bkt-cont}: ")
    n_folds = raw_input("Enter the number of cross validation folds: ")

    data = read_data(student_data) # dic {[row1, row2], [row1, row2]}
    skillMap = read_map(skill_map)

    if m == "bkt":
        bkt = ClassicBKT()
        bkt.generateParams()
        bkt.generateSkillMap(skillMap)
        crossValidate(bkt, data, int(n_folds))

