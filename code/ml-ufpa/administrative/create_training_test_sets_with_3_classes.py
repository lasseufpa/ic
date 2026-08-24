'''
Generates validation, training and test (all disjoint) sets.
Uses 3 classes instead of 2.
Aldebaro. 2021
'''
import csv
from random import randint
from random import seed
from csv import reader
from csv import writer
import numpy as np

file_name = 'matriculas.txt'
#number of examples
Nte = 4
Ntr = 7
Nva = 5
space_dimension = 3
num_classes = 3
min_x = np.array([-2, -4, -6]) # num of elements = num_classes
max_x = np.array([5, 3, 1])

seed(1231) #random seed. It will change during class

#writes a set of examples
def create_train_test_or_validation_set(output_file_name, num_examples=10):
    with open(output_file_name, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')

        #write examples of each class
        num_examples_per_class = int(num_examples/num_classes)
        for output_classe in range(num_classes):
            if output_classe == num_classes-1:
                num_examples_per_class = num_examples - num_examples_per_class*(num_classes-1)
            write_examples_in_csvwriter(output_classe, min_x[output_classe],
            max_x[output_classe], num_examples_per_class, space_dimension, csvwriter)

#writes CSV lines of examples in csvwriter already opened
def write_examples_in_csvwriter(output_classe, min_x, max_x, num_examples, space_dimension, csvwriter):
    for i in range(num_examples):
        output_row = list()
        for j in range(space_dimension):
            if j==0:
                output_row.append(1000*randint(min_x, max_x))
            else:
                output_row.append(randint(min_x, max_x))
        output_row.append(output_classe)
        csvwriter.writerow(output_row)

#Generate validation, train and test files:
# open file in read mode
with open(file_name, 'r') as read_obj:
    # pass the file object to reader() to get the reader object
    csv_reader = reader(read_obj)
    # Iterate over each row in the csv using reader object
    for row in csv_reader:
        # row variable is a list that represents a row in csv
        matricula = row[0]
        output_file_name = 'estudante_' + matricula + '_train.txt'
        print(output_file_name)
        create_train_test_or_validation_set(output_file_name, num_examples=Ntr)
        output_file_name = 'estudante_' + matricula + '_test.txt'
        print(output_file_name)        
        create_train_test_or_validation_set(output_file_name, num_examples=Nte)
        output_file_name = 'estudante_' + matricula + '_validation.txt'
        print(output_file_name)        
        create_train_test_or_validation_set(output_file_name, num_examples=Nva)