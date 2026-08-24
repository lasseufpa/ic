'''
Generates validation, training and test (all disjoint) sets.
Restricted to two classes (binary problems).
Aldebaro. 2022
'''
import csv
from random import randint
from random import seed
from csv import reader
from csv import writer

file_name = 'matriculas.txt'
#number of examples
Nte = 5 #test
Ntr = 8 #train
Nva = 4 #validation
space_dimension = 2 #input space dimension (number of features)

seed(131) #random seed. It will change during class

#writes a set of examples
def create_training_or_test_set(output_file_name, num_examples=10):
    #space_dimension = 2
    with open(output_file_name, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')

        #if want to write Weka header
        #header = create_header()
        #csvwriter.writerows(header)

        #write examples of class 0
        num_examples_class_0 = int(num_examples/2)
        output_classe = 0
        min_x = -4
        max_x = 4
        write_examples_in_csvwriter(output_classe, min_x, max_x, num_examples_class_0, space_dimension, csvwriter)
        #write examples of class 1
        output_classe = 1
        min_x = -6
        max_x = 0
        num_examples_class_1 = num_examples - num_examples_class_0
        write_examples_in_csvwriter(output_classe, min_x, max_x, num_examples_class_1, space_dimension, csvwriter)

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
        create_training_or_test_set(output_file_name, num_examples=Ntr)
        output_file_name = 'estudante_' + matricula + '_test.txt'
        print(output_file_name)        
        create_training_or_test_set(output_file_name, num_examples=Nte)
        output_file_name = 'estudante_' + matricula + '_validation.txt'
        print(output_file_name)        
        create_training_or_test_set(output_file_name, num_examples=Nva)