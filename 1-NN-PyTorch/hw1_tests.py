import os
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import random
import sys
import torch
import torch.nn as nn  # pytorch neural network sub-library
import traceback

_main = sys.modules['__main__']

# create grade data frame
grades = pd.DataFrame(columns=['question', 'score'])

# delete grades.csv if it exists
try:
    os.remove('grades.csv')
except OSError:
    pass

# save empty grades file (initialize)
grades.to_csv('grades.csv', index=False)


# Function to test get_batch()
def get_batch_unit_test(num_tests=10, print_all=False, auto_grade=False, train_data=_main.make_data(1000)):
    '''
  Args:
    num_test: number of tests we are running
    print_all: this print the values for debugging
    auto_grade: if true, this will not print anything
    train_data: the data set that we are using to test 
  Return:
    points earn for this test
  '''
    # Projected points
    points = 0
    # Run the unit test
    for i in range(num_tests):
        # Pick a random batch
        batch_size = random.randint(1, 10)
        # Pick a random index, except the first and last test
        index = 0
        if i > 0 and i < num_tests - 1:
            index = random.randint(0, (len(train_data) // batch_size) - 1)
        elif i == num_tests - 1:
            index = (len(train_data) // batch_size) - 1
        # Get the batch
        x, y = _main.get_batch(train_data, batch_size, index)
        # print batch info if user want
        if print_all:
            print('batch_size=', batch_size, 'index=', index)
            print('x:')
            print(x)
            print('shape of x=', x.size())
            print('y:')
            print(y)
            print('shape of y=', y.size())
        # Check the shape of the output tensors and add points
        if auto_grade:
            if x.size()[0] == batch_size and x.size()[1] == 4 and y.size()[0] == batch_size and y.size()[1] == 2:
                # output tensors are the correct shape
                points += 1
        else:
            if x.size()[0] == batch_size and x.size()[1] == 4 and y.size()[0] == batch_size and y.size()[1] == 2:
                # output tensors are the correct shape
                print("Test case #{}: pass\n".format(i + 1))
                points += 1
            else:
                print("Test case #{}: FAILED\n".format(i + 1))
    return points


def unit_test_count_fn(fn, target):
    count = 0
    name = str(type(fn).__name__)
    if name == target:
        count = count + 1
    if hasattr(fn, 'next_functions'):
        for u in fn.next_functions:
            if u[0] is not None:
                count = count + unit_test_count_fn(u[0], target)
    return count


def build_child_hash(fn, children, child_hash):
    name = (str(type(fn).__name__), id(fn))
    child_hash[name] = children
    if hasattr(fn, 'next_functions'):
        for u in fn.next_functions:
            if u[0] is not None:
                build_child_hash(u[0], [name] + children, child_hash)


def test_graph_structure(output_var, good_layers, bad_layers):
    # Get the bottom of the computation graph
    fn = output_var.grad_fn
    # build child_hash
    child_hash = {}
    build_child_hash(fn, [], child_hash)
    # Iterate through hash
    for layer in child_hash.keys():
        name = layer[0]
        # Are we looking at a layer we care about?
        if name in good_layers.keys():
            # Check there is is good layer before we see a bad layer in children
            check = False
            children = child_hash[layer]
            for child in children:
                if child[0] in good_layers[name]:
                    check = True
                    break
                elif child in bad_layers[name]:
                    print("layer " + str(name) + " is followed by " + str(child[0]) + " before a layer in " + str(
                        good_layers[name]))
                    return False
            if not check:
                print("layer " + str(name) + " is not followed by any layers from " + str(good_layers[name]))
                return False
    return True


# This function will test to see if the order is correct
def CarNet_unit_test_1():
    linear_points = 0
    tanh_points = 0
    in_list = [4, 16, 8]
    out_list = [16, 8, 2]
    counter = 0
    # loop through the param
    for name, param in _main.net.named_children():
        # Check to see if the values for linear inputs are correct
        if 'Linear' in str(param) and counter % 2 == 0:
            if param.in_features == in_list[linear_points] and param.out_features == out_list[linear_points]:
                linear_points += 1
        # count number of Tanh functions
        elif 'Tanh' in str(param) and counter % 2 == 1:
            tanh_points += 1
        else:
            pass
        counter += 1
    # return the points earned for this functions
    return linear_points + tanh_points


# Unit Test Case for Tanh()
def unit_test_count_Tanh(output_var):
    fn = output_var.grad_fn
    return unit_test_count_fn(fn, 'TanhBackward0')


# Part A
def PART_A():

    try:
        # make sure we are using the global variable
        global POINTS_A

        ''' test case a '''
        NUM_TESTS = 10
        DATA = _main.make_data(10000)
        POINTS_A = get_batch_unit_test(num_tests=NUM_TESTS, print_all=False, auto_grade=True,
                                       train_data=DATA) / NUM_TESTS * 40
        print('Part A: {}/40.0'.format(POINTS_A))
        # add grade to grade dataframe
        grades.loc[len(grades)] = ['A', POINTS_A]
        # update grades.csv
        grades.to_csv('grades.csv', index=False)

    except Exception as e:
        print('Part A: 0.0')
        print('Error during execution of Part A:')
        # print traceback
        traceback.print_exc()
        # add grade to grade dataframe
        grades.loc[len(grades)] = ['A', 0.0]
        # update grades.csv
        grades.to_csv('grades.csv', index=False)

    return


# Part B
def PART_B():

    try:
        # make sure we are using the global variable
        global POINTS_B

        ''' test case b '''
        # Get a batch
        DATA = _main.make_data(1000)
        x, y = _main.get_batch(DATA, 8, 0)
        # Call the forward pass
        y_hat = _main.net(x)
        # check tanh
        TANH_POINTS = 0.0
        # create the loss function
        loss_fn = nn.MSELoss()
        loss = loss_fn(y_hat, y)
        tanh_count = unit_test_count_Tanh(loss)
        if tanh_count == 3:
            TANH_POINTS = 5.0
        # Test if the tan and linear alternate
        # POINTS = CarNet_unit_test_1()/6*35 + TANH_POINTS
        STRUCTURE_POINTS = 0.0
        good_layers = {'AddmmBackward0': ['TanhBackward0']}
        bad_layers = {'AddmmBackward0': ['AddmmBackward0', 'MseLossBackward0']}
        if test_graph_structure(loss, good_layers, bad_layers):
            STRUCTURE_POINTS = 35.0
        POINTS_B = TANH_POINTS + STRUCTURE_POINTS
        print('Part B: {}/40.0'.format(POINTS_B))
        # add grade to grade dataframe
        grades.loc[len(grades)] = ['B', POINTS_B]
        # update grades.csv
        grades.to_csv('grades.csv', index=False)

    except Exception as e:
        print('Part B: 0.0')
        print('Error during execution of Part B:')
        # print traceback
        traceback.print_exc()
        # add grade to grade dataframe
        grades.loc[len(grades)] = ['B', 0.0]
        # update grades.csv
        grades.to_csv('grades.csv', index=False)

    return


# Part C
def PART_C():

    try:
        # make sure we are using the global variable
        global POINTS_C

        ''' test case c '''
        NUM_TESTS = 10
        NUM_DATA_POINTS = 1000
        THRESHOLD = 0.9
        POINTS_C = 0.0
        # test over X trials
        for i in range(NUM_TESTS):
            test_data = _main.make_data(NUM_DATA_POINTS)
            score = _main.evaluate(_main.net, test_data)
            accuracy = (len(test_data) - score) / len(test_data)
            print("test", i, "accuracy:", accuracy)
            POINTS_C = POINTS_C + min(1.0, (1.0 - (THRESHOLD - accuracy)))
        POINTS_C = (POINTS_C / NUM_TESTS) * 20.0
        print('Part C: {}/20.0'.format(POINTS_C))
        # add grade to grade dataframe
        grades.loc[len(grades)] = ['C', POINTS_C]
        # update grades.csv
        grades.to_csv('grades.csv', index=False)

    except Exception as e:
        print('Part C: 0.0')
        print('Error during execution of Part C:')
        # print traceback
        traceback.print_exc()
        # add grade to grade dataframe
        grades.loc[len(grades)] = ['C', 0.0]
        # update grades.csv
        grades.to_csv('grades.csv', index=False)

    return


# final grade
def FINAL_GRADE():

    try:
        ''' final grade '''
        # get grades from csv
        grades = pd.read_csv('grades.csv')
        # get total points
        TOTAL_POINTS = grades['score'].sum()
        print('Your projected points for this assignment is {}/100.'.format(TOTAL_POINTS))
        print('\nNOTE: THIS IS NOT YOUR FINAL GRADE. YOUR FINAL GRADE FOR THIS ASSIGNMENT WILL BE AT LEAST {} '
              'OR MORE, BUT NOT LESS\n'.format(TOTAL_POINTS))

    except Exception as e:
        print('Error during execution of FINAL_GRADE:')
        # print traceback
        traceback.print_exc()

    return