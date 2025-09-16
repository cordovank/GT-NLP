import numpy as np
import os
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import random
import scipy
import sys
import torch
import torch.nn.functional as F
import traceback
import warnings

from scipy import stats
from torch import nn

# ignore all warnings
warnings.filterwarnings('ignore')

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


def unit_test_count_fn(fn, target):
    count = 0
    name = str(type(fn).__name__)
    print(name)
    if target in name:
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


def RNN_linear_layer_size_check():
    linear_points = 0
    input_size = 100
    hidden_size = 50
    layers = [(input_size + hidden_size, hidden_size), (hidden_size, input_size)]
    counter = 0
    rnn = _main.MyRNN(input_size, hidden_size)
    # loop through the param
    for name, param in rnn.named_children():
        # Check to see if the values for linear inputs are correct
        if 'Linear' in str(param) and (param.in_features, param.out_features) in layers:
            linear_points += 1
    # return the points earned for this functions
    return linear_points


# test case a
def unit_test_RNN_structure():
    try:
        # ensure we are using the global variable
        global POINTS_A
        POINTS_A = 0

        ''' test case a '''
        score = 0
        input_size = 100
        hidden_size = 50
        rnn = _main.MyRNN(input_size, hidden_size)
        loss_fn = nn.NLLLoss()
        # Make dummy data
        input = torch.randn(1, input_size)
        hidden = torch.zeros(1, hidden_size)
        target = torch.randint(0, input_size, (1,))
        p, h = rnn(input, hidden)
        if p.size()[1] == input_size:
            score = score + 1
        else:
            print("Element 0 of network output is not the correct size.")
        if h.size()[1] == hidden_size:
            score = score + 1
        else:
            print("Element 1 of network output is not the correct size.")
        loss = loss_fn(p, target)
        fn = loss.grad_fn
        softmax_count = unit_test_count_fn(fn, 'Softmax')
        log_count = unit_test_count_fn(fn, 'Log')
        sigmoid_count = unit_test_count_fn(fn, 'Log')
        if softmax_count == 1:
            score = score + 1
        elif softmax_count == 0:
            print("No softmax layer found.")
        else:
            print("Too many softmax layers found.")
        if log_count == 1:
            score = score + 1
        elif log_count == 0:
            print("No log layer found.")
        else:
            print("Too many log layers found.")
        if sigmoid_count == 1:
            score = score + 1
        elif sigmoid_count == 0:
            print("No sigmoid layer found.")
        else:
            print("Too many sigmoid layers found.")
        child_hash = {}
        build_child_hash(fn, [], child_hash)
        good_layers = {'LogSoftmaxBackward0': ['NllLossBackward0'],
                       'LogBackward0': ['NllLossBackward0'],
                       'SigmoidBackward0': ['AddmmBackward0'],
                       'SoftmaxBackward0': ['LogBackward0'],
                       'AddmmBackward0': ['LogSoftmaxBackward0', 'SoftmaxBackward0']
                       }
        bad_layers = {'LogSoftmaxBackward0': ['AddmmBackward0, DropoutBackward0, SigmoidBackward0'],
                      'LogBackward0': ['AddmmBackward0, DropoutBackward0, SigmoidBackward0'],
                      'SigmoidBackward0': ['LogSoftmaxBackward0', 'LogBackward0', 'SoftmaxBackward0'],
                      'AddmmBackward0': ['LogBackward0']
                      }
        if test_graph_structure(loss, good_layers, bad_layers):
            score = score + 5

        # confirm score is 10 for full credit, otherwise 0
        if score == 10:
            POINTS_A = score
            print('Test A: {}/10'.format(POINTS_A))
            # add grade to grade dataframe
            grades.loc[len(grades)] = ['A', POINTS_A]
        else:
            print('Test A: 0/10')
            # add grade to grade dataframe
            grades.loc[len(grades)] = ['A', 0]
        # update grades.csv
        grades.to_csv('grades.csv', index=False)

    except Exception as e:
        print('Test A: 0/10')
        print('Error during execution of Test A:')
        # print traceback
        traceback.print_exc()
        # add grade to grade dataframe
        grades.loc[len(grades)] = ['A', 0]
        # update grades.csv
        grades.to_csv('grades.csv', index=False)

    return


def test_onehot(v, index, vocab_size):
    check = False
    if v[0][index] == 1:
        check = True
    for j in range(vocab_size):
        if j != index and v[0][j] != 0:
            check = False
    return check


# test case b
def unit_test_token2onehot():
    # ensure we are using the global variable
    global POINTS_B
    POINTS_B = 0

    ''' test case b '''
    try:
        score = 0
        vocab_size = 9
        v = _main.token2onehot(0, vocab_size)
        if v.size()[0] == 1 and v.size()[1] == vocab_size:
            score = score + 1
        else:
            print("token2onehot does not return a tensor of the correct shape")
        for i in range(vocab_size):
            v = _main.token2onehot(i, vocab_size)
            if test_onehot(v, i, vocab_size):
                score = score + 1

        # confirm score is 10 for full credit, otherwise 0
        if score == 10:
            POINTS_B = score
            print('Test B: {}/10'.format(POINTS_B))
            # add grade to grade dataframe
            grades.loc[len(grades)] = ['B', POINTS_B]
        else:
            print('Test B: 0/10')
            # add grade to grade dataframe
            grades.loc[len(grades)] = ['B', 0]
        # update grades.csv
        grades.to_csv('grades.csv', index=False)

    except Exception as e:
        print('Test B: 0/10')
        print('Error during execution of Test B:')
        # print traceback
        traceback.print_exc()
        # add grade to grade dataframe
        grades.loc[len(grades)] = ['B', 0]
        # update grades.csv
        grades.to_csv('grades.csv', index=False)

    return


# test case c
def unit_test_get_xy():
    try:
        # ensure we are using the global variable
        global POINTS_C
        POINTS_C = 0

        ''' test case c '''
        score = 0
        text = ' '.join([str(i) for i in range(101)])
        vocab = _main.Vocab('test')
        vocab.add_sentence(text)
        encoded_text = np.array([vocab.word2index(word) for word in text.split()])
        for i in range(vocab.num_words() - 3):
            x, y = _main.get_rnn_x_y(encoded_text, i)
            if test_onehot(x, i + 2, vocab.num_words()) and y.item() == i + 3:
                score = score + 1

        # confirm score is 100 for full credit, otherwise 0
        if score == 100:
            POINTS_C = 10
            print('Test C: {}/10'.format(POINTS_C))
            # add grade to grade dataframe
            grades.loc[len(grades)] = ['C', POINTS_C]
        else:
            print('Test C: 0/10')
            # add grade to grade dataframe
            grades.loc[len(grades)] = ['C', 0]
        # update grades.csv
        grades.to_csv('grades.csv', index=False)

    except Exception as e:
        print('Test C: 0/10')
        print('Error during execution of Test C:')
        # print traceback
        traceback.print_exc()
        # add grade to grade dataframe
        grades.loc[len(grades)] = ['C', 0]
        # update grades.csv
        grades.to_csv('grades.csv', index=False)

    return


# test case d
def evaluate_rnn(model, data, criterion):
    try:
        # ensure we are using the global variable
        global POINTS_D
        POINTS_D = 0

        ''' test case d '''
        model.eval()
        hidden_state = model.init_hidden()
        losses = []
        with torch.no_grad():
            for i in range(len(data)//10):
              x, y = _main.get_rnn_x_y(data, i)
              x = x.float()
              output, new_hidden = model(x, hidden_state)
              hidden_state = new_hidden.detach()
              loss = criterion(output, y)
              losses.append(loss)
        perplexity = torch.exp(torch.stack(losses).mean())
        print('Perplexity: ', perplexity)
        # confirm perplexity less than 2000 for full points, otherwise 0 points awarded
        if perplexity < 2000:
            POINTS_D = 20
            print('Test D: {}/20'.format(POINTS_D))
            # add grade to grade dataframe
            grades.loc[len(grades)] = ['D', POINTS_D]
        else:
            print('Test D: 0/20')
            # add grade to grade dataframe
            grades.loc[len(grades)] = ['D', 0]
        # update grades.csv
        grades.to_csv('grades.csv', index=False)

    except Exception as e:
        print('Test D: 0/20')
        print('Error during execution of Test D:')
        # print traceback
        traceback.print_exc()
        # add grade to grade dataframe
        grades.loc[len(grades)] = ['D', 0]
        # update grades.csv
        grades.to_csv('grades.csv', index=False)

    return


# test case e
def unit_test_my_sample():
    try:
        # ensure we are using the global variable
        global POINTS_E
        POINTS_E = 0

        ''' test case e '''
        # one spike
        num_trials = 50
        output_size = 100
        score = 0
        for i in range(num_trials):
            target = random.randint(0, output_size - 1)
            p = torch.zeros(1, output_size)
            p[0][target] = 111
            p = F.log_softmax(p)
            index = _main.my_sample(p)
            if index == target:
                score = score + 1
        # plateau
        for i in range(num_trials):
            target = random.randint(0, output_size - 11)
            p = torch.zeros(1, output_size)
            for j in range(10):
                p[0][target + j] = 111
            p = F.log_softmax(p)
            index = _main.my_sample(p)
            if index > target and index < target + 10:
                score = score + 1

        # print score
        print('Score: ', score)

        # if score > 90 then full credit, otherwise 0
        if score > 90:
            POINTS_E = 10
            print('Test E: {}/10'.format(POINTS_E))
            # add grade to grade dataframe
            grades.loc[len(grades)] = ['E', POINTS_E]
        else:
            print('Test E: 0/10')
            # add grade to grade dataframe
            grades.loc[len(grades)] = ['E', 0]
        # update grades.csv
        grades.to_csv('grades.csv', index=False)

    except Exception as e:
        print('Test E: 0/10')
        print('Error during execution of Test E:')
        # print traceback
        traceback.print_exc()
        # add grade to grade dataframe
        grades.loc[len(grades)] = ['E', 0]
        # update grades.csv
        grades.to_csv('grades.csv', index=False)

    return


# test case f
def unit_test_my_temperature_sample():
    try:
        # ensure we are using the global variable
        global POINTS_F
        POINTS_F = 0

        ''' test case f '''
        # two spikes
        num_trials = 10000
        output_size = 100
        t1_results = []
        t0_results = []
        baseline = [num_trials / 2] * num_trials
        p = torch.zeros(1, output_size)
        p[0][0] = 11
        p[0][1] = 12
        p = F.log_softmax(p)
        for j in range(num_trials):
            index_1 = _main.my_temperature_sample(p, 1.0)
            index_0 = _main.my_temperature_sample(p, 0.01)
            t1_results.append(int(index_1 == 1))
            t0_results.append(int(index_0 == 1))
        p_value = scipy.stats.ttest_ind(t1_results, t0_results).pvalue

        # print p-value
        print('p-value: ', p_value)

        # if p-value == 0.0 then full credit, otherwise 0
        if p_value == 0.0:
            POINTS_F = 10
            print('Test F: {}/10'.format(POINTS_F))
            # add grade to grade dataframe
            grades.loc[len(grades)] = ['F', POINTS_F]
        else:
            print('Test F: 0/10')
            # add grade to grade dataframe
            grades.loc[len(grades)] = ['F', 0]
        # update grades.csv
        grades.to_csv('grades.csv', index=False)

    except Exception as e:
        print('Test F: 0/10')
        print('Error during execution of Test F:')
        # print traceback
        traceback.print_exc()
        # add grade to grade dataframe
        grades.loc[len(grades)] = ['F', 0]
        # update grades.csv
        grades.to_csv('grades.csv', index=False)

    return


# final grade
def final_grade():
    try:
        ''' final grade '''
        # get grades from csv
        grades = pd.read_csv('grades.csv')
        # get total points
        TOTAL_POINTS = grades['score'].sum()
        print('Your projected points for this assignment is {}/70.'.format(TOTAL_POINTS))
        print('\nNOTE: THIS IS NOT YOUR FINAL GRADE. YOUR FINAL GRADE FOR THIS ASSIGNMENT WILL BE AT LEAST {} '
              'OR MORE, BUT NOT LESS\n'.format(TOTAL_POINTS))

    except Exception as e:
        print('Error during execution of FINAL_GRADE:')
        # print traceback
        traceback.print_exc()

    return
