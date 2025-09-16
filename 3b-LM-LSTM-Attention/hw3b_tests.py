import os
import pandas as pd
import sys
import torch
import torch.nn as nn  # pytorch neural network sub-library
import torch.nn.functional as F
import traceback
import warnings

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

# create empty grades file
grades.to_csv('grades.csv', index=False)


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


# COPIED FROM HW2-A
def get_rnn_x_y(data, index, vocab_size):
    x = None
    y = None
    x = data[index:index + 1]
    y = data[index + 1:index + 2]
    x = F.one_hot(torch.tensor(x), num_classes=vocab_size)
    y = torch.tensor(y)
    return x, y


# test case a
def LSTM_check():
    try:
        # make sure we are using the global variable
        global POINTS_A

        ''' test case a '''
        # initialize variables
        score = 0
        input_size = 100
        hidden_size = 50
        linear_layer_params = (hidden_size, input_size)
        LSTM_params = [(input_size, hidden_size), (hidden_size, hidden_size)]
        rnn = _main.MyLSTM(input_size, hidden_size)
        # loop through the param
        for name, param in rnn.named_children():
            # Check to see if the values for linear inputs are correct
            if 'LSTMCell' in str(param) and (param.input_size, param.hidden_size) in LSTM_params:
                score += 1
            elif 'Linear' in str(param) and (param.in_features, param.out_features) == linear_layer_params:
                score += 1
        # confirm score is 3 for full credit
        if score == 3:
            POINTS_A = 5  # set points to 5 if correct
            print('Number of layers found, {}, is correct.'.format(POINTS_A))
        else:
            POINTS_A = 0
            print('Number of layers found, {}, is incorrect.'.format(POINTS_A))
        print('Test A: {}/5'.format(POINTS_A))
        # add grade to grade dataframe
        grades.loc[len(grades)] = ['A', POINTS_A]
        # update grades.csv
        grades.to_csv('grades.csv', index=False)

    except Exception as e:
        print('Test A: 0/5')
        # add grade to grade dataframe
        grades.loc[len(grades)] = ['A', 0]
        # update grades.csv
        grades.to_csv('grades.csv', index=False)

    return


# test case b
def unit_test_LSTM_structure():
    try:
        # make sure we are using the global variable
        global POINTS_B
        POINTS_B = 0

        ''' test case b '''
        # initialize variables
        score = 0
        input_size = 100
        hidden_size = 50
        lstm = _main.MyLSTM(input_size, hidden_size)
        loss_fn = nn.NLLLoss()
        # Make dummy data
        input = torch.randn(1, input_size)
        hc = lstm.init_hidden()
        hc = (hc[0].to('cpu'), hc[1].to('cpu'))
        target = torch.randint(0, input_size, (1,))
        p, hc = lstm(input, hc)
        print(str(p.size()) + ' ' + str(hc[0].size()) + ' ' + str(hc[1].size()))
        if p.size()[1] == input_size:
            score = score + 1
        else:
            print("Element 0 of network output is not the correct size.")
            return 0
        if hc[0].size()[0] == 1 and hc[0].size()[1] == hidden_size:
            score = score + 1
        else:
            print("Hidden output is not the correct size.")
            return 0
        if hc[1].size()[0] == 1 and hc[1].size()[1] == hidden_size:
            score = score + 1
        else:
            print("Cell output is not the correct size.")
            return 0
        loss = loss_fn(p, target)
        fn = loss.grad_fn
        softmax_count = unit_test_count_fn(fn, 'Softmax')
        log_count = unit_test_count_fn(fn, 'Log')
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
        # confirm score is 5 for full credit
        if score == 5:
            POINTS_B = 5
            print('Number of layers found, {}, is correct.'.format(score))
        else:
            POINTS_B = 0
            print('Number of layers found, {}, is incorrect.'.format(score))
        print('Test B: {}/5'.format(POINTS_B))
        # add grade to grade dataframe
        grades.loc[len(grades)] = ['B', POINTS_B]
        # update grades.csv
        grades.to_csv('grades.csv', index=False)

    except Exception as e:
        print('Test B: 0/5')
        # add grade to grade dataframe
        grades.loc[len(grades)] = ['B', 0]
        # update grades.csv
        grades.to_csv('grades.csv', index=False)

    return


# test case c
def eval_lstm_1(max_perplexity):
    try:
        # make sure we are using the global variable
        global POINTS_C
        POINTS_C = 0

        # initialize variables
        net = _main.lstm
        data = _main.TEST
        criterion = _main.criterion_lstm

        # evaluate lstm
        net.eval()
        losses = []
        hc = net.init_hidden()
        with torch.no_grad():
            for i in range(len(data) // 10):
                x, y = get_rnn_x_y(data, i, _main.VOCAB.num_words())
                x = x.float()
                output, hc = net(x, hc)
                loss = criterion(output, y)
                losses.append(loss)
        perplexity = torch.exp(torch.stack(losses).mean())
        print(perplexity.item())

        # confirm perplexity is less than threshold - max_perplexity
        if perplexity.item() < max_perplexity:
            print('Perplexity is less than {}'.format(max_perplexity))
            POINTS_C = 10
            grades.loc[len(grades)] = ['C', POINTS_C]
            print('Test C: {}/10'.format(POINTS_C))
        else:
            print('Perplexity is greater than {}'.format(max_perplexity))
            POINTS_C = 0
            grades.loc[len(grades)] = ['C', POINTS_C]
            print('Test C: {}/10'.format(POINTS_C))
        # update grades.csv
        grades.to_csv('grades.csv', index=False)

    except Exception as e:
        # print traceback
        traceback.print_exc()
        print('Test C: 0/5')
        # add grade to grade dataframe
        grades.loc[len(grades)] = ['C', 0]
        # update grades.csv
        grades.to_csv('grades.csv', index=False)

    return


# test case d
def test_myLSTMCell_structure(MyLSTMCell):
    try:
        # make sure we are using the global variable
        global POINTS_D
        POINTS_D = 0

        ''' test case d '''
        # initialize variables
        input_size = 100
        hidden_size = 50
        score = 0
        cell = MyLSTMCell(input_size=input_size, hidden_size=hidden_size)
        hc = cell.init_hidden()
        hc = (hc[0].to('cpu'), hc[1].to('cpu'))
        input = torch.randn(1, input_size)
        hc = cell(input, hc)
        print(str(hc[0].size()) + ' ' + str(hc[1].size()))
        # Check output shapes
        if hc[0].size()[0] == 1 and hc[0].size()[1] == hidden_size:
            score = score + 1
        else:
            print("hidden is not the correct shape.")
            return 0
        if hc[1].size()[0] == 1 and hc[1].size()[1] == hidden_size:
            score = score + 1
        else:
            print("cell is not the correct shape.")
            return 0
        loss_fn = nn.NLLLoss()
        target = torch.randint(hidden_size, (1,))
        loss = loss_fn(hc[0], target)
        fn = loss.grad_fn
        # Count presence of layers
        # Counts: 3 sigmoids, 2 tanhs, 8 linears (4 input->hidden, 4 hidden->hidden), 3 Mul, 13 Add (8 from linear I think)
        sigmoid_count = unit_test_count_fn(fn, 'Sigmoid')
        tanh_count = unit_test_count_fn(fn, 'Tanh')
        mul_count = unit_test_count_fn(fn, 'Mul')
        add_count = unit_test_count_fn(fn, 'Add')
        linear_count = unit_test_count_fn(fn, 'Addmm')
        if sigmoid_count == 3:
            score = score + 1
        elif sigmoid_count == 0:
            print("No sigmoid layers found.")
        else:
            print("Wrong number of sigmoid layers found.")
        if tanh_count == 2:
            score = score + 1
        elif tanh_count == 0:
            print("No tanh layers found.")
        else:
            print("Wrong number of tanh layers found.")
        if mul_count == 3:
            score = score + 1
        elif mul_count == 0:
            print("No multiplications  found.")
        else:
            print("Wrong number of multiplications found.")
        if add_count - linear_count == 5:
            score = score + 1
        elif add_count - linear_count == 0:
            print("No adds found.")
        else:
            print("Wrong number of adds found.")
        # confirm score is 6 for full credit
        if score == 6:
            POINTS_D = 5  # set points to 5 if correct
            print('Number of layers found, {}, is correct.'.format(score))
        else:
            POINTS_D = 0  # set points to 0 if not correct
            print('Number of layers found, {}, is incorrect.'.format(score))
        print('Test D: {}/5.0'.format(POINTS_D))
        # add grade to grade dataframe
        grades.loc[len(grades)] = ['D', POINTS_D]
        # update grades.csv
        grades.to_csv('grades.csv', index=False)

    except Exception as e:
        print('Test D: 0/5')
        # add grade to grade dataframe
        grades.loc[len(grades)] = ['D', 0]
        # update grades.csv
        grades.to_csv('grades.csv', index=False)

    return


# test case e
def MyLSTMCell_linear_layer_size_check():
    try:
        # make sure we are using the global variable
        global POINTS_E
        POINTS_E = 0

        ''' test case e '''
        # initialize variables
        linear_points = 0
        input_size = 100
        hidden_size = 50
        i_h_layer_params = (input_size, hidden_size)
        h_h_layer_params = (hidden_size, hidden_size)
        i_h_counter = 0
        h_h_counter = 0
        rnn = _main.MyLSTMCell(input_size, hidden_size)
        # loop through the param
        for name, param in rnn.named_children():
            # Check to see if the values for linear inputs are correct
            if 'Linear' in str(param):
                if (param.in_features, param.out_features) == i_h_layer_params:
                    i_h_counter = i_h_counter + 1
                elif (param.in_features, param.out_features) == h_h_layer_params:
                    h_h_counter = h_h_counter + 1
        if i_h_counter == 4:
            linear_points = linear_points + 4
        if h_h_counter == 4:
            linear_points = linear_points + 4
        # confirm score is 8 for full credit
        if linear_points == 8:
            POINTS_E = 5  # set points to 5 if correct
            print('Number of linear layers found, {}, is correct.'.format(linear_points))
        else:
            POINTS_E = 0  # set points to 0 if not correct
            print('Number of linear layers found, {}, is incorrect.'.format(linear_points))
        print('Test E: {}/5.0'.format(POINTS_E))
        # add grade to grade dataframe
        grades.loc[len(grades)] = ['E', POINTS_E]
        # update grades.csv
        grades.to_csv('grades.csv', index=False)

    except Exception as e:
        print('Test E: 0/5')
        # add grade to grade dataframe
        grades.loc[len(grades)] = ['E', 0]
        # update grades.csv
        grades.to_csv('grades.csv', index=False)

    return


# test case f
def test_gate_structure(cell):
    try:
        # make sure we are using the global variable
        global POINTS_F
        POINTS_F = 0

        ''' test case f '''
        # initialize variables
        t1 = unit_test_forget_gate(cell)
        t2 = unit_test_input_gate(cell)
        t3 = unit_test_out_gate(cell)
        t4 = unit_test_cell_memory(cell)
        t5 = unit_test_hidden_out(cell)
        score = t1 + t2 + t3 + t4 + t5
        # confirm score is 22 for full credit
        if score == 22:
            POINTS_F = 10
            print('Number of layers found, {}, is correct.'.format(score))
        else:
            POINTS_F = 0
            print('Number of layers found, {}, is incorrect.'.format(score))
        print('Test F: {}/10'.format(POINTS_F))
        # add grade to grade dataframe
        grades.loc[len(grades)] = ['F', POINTS_F]
        # update grades.csv
        grades.to_csv('grades.csv', index=False)

    except Exception as e:
        print('Test F: 0/10')
        # add grade to grade dataframe
        grades.loc[len(grades)] = ['F', 0]
        # update grades.csv
        grades.to_csv('grades.csv', index=False)

    return


def unit_test_gate(cell, gate=lambda cell, x, h: cell.forget_gate(x, h)):
    score = 0
    input = torch.randn(1, cell.input_size)
    hidden = torch.randn(1, cell.hidden_size)
    # out = cell.forget_gate(input, hidden)
    out = gate(cell, input, hidden)
    fn = out.grad_fn
    sigmoid_count = unit_test_count_fn(fn, "Sigmoid")
    linear_count = unit_test_count_fn(fn, "Addmm")
    add_count = unit_test_count_fn(fn, "AddBackward")
    if sigmoid_count == 1:
        score = score + 1
    elif sigmoid_count == 0:
        print("No sigmoid layers found")
    else:
        print("Wrong number of sigmoids found")
    if linear_count == 2:
        score = score + 1
    elif linear_count == 0:
        print("No linear layers found")
    else:
        print("Wrong number of linear layers found")
    if add_count == 1:
        score = score + 1
    elif add_count == 0:
        print("No adds found")
    else:
        print("Wrong number of adds found")
    good_layers = {'AddmmBackward0': ['AddmmBackward', 'AddBackward0']}
    bad_layers = {'AddmmBackward0': ['SigmoidBackward0'], 'SigmoidBackward0': ['AddmmBackward0', 'AddBackward0']}
    if test_graph_structure(out, good_layers, bad_layers):
        score = score + 1
    else:
        print("order of layers incorrect")
    return score


def unit_test_forget_gate(cell):
    return unit_test_gate(cell, gate=lambda cell, x, h: cell.forget_gate(x, h))


def unit_test_input_gate(cell):
    return unit_test_gate(cell, gate=lambda cell, x, h: cell.input_gate(x, h))


def unit_test_out_gate(cell):
    return unit_test_gate(cell, gate=lambda cell, x, h: cell.out_gate(x, h))


def unit_test_cell_memory(cell):
    score = 0
    i = torch.randn(1, cell.hidden_size)
    f = torch.randn(1, cell.hidden_size)
    x = torch.randn(1, cell.input_size)
    c = torch.randn(1, cell.hidden_size)
    h = torch.randn(1, cell.hidden_size)
    out = cell.cell_memory(i, f, x, h, c)
    target = torch.randint(cell.hidden_size, (1,))
    loss_fn = nn.NLLLoss()
    loss = loss_fn(out, target)
    fn = loss.grad_fn
    tanh_count = unit_test_count_fn(fn, "Tanh")
    print("--")
    linear_count = unit_test_count_fn(fn, "Addmm")
    add_count = unit_test_count_fn(fn, "AddBackward")
    mul_count = unit_test_count_fn(fn, "Mul")
    print(str(tanh_count) + ' ' + str(linear_count) + ' ' + str(add_count) + ' ' + str(mul_count))
    if tanh_count == 1:
        score = score + 1
    elif tanh_count == 0:
        print("No tanh layers found")
    else:
        print("Wrong number of tanhs found")
    if linear_count == 2:
        score = score + 1
    elif linear_count == 0:
        print("No linear layers found")
    else:
        print("Wrong number of linear layers found")
    if add_count == 2:
        score = score + 1
    elif add_count == 0:
        print("No adds found")
    else:
        print("Wrong number of adds found")
    if mul_count == 1:
        score = score + 1
    elif mul_count == 0:
        print("No multiplications found")
    else:
        print("Wrong number of multiplications found")
    good_layers = {'AddmmBackward0': ['AddmmBackward', 'AddBackward0'],
                   'TanhBackward0': ['MulBackward0'],
                   'MulBackward0': ['AddBackward0'],
                   'AddBackward0': ['TanhBackward0', 'NllLossBackward0']}
    bad_layers = {'AddmmBackward0': ['TahnBackward0', 'SigmoidBackward0'],
                  'TanhBackward0': ['AddmmBackward0', 'AddBackward0'],
                  'MulBackward0': ['AddmmBackward0', 'TanhBackward0', 'SigmoidBackward0', 'AddBackward0'],
                  'AddBackward0': ['MulBackward0', 'AddmmBackward0']}
    if test_graph_structure(loss, good_layers, bad_layers):
        score = score + 1
    else:
        print("order of layers incorrect")
    return score


def unit_test_hidden_out(cell):
    score = 0
    o = torch.randn(1, cell.hidden_size)
    c = torch.randn(1, cell.hidden_size)
    # Need to run through some linears first so there will be gradient flow
    # Otherwise a computation graph will not be built
    linear1 = nn.Linear(cell.hidden_size, cell.hidden_size)
    linear2 = nn.Linear(cell.hidden_size, cell.hidden_size)
    temp1 = linear1(o)
    temp2 = linear2(c)
    out = cell.hidden_out(temp1, temp2)
    target = torch.randint(cell.hidden_size, (1,))
    loss_fn = nn.NLLLoss()
    loss = loss_fn(out, target)
    fn = loss.grad_fn
    tanh_count = unit_test_count_fn(fn, "Tanh")
    linear_count = unit_test_count_fn(fn, "Addmm")
    add_count = unit_test_count_fn(fn, "AddBackward")
    mul_count = unit_test_count_fn(fn, "Mul")
    print(str(tanh_count) + ' ' + str(linear_count) + ' ' + str(add_count) + ' ' + str(mul_count))
    if tanh_count == 1:
        score = score + 1
    elif tanh_count == 0:
        print("No tanh layers found")
    else:
        print("Wrong number of tanhs found")
    if linear_count == 2:  # linears are added before hidden_out to force gradient flow through comp graph
        score = score + 1
    else:
        print("Wrong number of linear layers found")
    if add_count == 0:
        score = score + 1
    else:
        print("Wrong number of adds found")
    if mul_count == 1:
        score = score + 1
    elif mul_count == 0:
        print("No multiplications found")
    else:
        print("Wrong number of multiplications found")
    good_layers = {'MulBackward0': ['NllLossBackward0'],
                   'TanhBackward0': ['MulBackward0']}
    bad_layers = {'TanhBackward0': ['NllLossBackward0'],
                  'MulBackward0': ['TanhBackward0', 'SigmoidBackward0']}
    if test_graph_structure(loss, good_layers, bad_layers):
        score = score + 1
    else:
        print("order of layers incorrect")
    return score


# test case g
def eval_lstm_2(max_perplexity):
    try:
        # make sure we are using the global variable
        global POINTS_G
        POINTS_G = 0

        # initialize variables
        net = _main.my_cell_lstm
        data = _main.TEST
        criterion = _main.criterion_my_cell

        # evaluate lstm
        net.eval()
        losses = []
        hc = net.init_hidden()
        with torch.no_grad():
            for i in range(len(data) // 10):
                x, y = get_rnn_x_y(data, i, _main.VOCAB.num_words())
                x = x.float()
                output, hc = net(x, hc)
                loss = criterion(output, y)
                losses.append(loss)
        perplexity = torch.exp(torch.stack(losses).mean())

        # confirm perplexity is less than threshold - max_perplexity
        print(perplexity.item())
        if perplexity.item() < max_perplexity:
            print('Perplexity is less than {}'.format(max_perplexity))
            POINTS_G = 20
            grades.loc[len(grades)] = ['G', POINTS_G]
            print('Test G: {}/20'.format(POINTS_G))
        else:
            print('Perplexity is greater than {}'.format(max_perplexity))
            POINTS_G = 0
            grades.loc[len(grades)] = ['G', POINTS_G]
            print('Test G: {}/20'.format(POINTS_G))
        # update grades.csv
        grades.to_csv('grades.csv', index=False)

    except Exception as e:
        print('Test G: 0/5')
        # add grade to grade dataframe
        grades.loc[len(grades)] = ['G', 0]
        # update grades.csv
        grades.to_csv('grades.csv', index=False)

    return


# test case h
def test_attention_structure():
    try:
        # make sure we are using the global variable
        global POINTS_H
        POINTS_H = 0

        ''' test case h '''
        # initialize variables
        input_size = 100
        hidden_size = 64
        max_length = 5
        score = 0
        decoder = _main.MyAttentionDecoder(input_size=input_size, hidden_size=hidden_size, max_length=max_length)
        hc = decoder.init_hidden()
        hc = (hc[0].to('cpu'), hc[1].to('cpu'))
        x = torch.randn(1, input_size)
        encoder_outputs = torch.randn(max_length, hidden_size)
        probs, hc = decoder(x, hc, encoder_outputs)
        print(str(probs.size()) + ' ' + str(hc[0].size()) + ' ' + str(hc[1].size()))
        # Check output shapes
        if hc[0].size()[0] == 1 and hc[0].size()[1] == hidden_size:
            score = score + 1
        else:
            print("hidden is not the correct shape.")
        if hc[1].size()[0] == 1 and hc[1].size()[1] == hidden_size:
            score = score + 1
        else:
            print("cell is not the correct shape.")
        if probs.size()[0] == 1 and probs.size()[1] == input_size:
            score = score + 1
        else:
            print("probabilities not the correct shape.")
        loss_fn = nn.NLLLoss()
        target = torch.randint(input_size, (1,))
        loss = loss_fn(probs, target)
        fn = loss.grad_fn
        print('--')
        # Count presence of layers
        # Counts (wrong): 2 softmaxes (one is logsoftmax or log and softmax), 1 bmm, 1 log, X cats (more than 2 so can't use), 1 relu
        # Counts (without LSTMCell): 1 softmax, 1 logSoftmax, 1 bmm, X cats (more than 2 so can't use), 1 relu
        # Counts (with LSTMCell): 5 softmax+logSoftmax, 4 bmm, 4 relu (don't know how to separate the two yet)
        softmax_count = unit_test_count_fn(fn, 'Softmax')
        bmm_count = unit_test_count_fn(fn, 'Bmm')
        relu_count = unit_test_count_fn(fn, 'Relu')
        print(softmax_count, bmm_count, relu_count)
        if softmax_count >= 5:
            score = score + 1
        elif softmax_count == 0:
            print("No softmax/logSoftmax layers found.")
        else:
            print("Wrong number of softmax/logSoftmax layers found.")
        if relu_count >= 4:
            score = score + 1
        elif relu_count == 0:
            print("No relu layers found.")
        else:
            print("Wrong number of relu layers found.")
        if bmm_count >= 4:
            score = score + 1
        elif bmm_count == 0:
            print("No bmm (batch matrix multiplication) found.")
        else:
            print("Wrong number of bmm (batch matrix multiplication) found.")
        # Check LSTM
        input_size = 100
        hidden_size = 64
        LSTM_params = (hidden_size, hidden_size)
        for name, param in decoder.named_children():
            # Check to see if the values for lstm inputs are correct
            if 'LSTMCell' in str(param) and (param.input_size, param.hidden_size) == LSTM_params:
                score += 1
        # confirm score is 7 for full credit
        if score == 7:
            POINTS_H = 10  # set points to 10 if correct
            print('Number of layers found, {}, is correct.'.format(score))
        else:
            POINTS_H = 0  # set points to 0 if not correct
            print('Number of layers found, {}, is incorrect.'.format(score))
        print('Test H: {}/10'.format(POINTS_H))
        # add grade to grade dataframe
        grades.loc[len(grades)] = ['H', POINTS_H]
        # update grades.csv
        grades.to_csv('grades.csv', index=False)

    except Exception as e:
        print('Test H: 0/10')
        # add grade to grade dataframe
        grades.loc[len(grades)] = ['H', 0]
        # update grades.csv
        grades.to_csv('grades.csv', index=False)

    return


# test case i
def attention_linear_layer_size_check():
    try:
        # make sure we are using the global variable
        global POINTS_I
        POINTS_I = 0

        ''' test case i '''
        # initialize variables
        linear_points = 0
        max_length = 5  # The number of past hidden states that can be attended to
        input_size = 100
        hidden_size = 64
        input_layer_params = (input_size, hidden_size)
        input_hidden_layer_params = (2 * hidden_size, max_length)
        hidden_hidden_layer_params = (2 * hidden_size, hidden_size)
        output_layer_params = (hidden_size, input_size)
        att_decoder = _main.MyAttentionDecoder(input_size=input_size, hidden_size=hidden_size, max_length=max_length)
        # loop through the param
        for name, param in att_decoder.named_children():
            # Check to see if the values for linear inputs are correct
            if 'Linear' in str(param):
                if (param.in_features, param.out_features) == input_layer_params:
                    linear_points = linear_points + 1
                elif (param.in_features, param.out_features) == input_hidden_layer_params:
                    linear_points = linear_points + 1
                elif (param.in_features, param.out_features) == hidden_hidden_layer_params:
                    linear_points = linear_points + 1
                elif (param.in_features, param.out_features) == output_layer_params:
                    linear_points = linear_points + 1
        # confirm score is 4 for full credit
        if linear_points == 4:
            POINTS_I = 10  # set points to 5 if correct
            print('Number of linear layers found, {}, is correct.'.format(linear_points))
        else:
            POINTS_I = 0  # set points to 0 if not correct
            print('Number of linear layers found, {}, is incorrect.'.format(linear_points))
        print('Test I: {}/10'.format(POINTS_I))
        # add grade to grade dataframe
        grades.loc[len(grades)] = ['I', POINTS_I]
        # update grades.csv
        grades.to_csv('grades.csv', index=False)

    except Exception as e:
        print('Test I: 0/10')
        # add grade to grade dataframe
        grades.loc[len(grades)] = ['I', 0]
        # update grades.csv
        grades.to_csv('grades.csv', index=False)

    return


# test case j
def eval_attn(max_perplexity=1000):
    try:
        # make sure we are using the global variable
        global POINTS_J
        POINTS_J = 0

        # initialize variables
        encoder = _main.attn_encoder
        decoder = _main.attn_decoder
        data = _main.TEST
        criterion = _main.attn_criterion
        max_length = _main.ATTN_MAX_LENGTH
        vocab_size = _main.VOCAB.num_words()

        # evaluate attn
        encoder.eval()
        decoder.eval()
        losses = []
        hc = decoder.init_hidden()
        history = torch.zeros(max_length, encoder.hidden_size)
        with torch.no_grad():
            for i in range(len(data) // 10):
                x, y = get_rnn_x_y(data, i, vocab_size)
                x = x.float()
                encoder_hc = encoder(x, hc)
                output = encoder_hc[0]
                # unbatch the hidden
                output = output[0]
                # Shift all the previous outputs
                # Grab elements 1...max (dropping row 0) and flatten
                history = history[1:, :].view(-1)
                # Add the new output
                history = torch.cat((history, output.detach()))
                # re-fold
                history = history.view(max_length, -1)
                # decoder's input hc is the encoder's output hc
                decoder_hc = encoder_hc
                # Call the decoder
                decoder_output, decoder_hc = decoder(x, decoder_hc, history)
                loss = criterion(decoder_output, y)
                losses.append(loss)
        perplexity = torch.exp(torch.stack(losses).mean())
        print('Perplexity = ' + str(perplexity.item()))
        # confirm perplexity is less than threshold - max_perplexity
        if perplexity.item() < max_perplexity:
            print('Perplexity is less than 1000')
            POINTS_J = 20
            grades.loc[len(grades)] = ['J', POINTS_J]
            print('Test J: {}/20'.format(POINTS_J))
        else:
            print('Perplexity is greater than 1000')
            POINTS_J = 0
            grades.loc[len(grades)] = ['J', POINTS_J]
            print('Test J: {}/20'.format(POINTS_J))
        # update grades.csv
        grades.to_csv('grades.csv', index=False)

    except Exception as e:
        print('Test J: 0/20')
        # add grade to grade dataframe
        grades.loc[len(grades)] = ['J', 0]
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
        print('Your projected points for this assignment is {}/100.'.format(TOTAL_POINTS))
        print('\nNOTE: THIS IS NOT YOUR FINAL GRADE. YOUR FINAL GRADE FOR THIS ASSIGNMENT WILL BE AT LEAST {} '
              'OR MORE, BUT NOT LESS\n'.format(TOTAL_POINTS))

    except Exception as e:
        print('Error during execution of FINAL_GRADE:')
        # print traceback
        traceback.print_exc()

    return
