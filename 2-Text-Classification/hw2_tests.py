import os
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import sys
import torch
import torch.nn as nn  # pytorch neural network sub-library
import traceback

from sklearn.metrics import classification_report

_main = sys.modules['__main__']

# create grade data frame
grades = pd.DataFrame(columns=['question', 'score'])

# delete grades.csv if it exists
try:
    os.remove('grades.csv')
except OSError:
    pass

# save empty grades file
grades.to_csv('grades.csv', index=False)


# test case a
def test_naive_bayes(x_train, y_train, x_test, y_test):
    try:
        # make sure we are using the global variable
        global POINTS_A
        POINTS_A = 0

        ''' test case a '''
        # Get the positive and negative feature probabilities
        pos_probs = _main.prob_pos_given_features(x_train, y_train)
        neg_probs = _main.prob_neg_given_features(x_train, y_train)
        correct = 0  # How many tests are correct
        # Iterate through the test set
        for x, y in zip(x_test, y_test):
            # Get the naive_bayes label
            label = _main.naive_bayes(x, pos_probs, neg_probs)
            # Compare the label against the true label
            correct = correct + int(label == y)
        # check if score exceeds threshold
        if correct / x_test.shape[0] >= 0.78:
            print('Accuracy: ', correct / x_test.shape[0])
            print('Test A: 20/20')
            POINTS_A = 20
        else:
            print('Accuracy: ', correct / x_test.shape[0])
            print('Test A: 0/20')
            POINTS_A = 0
        # add grade to dataframe
        grades.loc[len(grades)] = ['A', POINTS_A]
        # save grades
        grades.to_csv('grades.csv', index=False)
    except Exception as e:
        print('Test A: 0/20')
        print('Error during execution of Test A:')
        # print traceback
        traceback.print_exc()
        # add grade to grade dataframe
        grades.loc[len(grades)] = ['A', 0]
        # update grades.csv
        grades.to_csv('grades.csv', index=False)

    return


# test case b
def test_batch_output_shape(get_batch, X_TRAIN, Y_TRAIN, VOCAB_SIZE):
    try:
        # make sure we are using the global variable
        global POINTS_B
        POINTS_B = 0

        ''' test case b'''
        # test batch output shape
        batch_index = 0
        batch_size = 1000
        x, y = get_batch(batch_index, batch_size, X_TRAIN, Y_TRAIN)
        if list(x.shape) == [batch_size, VOCAB_SIZE]:
            print('Output shape looks good!')
            print('Test B: 5/5')
            POINTS_B = 5
            # add grade to dataframe
            grades.loc[len(grades)] = ['B', POINTS_B]
            # save grades
            grades.to_csv('grades.csv', index=False)
        else:
            print('Seems to be an issue with output shape in your get_batch() function...')
            print('Test B: 0/5')
            # add grade to grade dataframe
            grades.loc[len(grades)] = ['B', 0]
            # update grades.csv
            grades.to_csv('grades.csv', index=False)

    except Exception as e:
        print('Test B: 0/5')
        print('Error during execution of Test B:')
        # print traceback
        traceback.print_exc()
        # add grade to grade dataframe
        grades.loc[len(grades)] = ['B', 0]
        # update grades.csv
        grades.to_csv('grades.csv', index=False)

    return


# test case c and d
def check_bow_architecture(model):
    check_bow_architecture_c(model)
    check_bow_architecture_d(model)
    return


def check_bow_architecture_c(model):
    try:
        # ensure we are using the global variable
        global POINTS_C
        POINTS_C = 0

        ''' test case c '''
        # Check if the model has only one layer
        if len(list(model.children())) != 1:
            print('Model does not have the expected number of layers.')
            print('Test C: 0/5')
            # add grade to grade dataframe
            grades.loc[len(grades)] = ['C', 0]
            # update grades.csv
            grades.to_csv('grades.csv', index=False)
        else:
            print("Model has the expected number of layers.")
            print('Test C: 5/5')
            POINTS_C = 5
            # add grade to dataframe
            grades.loc[len(grades)] = ['C', POINTS_C]
            # save grades
            grades.to_csv('grades.csv', index=False)

    except Exception as e:
        print('Test C: 0/5')
        print('Error during execution of Test C:')
        # print traceback
        traceback.print_exc()
        # add grade to grade dataframe
        grades.loc[len(grades)] = ['C', 0]
        # update grades.csv
        grades.to_csv('grades.csv', index=False)

    return


def check_bow_architecture_d(model):
    try:
        # ensure we are using the global variable
        global POINTS_D
        POINTS_D = 0

        ''' test case d '''
        # Check if the first layer is a Linear layer
        if not isinstance(list(model.children())[0], nn.Linear):
            print('First layer is not a Linear layer.')
            print('Test D: 0/5')
            # add grade to grade dataframe
            grades.loc[len(grades)] = ['D', 0]
            # update grades.csv
            grades.to_csv('grades.csv', index=False)
        else:
            print("First layer is a Linear layer.")
            print('Test D: 5/5')
            POINTS_D = 5
            # add grade to dataframe
            grades.loc[len(grades)] = ['D', POINTS_D]
            # save grades
            grades.to_csv('grades.csv', index=False)

    except Exception as e:
        print('Test D: 0/5')
        print('Error during execution of Test D:')
        # print traceback
        traceback.print_exc()
        # add grade to grade dataframe
        grades.loc[len(grades)] = ['D', 0]
        # update grades.csv
        grades.to_csv('grades.csv', index=False)

    return


# test case e
def test_forward_pass_shape(X_TRAIN, Y_TRAIN, bow_nn_model):
    try:
        # make sure we are using the global variable
        global POINTS_E
        POINTS_E = 0

        ''' test case e'''
        batch_index = 0
        batch_size = 1000
        x, y = _main.get_batch(batch_index, batch_size, X_TRAIN, Y_TRAIN)
        y_hat = bow_nn_model(x)
        y_hat = y_hat.reshape(-1)

        # test forward pass shape
        if y_hat.shape[0] == batch_size:
            print("Forward pass output shape looks good!")
            print('Test E: 5/5')
            POINTS_E = 5
            # add grade to dataframe
            grades.loc[len(grades)] = ['E', POINTS_E]
            # save grades
            grades.to_csv('grades.csv', index=False)
            # visualize the output
            #_main.make_dot(y_hat, dict(bow_nn_model.named_parameters()))
        else:
            print("Seems to be an issue with output shape in your model...")
            print('Test E: 0/5')
            # add grade to dataframe
            grades.loc[len(grades)] = ['E', 0]
            # save grades
            grades.to_csv('grades.csv', index=False)

    except Exception as e:
        print('Test E: 0/5')
        print('Error during execution of Test A:')
        # print traceback
        traceback.print_exc()
        # add grade to grade dataframe
        grades.loc[len(grades)] = ['E', 0]
        # update grades.csv
        grades.to_csv('grades.csv', index=False)

    return


# test case f
def test_model_accuracy_lr(TEST, bow_nn_model):
    try:
        # ensure we are using the global variable
        global POINTS_F
        POINTS_F = 0

        ''' test case f '''
        bow_nn_predictions = []
        with torch.no_grad():
            for index, row in TEST.iterrows():
                bow_vec = _main.make_bow(row['review'])
                probs = bow_nn_model(bow_vec.float())
                pred = 1 if probs[0][0] > 0.5 else 0
                bow_nn_predictions.append(pred)
        accuracy = round((bow_nn_predictions == TEST['label']).mean(), 1)
        if accuracy >= .78:
            print(classification_report(TEST['label'], bow_nn_predictions))
            print('Accuracy: ', accuracy)
            print('Test F: 20/20')
            # add grade to dataframe
            grades.loc[len(grades)] = ['F', 20]
            # save grades
            grades.to_csv('grades.csv', index=False)
        else:
            print('Accuracy: ', accuracy)
            print('Test F: 0/20')
            # add grade to dataframe
            grades.loc[len(grades)] = ['F', 0]
            # save grades
            grades.to_csv('grades.csv', index=False)

    except Exception as e:
        print('Test F: 0/20')
        print('Error during execution of Test F:')
        # print traceback
        traceback.print_exc()
        # add grade to grade dataframe
        grades.loc[len(grades)] = ['F', 0]
        # update grades.csv
        grades.to_csv('grades.csv', index=False)

    return


# test case g
def test_model_accuracy_mr(X_NEWS_TEST, Y_NEWS_TEST, multibow_model):
    try:
        # ensure we are using the global variable
        global POINTS_G
        POINTS_G = 0

        ''' test case g '''
        multibow_model.eval()
        with torch.no_grad():
            text_vec = torch.tensor(X_NEWS_TEST, dtype=torch.float)
            probs = multibow_model(text_vec)
            pred = probs.argmax(dim=1)
        targets = torch.tensor(Y_NEWS_TEST)
        accuracy = (pred == targets).float().mean().item()
        if accuracy >= .78:
            print('Accuracy: ', accuracy)
            print('Test G: 40/40')
            # add grade to dataframe
            grades.loc[len(grades)] = ['G', 40]
            # save grades
            grades.to_csv('grades.csv', index=False)
        else:
            print('Accuracy: ', accuracy)
            print('Test G: 0/40')
            # add grade to dataframe
            grades.loc[len(grades)] = ['G', 0]
            # save grades
            grades.to_csv('grades.csv', index=False)

    except Exception as e:
        print('Test G: 0/40')
        print('Error during execution of Test G:')
        # print traceback
        traceback.print_exc()
        # add grade to grade dataframe
        grades.loc[len(grades)] = ['G', 0]
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
