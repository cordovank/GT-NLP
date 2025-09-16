import numpy as np
import pandas as pd
import sys
import torch
import torch.nn as nn  # pytorch neural network sub-library
import torch.nn.functional as F
import traceback
import warnings

from datasets import load_dataset

# ignore all warnings
warnings.filterwarnings('ignore')

_main = sys.modules['__main__']

# create grade data frame
grades = pd.DataFrame(columns=['question', 'score'])

# create empty grades file
grades.to_csv('grades.csv', index=False)


# test case a
def test_glove_analogy(model, glove_analogy_fn, k=5):

    try:

        global POINTS_A
        POINTS_A = 0

        def _k_expected_in_results(results, expected_results, k=5):
            # Check if at least k results in the expected results are in results
            intersection = set(results).intersection(expected_results)
            return len(intersection) >= k

        results_1 = glove_analogy_fn(model, 'driver', 'car', 'pilot', k=10)
        expected_results_1 = ['aircraft', 'plane', 'air', 'jet', 'helicopter', 'flight', 'planes', 'pilot', 'airplane',
                              'car']
        passed_1 = _k_expected_in_results(results_1, expected_results_1, k=5)

        results_2 = glove_analogy_fn(model, 'obese', 'fat', 'slender', k=10)
        expected_results_2 = ['fat', 'slender', 'thin', 'thick', 'protein', 'onion', 'soft', 'tall', 'butter', 'tail']
        passed_2 = _k_expected_in_results(results_2, expected_results_2, k=5)

        results_3 = glove_analogy_fn(model, 'pound', 'kilogram', 'quart', k=10)
        expected_results_3 = ['quart', 'liter', 'kilogram', 'tael', 'saucepan', 'micrograms', 'litres', 'tablespoon',
                              'kiloton', 'milliliters']
        passed_3 = _k_expected_in_results(results_3, expected_results_3, k=3)

        # print results
        if passed_1 & passed_2 & passed_3:
            print('Test passed!')
            POINTS_A = 5
            grades.loc[len(grades)] = ['A', POINTS_A]
            print('Test A: {}/5'.format(POINTS_A))
        else:
            print('Test failed - check your glove_analogy implementation.')
            POINTS_A = 0
            grades.loc[len(grades)] = ['A', POINTS_A]
            print('Test A: {}/5'.format(POINTS_A))

        # update grades
        grades.to_csv('grades.csv', index=False)

    except Exception as e:
        print('Error during execution of test_glove_analogy:')
        # print traceback
        traceback.print_exc()
        # print results
        print('Test A: 0/5')
        # add grade to grade dataframe
        grades.loc[len(grades)] = ['A', 0]
        # update grades.csv
        grades.to_csv('grades.csv', index=False)

    return


# test case b
def unit_test_embed_dataset(model, embed_dataset_fn):

    try:

        global POINTS_B
        POINTS_B = 0

        test_words = ['yes', 'the', 'some', 'leptodactylidae']
        test_df = pd.DataFrame(test_words, columns=['text'])
        test_embedded_data = embed_dataset_fn(test_df, model)
        # Hard code Glove embeddings
        test_result = np.array([[-2.50654994e-04, 3.31779988e-03, 4.21574991e-03,
                                 -3.76474997e-03, -2.14240002e-03, 8.05000018e-04,
                                 -1.83650001e-04, 3.73885006e-04, 1.76254997e-03,
                                 -1.42780005e-03, 1.03205000e-03, -2.15509994e-04,
                                 -1.89890002e-03, -2.62935006e-04, 1.46555004e-03,
                                 -2.29394995e-03, -2.36815005e-03, 1.01524999e-03,
                                 -2.25925003e-03, 4.08645021e-03, 8.47749994e-04,
                                 2.76939990e-03, -2.83335010e-03, -3.40200006e-03,
                                 5.08100027e-04, 2.16529984e-03, -2.05175005e-04,
                                 -1.65979995e-03, 1.99274998e-03, -2.14620004e-03,
                                 8.01850052e-04, 3.35274986e-03, 2.83874990e-03,
                                 -1.83975004e-04, 1.83809991e-03, 1.02810003e-03,
                                 2.41860002e-03, -1.18284997e-05, 1.54480000e-03,
                                 -2.79875007e-03, -4.69464983e-04, 2.26934993e-04,
                                 6.50150003e-04, -2.38914997e-03, -3.13775009e-03,
                                 -4.61085001e-03, 1.70394997e-04, -1.18224998e-03,
                                 -3.98799963e-03, -5.06850006e-03, 5.94199984e-04,
                                 2.56730011e-04, 9.16199991e-04, 2.21389998e-03,
                                 -2.36569997e-03, -7.76549987e-03, 2.83814990e-03,
                                 3.18984990e-03, 3.21585010e-03, 1.07394997e-03,
                                 -1.00289995e-03, 1.44830009e-03, -3.33710015e-03,
                                 -3.87149979e-03, 3.54914996e-03, 1.29859999e-03,
                                 2.29600002e-03, 1.35890010e-03, -2.86655012e-03,
                                 -3.62120016e-04, 1.84235012e-03, 2.31774990e-03,
                                 -9.21399987e-05, -2.09759991e-03, 6.70699985e-04,
                                 1.66069996e-03, -1.66494996e-04, 5.03699994e-04,
                                 -2.19304999e-03, -1.88930007e-03, 1.67099992e-03,
                                 -3.31889978e-03, 1.04730006e-03, -1.80860003e-03,
                                 -6.33800030e-03, -2.12395005e-03, 1.73329993e-03,
                                 -2.11524987e-03, -1.69974996e-03, 9.36150027e-04,
                                 -2.11750012e-04, -1.67395000e-03, 1.21655001e-03,
                                 2.88224989e-03, -2.48895003e-03, 3.61985003e-04,
                                 -1.31504994e-03, 8.24450050e-04, 2.43989998e-04,
                                 2.51795002e-03],
                                [-1.90970008e-04, -1.22435007e-03, 3.64060025e-03,
                                 -1.99805014e-03, 4.15860006e-04, 2.19765003e-04,
                                 -1.95705006e-03, 1.67200004e-03, -2.87725008e-03,
                                 4.37294977e-04, 1.43934996e-03, -3.36550002e-04,
                                 1.54530001e-03, -1.31919992e-03, -6.61550032e-04,
                                 -1.03785004e-03, 1.66975008e-03, -1.69239996e-03,
                                 -1.58714992e-03, -2.41680001e-03, 7.32000044e-04,
                                 -1.86520000e-03, 1.72884995e-03, 2.60205008e-04,
                                 2.24730000e-03, -2.34854990e-03, 1.31400011e-04,
                                 -2.70774984e-03, -7.75900029e-04, -7.05349958e-04,
                                 -1.98609996e-04, 1.41385000e-03, 7.19650008e-04,
                                 1.17319997e-03, -1.55104999e-03, 4.30864980e-04,
                                 1.01985002e-03, 2.63120001e-03, 8.58199957e-04,
                                 -4.11890011e-04, -3.58935003e-03, -2.07654992e-03,
                                 1.01675000e-03, -6.38149970e-04, 2.06835009e-03,
                                 2.75935000e-03, 2.89539993e-03, -1.67385000e-03,
                                 -1.82795001e-03, -2.74284999e-03, -3.14459990e-04,
                                 1.32919999e-03, 1.51024992e-03, 4.98874998e-03,
                                 -4.02404973e-03, -1.51215009e-02, 6.26999972e-05,
                                 -1.84709998e-03, 1.10835005e-02, 3.61005007e-03,
                                 -1.24889996e-03, 4.60680015e-03, 1.72569999e-04,
                                 2.33724993e-03, 5.53950015e-03, -9.67900036e-04,
                                 -3.72874987e-04, 1.16764999e-03, -2.60310015e-04,
                                 -1.10220001e-03, 2.85810005e-04, -7.90300022e-04,
                                 -1.53989997e-03, -2.08124984e-03, 1.89860002e-03,
                                 7.50300009e-04, -2.66060000e-03, -1.02750002e-03,
                                 -6.26299996e-03, 3.58120014e-04, 3.52824992e-03,
                                 2.48720008e-03, -2.10315012e-03, 1.30740006e-03,
                                 -7.68999988e-03, -1.51115004e-03, -3.67190019e-04,
                                 -1.41560007e-03, 1.85519992e-03, -1.26085000e-03,
                                 8.10749989e-05, -8.54950049e-05, -1.94920006e-03,
                                 4.37119976e-03, -3.62844998e-03, -2.55290000e-03,
                                 -2.60140002e-03, -7.29499967e-04, 4.13900008e-03,
                                 1.35309994e-03],
                                [-7.22599973e-04, 2.80115008e-03, 1.02700002e-03,
                                 -1.33320002e-03, -1.28344994e-03, 2.62739975e-03,
                                 -2.29909993e-03, 1.09989999e-03, -6.96850038e-05,
                                 -1.21124997e-03, 9.04150016e-04, 3.56610020e-04,
                                 1.91850006e-03, -1.83204992e-03, 2.56519997e-03,
                                 -1.86230009e-03, -9.94850066e-04, -2.75780010e-04,
                                 1.03685001e-04, 3.50310002e-03, 4.08534985e-03,
                                 -4.99774993e-04, 8.49400007e-04, -1.25005003e-03,
                                 -1.34500000e-03, -3.48885008e-03, -1.79210008e-04,
                                 -2.54110014e-03, -2.97075003e-05, -1.64755003e-03,
                                 9.45599983e-04, 4.22634999e-04, -2.15804996e-03,
                                 -1.41055003e-04, -9.93449939e-04, 2.39564991e-03,
                                 3.23630025e-04, 1.48939993e-03, -1.36425009e-03,
                                 6.30100025e-04, -4.80034994e-03, -2.26724986e-03,
                                 9.43550025e-04, -2.09920015e-03, 2.30260004e-04,
                                 -1.21080002e-03, 2.39700009e-03, -1.11359998e-03,
                                 -2.79725005e-04, -2.39484990e-03, 4.51414991e-04,
                                 -2.95659993e-04, -1.24224997e-03, 5.15149999e-03,
                                 -2.15204991e-03, -1.20449997e-02, 8.74949968e-04,
                                 -3.07704974e-03, 9.12149996e-03, 3.85600002e-03,
                                 1.20315002e-03, 8.26099981e-03, -1.97139991e-04,
                                 2.55904999e-03, 3.64020001e-03, -1.50910008e-03,
                                 4.06860001e-03, 1.64320006e-03, 3.36749991e-03,
                                 -1.98585005e-03, -2.25959998e-03, -1.93715008e-04,
                                 -5.97750011e-04, -1.06295000e-03, 1.00359996e-03,
                                 1.49309996e-03, -1.54330002e-04, 9.15999990e-04,
                                 -6.51000021e-03, 9.75999981e-04, 5.96199976e-03,
                                 -1.94120000e-03, -1.65324996e-03, 7.33700017e-06,
                                 -8.62849969e-03, 5.33099985e-04, 1.59140001e-03,
                                 5.11150029e-05, -2.16079992e-03, -7.71499996e-04,
                                 1.10514998e-03, -7.36299960e-04, -2.47005001e-03,
                                 -2.09435006e-03, -5.16799977e-03, -1.72015000e-03,
                                 -3.87315010e-03, -1.81975006e-03, 4.58045024e-03,
                                 4.10474977e-03],
                                [1.98179996e-03, 5.82700013e-04, 5.91949979e-03,
                                 8.56450002e-04, -2.79859989e-03, 5.58250025e-03,
                                 -3.66110005e-03, 8.43800022e-04, -1.06499996e-03,
                                 2.33860011e-03, -1.91310002e-03, 3.29865003e-03,
                                 1.80999993e-03, 1.75624993e-03, -7.37150013e-03,
                                 2.51085008e-03, 4.54459987e-05, -2.24099983e-03,
                                 2.61450000e-03, 1.12214999e-03, -6.00750046e-03,
                                 4.16534999e-03, 2.28050002e-03, 3.05905007e-03,
                                 1.59170001e-03, 7.59749999e-03, -3.98295000e-04,
                                 2.69394997e-03, 5.69900032e-04, 2.04359996e-03,
                                 2.57629994e-03, 1.29415002e-03, 3.95134976e-03,
                                 -8.76599981e-04, 2.56635016e-03, 3.78024997e-04,
                                 -1.69534993e-03, -1.42095005e-03, -8.11849954e-04,
                                 -1.84330007e-03, 2.61734985e-03, 2.07445002e-03,
                                 -7.13799964e-04, -2.75460002e-03, 4.95565007e-04,
                                 3.86435003e-03, 1.00110006e-03, 3.96624999e-03,
                                 -3.38894990e-03, -2.56889989e-03, 4.69635008e-04,
                                 -4.98405006e-03, 3.33730015e-03, -3.50870006e-03,
                                 1.07079999e-04, 1.05105003e-03, -3.21414997e-03,
                                 -4.66444995e-03, 4.06215014e-03, -8.31449986e-04,
                                 -4.58085007e-04, -2.91949976e-03, 5.13900013e-04,
                                 -2.01710011e-03, 4.23979992e-03, -2.41774993e-04,
                                 -6.52100006e-03, -5.69400005e-03, -3.68229975e-03,
                                 -6.77900040e-04, 3.80349997e-03, 2.59070005e-03,
                                 4.39549971e-04, 8.47149990e-04, 3.03730019e-03,
                                 8.27400014e-04, 2.14935001e-03, -2.48240004e-03,
                                 -7.80700007e-04, 6.70549972e-03, -8.17449950e-03,
                                 3.48969991e-03, 3.14049982e-03, -7.06800027e-04,
                                 1.56990008e-03, 1.79600006e-03, -3.66719998e-03,
                                 -3.56954988e-03, -1.62005005e-03, 8.59849970e-05,
                                 3.46964994e-03, 2.36399984e-03, 4.40909993e-03,
                                 -2.65529985e-03, -8.42500012e-04, -1.66980003e-03,
                                 -1.43169996e-03, -3.73959984e-03, -1.13865000e-03,
                                 -2.13579996e-03]])

        if np.allclose(test_result, test_embedded_data):
            print('Test passed!')
            POINTS_B = 10
            grades.loc[len(grades)] = ['B', POINTS_B]
            print('Test B: {}/10'.format(POINTS_B))
        else:
            print('Test failed - check your embed_dataset implementation.')
            POINTS_B = 0
            grades.loc[len(grades)] = ['B', POINTS_B]
            print('Test B: {}/10'.format(POINTS_B))

        # update grades.csv
        grades.to_csv('grades.csv', index=False)

    except Exception as e:
        print('Error during execution of unit_test_embed_dataset:')
        # print traceback
        traceback.print_exc()
        # print results
        print('Test B: 0/5')
        # add grade to grade dataframe
        grades.loc[len(grades)] = ['B', 0]
        # update grades.csv
        grades.to_csv('grades.csv', index=False)

    return


# test case c
def unit_test_retrieve_top_k(model, embed_dataset_fn, retrieve_top_k_fn, k=10):

    try:

        global POINTS_C
        POINTS_C = 0

        unshuffled_wiki_data_train = load_dataset("wikitext", 'wikitext-2-v1', split="train")
        unshuffled_wiki_train = pd.DataFrame(unshuffled_wiki_data_train)
        unshuffled_embedded_data = embed_dataset_fn(unshuffled_wiki_train, model)

        def _indices_in_top_k(word, indices, embedded_data, k=10):
            output_indices = retrieve_top_k_fn(word, model, embedded_data, k=k)
            return all(x in output_indices for x in indices)

        word_1 = 'mars'
        indices_1 = [31433, 11468, 16496, 31354, 11450]
        passed_test_1 = _indices_in_top_k(word_1, indices_1, embedded_data=unshuffled_embedded_data, k=k)

        word_2 = 'gallop'
        indices_2 = [29385, 19001, 29387, 19959, 19006]
        passed_test_2 = _indices_in_top_k(word_2, indices_2, embedded_data=unshuffled_embedded_data, k=k)

        word_3 = 'extravagant'
        indices_3 = [18802, 742, 29266, 28611, 20494]
        passed_test_3 = _indices_in_top_k(word_3, indices_3, embedded_data=unshuffled_embedded_data, k=k)

        if passed_test_1 & passed_test_2 & passed_test_3:
            print('Test passed!')
            POINTS_C = 5
            grades.loc[len(grades)] = ['C', POINTS_C]
            print('Test C: {}/5'.format(POINTS_C))
        else:
            print('Test failed - check your retrieve_top_k implementation.')
            POINTS_C = 0
            grades.loc[len(grades)] = ['C', POINTS_C]
            print('Test C: {}/5'.format(POINTS_C))

        # update grades.csv
        grades.to_csv('grades.csv', index=False)

    except Exception as e:
        print('Error during execution of unit_test_retrieve_top_k:')
        # print traceback
        traceback.print_exc()
        # print results
        print('Test C: 0/5')
        # add grade to grade dataframe
        grades.loc[len(grades)] = ['C', 0]
        # update grades.csv
        grades.to_csv('grades.csv', index=False)
        return

    return


# test case d
def check_data_size_d(dataframe, window, xy_data, vocab, max_length, tokenizer_fn):

    try:

        global POINTS_D
        POINTS_D = 0

        df = dataframe['text'].apply(lambda x: len(tokenizer_fn(x)))
        cropped = (window * 2) + 1
        mask = df >= cropped
        expected = (df[mask].apply(lambda x: min(x, max_length)) - cropped + 1).sum()
        difference = abs(expected - len(xy_data))
        print("expected data points", expected)
        print("actual data points", len(xy_data))
        print("difference", difference)
        print()

        text_df = dataframe['text'].apply(lambda x: tokenizer_fn(x))
        all_words = []
        for row in text_df:
            all_words.extend(row)
        all_words = list(set(all_words))
        print('least vocab size', len(all_words))
        print('actual vocab size', vocab.num_words())
        print()

        passed_test = (difference == 0) & (vocab.num_words() >= len(all_words))

        if passed_test:
            print('Test passed!')
            POINTS_D = 10
            grades.loc[len(grades)] = ['D', POINTS_D]
            print('Test D: {}/10'.format(POINTS_D))
        else:
            print('Test failed - check your data size implementation.')
            POINTS_D = 0
            grades.loc[len(grades)] = ['D', POINTS_D]
            print('Test D: {}/10'.format(POINTS_D))

        # update grades.csv
        grades.to_csv('grades.csv', index=False)

    except Exception as e:
        print('Error during execution of check_data_size:')
        # print traceback
        traceback.print_exc()
        # print results
        print('Test D: 0/10')
        # add grade to grade dataframe
        grades.loc[len(grades)] = ['D', 0]
        # update grades.csv
        grades.to_csv('grades.csv', index=False)

    return


# test case e
def unit_test_get_batch(data, window, batch_size, get_batch_fn):

    try:

        global POINTS_E
        POINTS_E = 0

        x, y = get_batch_fn(data, 0, batch_size)

        if x.size()[0] == batch_size and x.size()[1] == window * 2 and y.size()[0] == batch_size:
            print('Test passed!')
            POINTS_E = 10
            grades.loc[len(grades)] = ['E', POINTS_E]
            print('Test E: {}/10'.format(POINTS_E))
        else:
            print('Test failed - check your get_batch implementation.')
            POINTS_E = 0
            grades.loc[len(grades)] = ['E', POINTS_E]
            print('Test E: {}/10'.format(POINTS_E))

        # update grades.csv
        grades.to_csv('grades.csv', index=False)

    except Exception as e:
        print('Error during execution of unit_test_get_batch:')
        # print traceback
        traceback.print_exc()
        # print results
        print('Test E: 0/10')
        # add grade to grade dataframe
        grades.loc[len(grades)] = ['E', 0]
        # update grades.csv
        grades.to_csv('grades.csv', index=False)

    return


# test case f
def test_cbow_structure(model):

    try:

        global POINTS_F
        POINTS_F = 0

        if len(list(model.children())) == 2:
            print('Your model has two layers as expected!')
            POINTS_F = 5
        else:
            print('Model does not have the expected number of layers.')
            POINTS_F = 0

        if [type(x) for x in list(model.children())] == [torch.nn.modules.sparse.Embedding,
                                                         torch.nn.modules.linear.Linear]:
            print('Your layers orders are as expected!')
            POINTS_F = POINTS_F + 5
            grades.loc[len(grades)] = ['F', POINTS_F]
            print('Test F: {}/10'.format(POINTS_F))
        else:
            print('Model does not have the expected layer orders.')
            POINTS_F = POINTS_F + 0
            grades.loc[len(grades)] = ['F', POINTS_F]
            print('Test F: {}/10'.format(POINTS_F))

        # update grades.csv
        grades.to_csv('grades.csv', index=False)

    except Exception as e:
        print('Error during execution of test_cbow_structure:')
        # print traceback
        traceback.print_exc()
        # print results
        print('Test F: 0/10')
        # add grade to grade dataframe
        grades.loc[len(grades)] = ['F', 0]
        # update grades.csv
        grades.to_csv('grades.csv', index=False)

    return


# test case g
def test_cbow_performance(model, data, batch_size, get_batch_fn):

    try:

        global POINTS_G
        POINTS_G = 0

        num_correct = 0.
        for i in range(len(data) // batch_size):
            x, y = get_batch_fn(data, i, batch_size)
            y_hat = model(x)
            y_hat = torch.topk(y_hat, 10, dim=1).indices
            num_correct += ((y_hat - y.unsqueeze(dim=1)) == 0).any(dim=1).sum()
        accuracy = num_correct / (len(data) // batch_size * batch_size)

        if accuracy >= 0.3:
            print('Test passed! Accuracy = {}/20'.format(accuracy))
            POINTS_G = 20
            grades.loc[len(grades)] = ['G', POINTS_G]
            print('Test G: {}/20'.format(POINTS_G))
        else:
            print('Test failed! Accuracy = {}<0.3'.format(accuracy))
            POINTS_G = 0
            grades.loc[len(grades)] = ['G', POINTS_G]
            print('Test G: {}/20'.format(POINTS_G))

        # update grades.csv
        grades.to_csv('grades.csv', index=False)

    except Exception as e:
        print('Error during execution of test_cbow_performance:')
        # print traceback
        traceback.print_exc()
        # print results
        print('Test G: 0/20')
        # add grade to grade dataframe
        grades.loc[len(grades)] = ['G', 0]
        # update grades.csv
        grades.to_csv('grades.csv', index=False)

    return


# test case h
def check_data_size_h(dataframe, window, xy_data, vocab, max_length, tokenizer_fn):

    try:

        global POINTS_H
        POINTS_H = 0

        df = dataframe['text'].apply(lambda x: len(tokenizer_fn(x)))
        cropped = (window * 2) + 1
        mask = df >= cropped
        expected = (df[mask].apply(lambda x: min(x, max_length)) - cropped + 1).sum()
        difference = abs(expected - len(xy_data))
        print("expected data points", expected)
        print("actual data points", len(xy_data))
        print("difference", difference)
        print()

        text_df = dataframe['text'].apply(lambda x: tokenizer_fn(x))
        all_words = []
        for row in text_df:
            all_words.extend(row)
        all_words = list(set(all_words))
        print('least vocab size', len(all_words))
        print('actual vocab size', vocab.num_words())
        print()

        passed_test = (difference == 0) & (vocab.num_words() >= len(all_words))

        if passed_test:
            print('Test passed!')
            POINTS_H = 5
            grades.loc[len(grades)] = ['H', POINTS_H]
            print('Test H: {}/5'.format(POINTS_H))
        else:
            print('Test failed - check your data size implementation.')
            POINTS_H = 0
            grades.loc[len(grades)] = ['H', POINTS_H]
            print('Test H: {}/5'.format(POINTS_H))

        # update grades.csv
        grades.to_csv('grades.csv', index=False)

    except Exception as e:
        print('Error during execution of check_data_size:')
        # print traceback
        traceback.print_exc()
        # print results
        print('Test H: 0/5')
        # add grade to grade dataframe
        grades.loc[len(grades)] = ['H', 0]
        # update grades.csv
        grades.to_csv('grades.csv', index=False)

    return


# test case i
def test_skipgram_structure(model):

    try:

        global POINTS_I
        POINTS_I = 0

        if len(list(model.children())) == 2:
            print('Your model has two layers as expected!')
            layer_test_passed = True
        else:
            print('Model does not have the expected number of layers.')
            layer_test_passed = False

        if [type(x) for x in list(model.children())] == [torch.nn.modules.sparse.Embedding,
                                                         torch.nn.modules.linear.Linear]:
            print('Your layers orders are as expected!')
            order_test_passed = True
        else:
            print('Model does not have the expected layer orders.')
            order_test_passed = False

        if layer_test_passed & order_test_passed:
            print('Test passed!')
            POINTS_I = 5
            grades.loc[len(grades)] = ['I', POINTS_I]
            print('Test I: {}/5'.format(POINTS_I))
        else:
            print('Test failed - check your model structure.')
            POINTS_I = 0
            grades.loc[len(grades)] = ['I', POINTS_I]
            print('Test I: {}/5'.format(POINTS_I))

        # update grades.csv
        grades.to_csv('grades.csv', index=False)

    except Exception as e:
        print('Error during execution of test_skipgram_structure:')
        # print traceback
        traceback.print_exc()
        # print results
        print('Test I: 0/5')
        # add grade to grade dataframe
        grades.loc[len(grades)] = ['I', 0]
        # update grades.csv
        grades.to_csv('grades.csv', index=False)

    return


# test case j
def test_skip_performance(model, data, batch_size, get_batch_fn):

    try:

        global POINTS_J
        POINTS_J = 0

        num_correct = 0.
        for i in range(len(data) // batch_size):
            x, y = get_batch_fn(data, i, batch_size)
            y_hat = model(x)
            y_hat = torch.argmax(y_hat, axis=1)
            y_hat = y_hat.unsqueeze(dim=1)
            num_correct += ((y - y_hat) == 0).any(dim=1).sum().item()
        accuracy = num_correct / (len(data) // batch_size * batch_size)

        if accuracy >= 0.3:
            print('Test passed! Accuracy = {}/1'.format(accuracy))
            POINTS_J = 20
            grades.loc[len(grades)] = ['J', POINTS_J]
            print('Test J: {}/20'.format(POINTS_J))
        else:
            print('Test failed! Accuracy = {}/1'.format(accuracy))
            POINTS_J = 0
            grades.loc[len(grades)] = ['J', POINTS_J]
            print('Test J: {}/20'.format(POINTS_J))

        # update grades.csv
        grades.to_csv('grades.csv', index=False)

    except Exception as e:
        print('Error during execution of test_skip_performance:')
        # print traceback
        traceback.print_exc()
        # print results
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
