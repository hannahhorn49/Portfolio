##############################################
# Programmer: Hannah Horn
# Class: CPSC 322-01 Fall 2024
# Programming Assignment #7
# 11/20/24
# I did not attempt the bonus
# Description: This program contains the tests for the
# various classifer implementations
##############################################


import numpy as np
from mysklearn.myclassifiers import MyNaiveBayesClassifier
from mysklearn.myclassifiers import MySimpleLinearRegressionClassifier
from mysklearn.myclassifiers import MyKNeighborsClassifier
from mysklearn.myclassifiers import MySimpleLinearRegressor
from mysklearn.myclassifiers import MyDummyClassifier
from mysklearn.myclassifiers import MyDecisionTreeClassifier
from mysklearn import myutils


def test_simple_linear_regression_classifier_fit():
    
    # step 1: call function to generate training data
    X_train, y_train = myutils.generate_training_data(seed = 0, num_samples = 150, slope = 2, intercept = 0, noise_std = 10)

    # step 2: discretize y_train result using function
    discretize_result = []
    for value in y_train:
        new_value = myutils.discretizer(value)
        discretize_result.append(new_value)

    # step 3: create instance of model and use fit method
    model = MySimpleLinearRegressionClassifier(discretizer = myutils.discretizer)
    model.fit(X_train, y_train)

    # step 4: perform desk check using MySimpleLinearRegressor calculations
    calculations = MySimpleLinearRegressor()
    expected_slope, expected_intercept = calculations.compute_slope_intercept(X_train, y_train)

    print(f"Model slope: {model.regressor.slope}, Expected slope: {expected_slope}")
    print(f"Model intercept: {model.regressor.intercept}, Expected intercept: {expected_intercept}")

    # final step: assert that model's computed slope and intercept match expected
    assert np.isclose(model.regressor.slope, expected_slope)
    assert np.isclose(model.regressor.intercept, expected_intercept)
    print("after assert statements in test_simple_linear_regression_classifier_fit: passed!")

def test_simple_linear_regression_classifier_predict():
    # test case #1: slope = 3 and intercept = 50
    X_train, y_train = myutils.generate_training_data(seed = 0, num_samples = 150, slope = 3, intercept = 50, noise_std = 10)
    discretize_result = []
    for value in y_train:
        new_value = myutils.discretizer(value)
        discretize_result.append(new_value)

    # create instance of model and use fit method
    model = MySimpleLinearRegressionClassifier(discretizer = myutils.discretizer)
    model.fit(X_train, y_train)

    # test predictions for test case 1
    X_test_case1 = [[0], [50], [100]]
    expected_case1 = ["low", "high", "high"]

    # call predict method of model and use assert to compare
    model_predictions1 = model.predict(X_test_case1)
    assert model_predictions1 == expected_case1

    # test case #2: slope = 2 and intercept = 5
    X_train, y_train = myutils.generate_training_data(seed = 0, num_samples = 150, slope = 2, intercept = 5, noise_std = 10)
    discretize_result = []
    for value in y_train:
        new_value = myutils.discretizer(value)
        discretize_result.append(new_value)

    # create instance of model and use fit method 
    model = MySimpleLinearRegressionClassifier(discretizer = myutils.discretizer)
    model.fit(X_train, y_train)

    # test predictions for test case 1
    X_test_case2 = [[0], [25], [30]]
    expected_case2 = ["low", "low", "low"]

    # call predict method of model and use assert to compare
    model_predictions2 = model.predict(X_test_case2)
    assert model_predictions2 == expected_case2

    print(f"Predicted values using model for test case 1: {model_predictions1}")
    print(f"Predicted values using for test case 2: {model_predictions2}")
    print("after assert statements in test_simple_linear_regression_classifier_predict: passed!")

def test_kneighbors_classifier_kneighbors():
    # Test A (4 instance)
    X_train_class_example1 = [[1, 1], [1, 0], [0.33, 0], [0, 0]]
    y_train_class_example1 = ["bad", "bad", "good", "good"]

    # create instance of classifier and fit model
    model = MyKNeighborsClassifier() 
    model.fit(X_train_class_example1, y_train_class_example1)

    # set up the "unseen" test instance
    X_test1 = [[0.33,1]]

    # what the top k (3) neighbors are and their index
    k_top_distances1 = [[0, 0.67], [2, 1.0], [3, 1.05]]

    # use kneighbors method to assert the results
    index, distance = model.kneighbors(X_test1) 
    print(f"DEBUG: distances = {distance}, index = {index}")

    print("DEBUG: this is before assert index TestA for kneighbors")
    assert index[0] == [0,2,3]
    print("DEBUG this is after assert index TestA for kneighbors")

    rounded_distance_list = []
    for value in distance[0]:
        result = round(value, 2)
        rounded_distance_list.append(result)
    
    # Print the rounded distances
    print(f"Rounded distances: {rounded_distance_list}")

    assert np.allclose(rounded_distance_list, [0.67, 1.0, 1.05])
    print("DEBUG: this is after assert TestA for kneighbors, test passed! ")

    # Test B (8 instance)
    X_train_class_example2 = [
        [3, 2],
        [6, 6],
        [4, 1],
        [4, 4],
        [1, 2],
        [2, 0],
        [0, 3],
        [1, 6]]

    y_train_class_example2 = ["no", "yes", "no", "no", "yes", "no", "yes", "yes"]

    # create instance of classifier and fit model
    model = MyKNeighborsClassifier()
    model.fit(X_train_class_example2, y_train_class_example2)

    # set up the "unseen" test instance
    X_test2 = [[2,3]]

    # what the top k (3) neighbors are and their index
    k_top_distances2 = [(0, 1.4142135623730951),(4, 1.4142135623730951), (6, 2.0)]

    # use kneighbors method to assert the results
    index, distance = model.kneighbors(X_test2) 
    print(f"DEBUG: distances = {distance}, index = {index}")

    print("DEBUG: this is before assert TestB for kneighbors")
    assert index[0] == [0,4,6]
    assert np.allclose(distance[0], [1.4142135623730951, 1.4142135623730951, 2.0])
    print("DEBUG: this is after assert TestB for kneighbors, test passed! ")

    # Test C (Bramer)
    X_train_bramer_example = [
    [0.8, 6.3],
    [1.4, 8.1],
    [2.1, 7.4],
    [2.6, 14.3],
    [6.8, 12.6],
    [8.8, 9.8],
    [9.2, 11.6],
    [10.8, 9.6],
    [11.8, 9.9],
    [12.4, 6.5],
    [12.8, 1.1],
    [14.0, 19.9],
    [14.2, 18.5],
    [15.6, 17.4],
    [15.8, 12.2],
    [16.6, 6.7],
    [17.4, 4.5],
    [18.2, 6.9],
    [19.0, 3.4],
    [19.6, 11.1]]

    y_train_bramer_example = ["-", "-", "-", "+", "-", "+", "-", "+", "+", "+", "-", "-", "-",\
            "-", "-", "+", "+", "+", "-", "+"]
    
    # create instance of classifier and fit model
    model = MyKNeighborsClassifier()
    model.fit(X_train_bramer_example, y_train_bramer_example)

    # set up the "unseen" test instance
    X_test3 = [[9.1, 11.0]]

    # what the top k (3) neighbors are and their index
    k_top_distances3 = [
        (6, 0.608),
        (5, 1.237),
        (7, 2.202),
        (4, 2.802),
        (8, 2.915)
    ]

    # use kneighbors method to assert the results
    index, distance = model.kneighbors(X_test3, n_neighbors = 5) 
    print(f"DEBUG: distances = {distance}, index = {index}")

    print("DEBUG: this is before assert index TestC for kneighbors")
    assert index[0] == [6,5,7,4,8]
    print("DEBUG: this is after assert index TestC for kneighbors")

    rounded_distance_list = []
    for value in distance[0]:
        result = round(value, 2)
        rounded_distance_list.append(result)

    print(f"Rounded distances: {rounded_distance_list}")

    assert np.allclose(rounded_distance_list, [0.61, 1.24, 2.20, 2.80, 2.92])
    print("DEBUG: this is after assert TestC for kneighbors, test passed! ")


def test_kneighbors_classifier_predict():
    # Test A (4 instance)
    X_train_class_example1 = [[1, 1], [1, 0], [0.33, 0], [0, 0]]
    y_train_class_example1 = ["bad", "bad", "good", "good"]

    # create instance of classifier and fit model
    model = MyKNeighborsClassifier() 
    model.fit(X_train_class_example1, y_train_class_example1)

    # set up the "unseen" test instance and call kneighbors method
    X_test1 = [[0.33,1]]
    model.kneighbors(X_test1)

    predicted_class_labels = [("bad"), ("good"), ("good")]
    expected_class = ["good"]

    predicted_class = model.predict(X_test1)
    # Debug statements to print expected and predicted class
    print(f"DEBUG: Expected class: {expected_class}")
    print(f"DEBUG: Predicted class: {predicted_class}")

    print("DEBUG: this is before assert TestA for predict class")
    assert expected_class == predicted_class
    print("DEBUG: this is after assert TestA for predict class,testA passed!")

    # Test B (8 instance)
    X_train_class_example2 = [
        [3, 2],
        [6, 6],
        [4, 1],
        [4, 4],
        [1, 2],
        [2, 0],
        [0, 3],
        [1, 6]]

    y_train_class_example2 = ["no", "yes", "no", "no", "yes", "no", "yes", "yes"]

    # create instance of classifier and fit model
    model = MyKNeighborsClassifier()
    model.fit(X_train_class_example2, y_train_class_example2)

    # set up the "unseen" test instance
    X_test2 = [[2,3]]
    model.kneighbors(X_test2)

    predicted_class_labels2 = [("no"), ("yes"), ("yes")]
    expected_class2 = ["yes"]

    predicted_class2 = model.predict(X_test2)
    # Debug statements to print expected and predicted class
    print(f"DEBUG: Expected class: {expected_class2}")
    print(f"DEBUG: Predicted class: {predicted_class2}")

    print("DEBUG: this is before assert TestB for predict class")
    assert expected_class2 == predicted_class2
    print("DEBUG: this is after assert TestB for predict class, test passed!")

    # Test C (Bramer)
    X_train_bramer_example = [
    [0.8, 6.3],
    [1.4, 8.1],
    [2.1, 7.4],
    [2.6, 14.3],
    [6.8, 12.6],
    [8.8, 9.8],
    [9.2, 11.6],
    [10.8, 9.6],
    [11.8, 9.9],
    [12.4, 6.5],
    [12.8, 1.1],
    [14.0, 19.9],
    [14.2, 18.5],
    [15.6, 17.4],
    [15.8, 12.2],
    [16.6, 6.7],
    [17.4, 4.5],
    [18.2, 6.9],
    [19.0, 3.4],
    [19.6, 11.1]]

    y_train_bramer_example = ["-", "-", "-", "+", "-", "+", "-", "+", "+", "+", "-", "-", "-",\
            "-", "-", "+", "+", "+", "-", "+"]
      
    # create instance of classifier and fit model
    model = MyKNeighborsClassifier()
    model.fit(X_train_bramer_example, y_train_bramer_example)

    # set up the "unseen" test instance
    X_test3 = [[9.1,11.0]]
    model.kneighbors(X_test3, n_neighbors = 5)

    expected_class_label3 = ["+"]

    predicted_class3 = model.predict(X_test3)
    # Debug statements to print expected and predicted class
    print(f"DEBUG: Expected class: {expected_class_label3}")
    print(f"DEBUG: Predicted class: {predicted_class3}")

    print("DEBUG: this is before assert TestC for predict class")
    assert expected_class_label3 == predicted_class3
    print("DEBUG: this is after assert TestC for predict class, test passed!")


def test_dummy_classifier_fit():
    # case A: "yes" is more frequent than "no"
    y_train_a = list(np.random.choice(["yes", "no"], 100, replace=True, p=[0.7, 0.3]))
    X_train_a = np.random.rand(100,5)

    dummy_classifier_a = MyDummyClassifier()
    dummy_classifier_a.fit(X_train_a, y_train_a)
    
    print("DEBUG: before assert a statement in test_dummy_classifier_fit")
    assert dummy_classifier_a.most_common_label == "yes"
    print("DEBUG: after assert a statement in test_dummy_classifier_fit, test passed!!")

    # case B: "no" should be more frequent than "yes" and "maybe"
    y_train_b = list(np.random.choice(["yes", "no", "maybe"], 100, replace=True, p=[0.2, 0.6, 0.2]))
    X_train_b = np.random.rand(100,5)

    dummy_classifier_b = MyDummyClassifier()
    dummy_classifier_b.fit(X_train_b, y_train_b)

    print("DEBUG: before assert b statement in test_dummy_classifier_fit")
    assert dummy_classifier_b.most_common_label == "no"
    print("DEBUG: after assert b statement in test_dummy_classifier_fit, test passed!!")

    # case C:
    y_train_c = list(np.random.choice(["yes", "yea", "nope", "no"], 100, replace = True, p = [0.01, 0.05, 0.7, 0.24]))
    X_train_c = np.random.rand(100,5)

    dummy_classifier_c = MyDummyClassifier()
    dummy_classifier_c.fit(X_train_c, y_train_c)

    print("DEBUG: before assert c statement in test_dummy_classifier_fit")
    assert dummy_classifier_c.most_common_label == "nope"
    print("DEBUG: after assert c statement in test_dummy_classifier_fit, test passed!!")

def test_dummy_classifier_predict():
    # set up same y_train, X_train labels for case A
    y_train_a = list(np.random.choice(["yes", "no"], 100, replace=True, p=[0.7, 0.3]))
    X_train_a = np.random.rand(100,5)

    # create instance and fit the classifier
    dummy_classifier_a = MyDummyClassifier()
    dummy_classifier_a.fit(X_train_a, y_train_a)

    # generate random test data for predict method
    X_test_a = np.random.rand(10,5)

    # call predict method on test data
    predictions_a = dummy_classifier_a.predict(X_test_a)

    # all predictions in list should have the label ("yes")
    expected_predictions_a = ["yes"] * len(X_test_a)

    # assert predictions statement
    print("DEBUG: before assert a statement in test_dummy_classifier_predict")
    assert predictions_a == expected_predictions_a
    print("DEBUG: after assert a statement in test_dummy_classifier_predict, test passed!!")
    
    # set up same y_train, X_train labels for case B
    y_train_b = list(np.random.choice(["yes", "no", "maybe"], 100, replace=True, p=[0.2, 0.6, 0.2]))
    X_train_b = np.random.rand(100,5)

    # create instance and fit the classifier
    dummy_classifier_b = MyDummyClassifier()
    dummy_classifier_b.fit(X_train_b, y_train_b)

    # generate random test data for predict method
    X_test_b = np.random.rand(10,5)

    # call predict method on test data
    predictions_b = dummy_classifier_b.predict(X_test_b)

    # all predictions in list should have the label ("no")
    expected_predictions_b = ["no"] * len(X_test_b)

    # assert predictions statement
    print("DEBUG: before assert b statement in test_dummy_classifier_predict")
    assert predictions_b == expected_predictions_b
    print("DEBUG: after assert b statement in test_dummy_classifier_predict, test passed!!")

    # set up same y_train, X_train labels for case C
    y_train_c = list(np.random.choice(["yes", "yea", "nope", "no"], 100, replace = True, p = [0.01, 0.05, 0.7, 0.24]))
    X_train_c = np.random.rand(100,5)
    
    dummy_classifier_c = MyDummyClassifier()
    dummy_classifier_c.fit(X_train_c, y_train_c)

    # generate random test data for predict method
    X_test_c = np.random.rand(10,5)

    # call predict method on test data
    predictions_c = dummy_classifier_c.predict(X_test_c)

    # all predictions in list should have the label ("no")
    expected_predictions_c = ["nope"] * len(X_test_c)

    # assert predictions statement
    print("DEBUG: before assert c statement in test_dummy_classifier_predict")
    assert predictions_c == expected_predictions_c
    print("DEBUG: after assert c statement in test_dummy_classifier_predict, test passed!!")

def test_naive_bayes_classifier_fit():
    # in-class Naive Bayes example (lab task #1)
    # header_inclass_example = ["att1", "att2"]
    X_train_inclass_example = [
        [1, 5], # yes
        [2, 6], # yes
        [1, 5], # no
        [1, 5], # no
        [1, 6], # yes
        [2, 6], # no
        [1, 5], # yes
        [1, 6] # yes
    ]
    y_train_inclass_example = ["yes", "yes", "no", "no", "yes", "no", "yes", "yes"]

    # create instance of model and fit
    model = MyNaiveBayesClassifier()
    model.fit(X_train_inclass_example, y_train_inclass_example)

    # expected priors (hard code-- desk check)
    expected_priors = {"yes": 5/8, "no": 3/8}

    # assert check against model
    for label, probability in expected_priors.items():
        model_prior = model.priors[label]
        assert model_prior == probability
    print("Test Case1 (class example), prior asserts passed!!")
    
    # expected posteriors (hard code --desk check)
    expected_posteriors = {
        "yes": {  # for the class "yes"
            0: {1: 4/5, 2: 1/5},  # for feature index 0 (att1)
            1: {5: 2/5, 6: 3/5}   # for feature index 1 (att2)
        },
        "no": {  # for the class "no"
            0: {1: 2/3, 2: 1/3},  # for feature index 0 (att1)
            1: {5: 2/3, 6: 1/3}   # for feature index 1 (att2)
        }
    }

    # assert check against model
    # outer loop through each label (yes/no)
    for label, attribute_type in expected_posteriors.items():

        # loop through each attribute index for current label (e.g. att1 and att2)
        for attribute_index, expected_probability in attribute_type.items():

            # loop through each attribute value and expected probability
            for attribute_value, expected_prob in expected_probability.items():
                model_posterior = model.posteriors[label][attribute_index][attribute_value]

                # check if model's posterior matches the expected posterior probability
                assert model_posterior == expected_prob
    print("Test Case1 (class example), posterior asserts passed!!")

    # MA7 (fake) iPhone purchases dataset
    # header_iphone = ["standing", "job_status", "credit_rating", "buys_iphone"]
    X_train_iphone = [
        [1, 3, "fair"],
        [1, 3, "excellent"],
        [2, 3, "fair"],
        [2, 2, "fair"],
        [2, 1, "fair"],
        [2, 1, "excellent"],
        [2, 1, "excellent"],
        [1, 2, "fair"],
        [1, 1, "fair"],
        [2, 2, "fair"],
        [1, 2, "excellent"],
        [2, 2, "excellent"],
        [2, 3, "fair"],
        [2, 2, "excellent"],
        [2, 3, "fair"]
    ]
    y_train_iphone = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no", "yes"]

    # create instance of model and fit
    model = MyNaiveBayesClassifier()
    model.fit(X_train_iphone, y_train_iphone)

    # expected priors (hard code-- desk check)
    expected_priors2 = {"yes": 10/15, "no": 5/15}

    # assert check against model
    for label, probability in expected_priors2.items():
        model_prior = model.priors[label]
        assert model_prior == probability
    
    print("Test Case2 (MA7 phone), prior asserts passed!!")
    
    # expected posteriors (hard code --desk check)
    expected_posteriors2 = {
        "yes": {  # for the class "yes"
            0: {1: 2/10, 2: 8/10},  # for feature index 0 ("standing")
            1: {1: 3/10, 2: 4/10, 3: 3/10},  # for feature index 1 ("job_status")
            2: {"fair": 7/10, "excellent": 3/10}  # for feature index 2 ("credit_rating")
        },
        "no": {  # for the class "no"
            0: {1: 3/5, 2: 2/5},  # for feature index 0 ("standing")
            1: {1: 1/5, 2: 2/5, 3: 2/5},  # for feature index 1 ("job_status")
            2: {"fair": 2/5, "excellent": 3/5}  # for feature index 2 ("credit_rating")
        }
    }

    # assert check against model
    # outer loop through each label (yes/no)
    for label, attribute_type in expected_posteriors2.items():

        # loop through each attribute index for current label (e.g. att1 and att2)
        for attribute_index, expected_probability in attribute_type.items():

            # loop through each attribute value and expected probability
            for attribute_value, expected_prob in expected_probability.items():
                model_posterior2 = model.posteriors[label][attribute_index][attribute_value]

                # check if model's posterior matches the expected posterior probability
                assert model_posterior2 == expected_prob
    print("Test Case2 (MA7 phone), posterior asserts passed!!")

    # Bramer train dataset example
    # header_train = ["day", "season", "wind", "rain", "class"]
    X_train_train = [
            ["weekday", "spring", "none", "none"],
            ["weekday", "winter", "none", "slight"],
            ["weekday", "winter", "none", "slight"],
            ["weekday", "winter", "high", "heavy"],
            ["saturday", "summer", "normal", "none"],
            ["weekday", "autumn", "normal", "none"],
            ["holiday", "summer", "high", "slight"],
            ["sunday", "summer", "normal", "none"],
            ["weekday", "winter", "high", "heavy"],
            ["weekday", "summer", "none", "slight"],
            ["saturday", "spring", "high", "heavy"],
            ["weekday", "summer", "high", "slight"],
            ["saturday", "winter", "normal", "none"],
            ["weekday", "summer", "high", "none"],
            ["weekday", "winter", "normal", "heavy"],
            ["saturday", "autumn", "high", "slight"],
            ["weekday", "autumn", "none", "heavy"],
            ["holiday", "spring", "normal", "slight"],
            ["weekday", "spring", "normal", "none"],
            ["weekday", "spring", "normal", "slight"]
        ]
    y_train_train = ["on time", "on time", "on time", "late", "on time", "very late", "on time",
                        "on time", "very late", "on time", "cancelled", "on time", "late", "on time",
                        "very late", "on time", "on time", "on time", "on time", "on time"]
   
    # create instance of model and fit
    model = MyNaiveBayesClassifier()
    model.fit(X_train_train, y_train_train)

    # expected priors (hard code-- desk check)
    expected_priors3 = {
        "on time": 0.70,       # 14/20
        "late": 0.10,          # 2/20
        "very late": 0.15,     # 3/20
        "cancelled": 0.05      # 1/20
    }

    # assert check against model
    for label, probability in expected_priors3.items():
        model_prior = model.priors[label]
        assert model_prior == probability
    
    print("Test Case3, prior asserts passed!!")

    # expected posteriors (hard code --desk check)
    expected_posteriors3 = {
        "on time": {
            0: {  # att is day (index 0)
                "weekday": 0.64,  # 9/14
                "saturday": 0.14,  # 2/14
                "sunday": 0.07,    # 1/14
                "holiday": 0.14    # 2/14
            },
            1: {  # att is season (index 1)
                "spring": 0.29,    # 4/14
                "summer": 0.43,    # 6/14
                "autumn": 0.14,    # 2/14
                "winter": 0.14     # 2/14
            },
            2: {  # att is wind (index 2)
                "none": 0.36,      # 5/14
                "high": 0.29,      # 4/14
                "normal": 0.36     # 5/14
            },
            3: {  # att is rain (index 3)
                "none": 0.36,      # 5/14
                "slight": 0.57,    # 8/14
                "heavy": 0.07      # 1/14
            }
        },
        "late": {
            0: {  # att is day (index 0)
                "weekday": 0.5,    # 1/2
                "saturday": 0.5,   # 1/2
                "sunday": 0,       # 0/2
                "holiday": 0       # 0/2
            },
            1: {  # att is season (index 1)
                "spring": 0,       # 0/2
                "summer": 0,       # 0/2
                "autumn": 0,       # 0/2
                "winter": 1        # 2/2
            },
            2: {  # att is wind (index 2)
                "none": 0,         # 0/2
                "high": 0.5,       # 1/2
                "normal": 0.5      # 1/2
            },
            3: {  # att is rain (index 3)
                "none": 0.5,       # 1/2
                "slight": 0,       # 0/2
                "heavy": 0.5       # 1/2
            }
        },
        "very late": {
            0: {  # att is day (index 0)
                "weekday": 1,      # 3/3
                "saturday": 0,     # 0/3
                "sunday": 0,       # 0/3
                "holiday": 0       # 0/3
            },
            1: {  # att is season (index 1)
                "spring": 0,       # 0/3
                "summer": 0,       # 0/3
                "autumn": 0.33,    # 1/3
                "winter": 0.67     # 2/3
            },
            2: {  # att is wind (index 2)
                "none": 0,         # 0/3
                "high": 0.33,      # 1/3
                "normal": 0.67     # 2/3
            },
            3: {  # att is rain (index 3)
                "none": 0.33,      # 1/3
                "slight": 0,       # 0/3
                "heavy": 0.67      # 2/3
            }
        },
        "cancelled": {
            0: {  # att is day (index 0)
                "weekday": 0,      # 0/1
                "saturday": 1,     # 1/1
                "sunday": 0,       # 0/1
                "holiday": 0       # 0/1
            },
            1: {  # att is season (index 1)
                "spring": 1,       # 1/1
                "summer": 0,       # 0/1
                "autumn": 0,       # 0/1
                "winter": 0        # 0/1
            },
            2: {  # att is wind (index 2)
                "none": 0,         # 0/1
                "high": 1,         # 1/1
                "normal": 0        # 0/1
            },
            3: {
                "none": 0,         # 0/1
                "slight": 0,       # 0/1
                "heavy": 1         # 1/1
            }
        }
    }

    # assert check against model
    # outer loop through each label
    for label, attribute_type in expected_posteriors3.items():

        # loop through each attribute index for current label (e.g. att1 and att2)
        for attribute_index, expected_probability in attribute_type.items():

            # loop through each attribute value and expected probability
            for attribute_value, expected_prob in expected_probability.items():
                model_posterior3 = model.posteriors[label][attribute_index][attribute_value]

                # check if model's posterior matches the expected posterior probability
                assert round(model_posterior3, 2) == round(expected_prob , 2)
    print("Test Case3, posterior asserts passed!!")

    print("ALL TESTS FOR FIT PASSED!!")

def test_naive_bayes_classifier_predict():
    # in-class Naive Bayes example (lab task #1)
    # header_inclass_example = ["att1", "att2"]
    X_train_inclass_example = [
        [1, 5], # yes
        [2, 6], # yes
        [1, 5], # no
        [1, 5], # no
        [1, 6], # yes
        [2, 6], # no
        [1, 5], # yes
        [1, 6] # yes
    ]
    y_train_inclass_example = ["yes", "yes", "no", "no", "yes", "no", "yes", "yes"]

    # create instance of model and fit
    model = MyNaiveBayesClassifier()
    model.fit(X_train_inclass_example, y_train_inclass_example)

    # test cases here
    X_test = [[1,5]] # needs to be a "list of lists" for predict method
    actual_class1 = ["yes"]

    model_predictions1 = model.predict(X_test)
    assert model_predictions1 == actual_class1

    # MA7 (fake) iPhone purchases dataset
    # header_iphone = ["standing", "job_status", "credit_rating", "buys_iphone"]
    X_train_iphone = [
        [1, 3, "fair"],
        [1, 3, "excellent"],
        [2, 3, "fair"],
        [2, 2, "fair"],
        [2, 1, "fair"],
        [2, 1, "excellent"],
        [2, 1, "excellent"],
        [1, 2, "fair"],
        [1, 1, "fair"],
        [2, 2, "fair"],
        [1, 2, "excellent"],
        [2, 2, "excellent"],
        [2, 3, "fair"],
        [2, 2, "excellent"],
        [2, 3, "fair"]
    ]
    y_train_iphone = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no", "yes"]

    # create instance of model and fit
    model = MyNaiveBayesClassifier()
    model.fit(X_train_iphone, y_train_iphone)

    # test cases here
    X_test1 = [[2,2, "fair"]]
    actual_class1 = ["yes"]

    X_test2 = [[1,1, "excellent"]]
    actual_class2 = ["no"]

    model_predictions1 = model.predict(X_test1)
    assert model_predictions1 == actual_class1

    model_predictions2 = model.predict(X_test2)
    assert model_predictions2 == actual_class2

    # Bramer 3.2 train dataset
    # header_train = ["day", "season", "wind", "rain", "class"]
    X_train_train = [
        ["weekday", "spring", "none", "none"],
        ["weekday", "winter", "none", "slight"],
        ["weekday", "winter", "none", "slight"],
        ["weekday", "winter", "high", "heavy"],
        ["saturday", "summer", "normal", "none"],
        ["weekday", "autumn", "normal", "none"],
        ["holiday", "summer", "high", "slight"],
        ["sunday", "summer", "normal", "none"],
        ["weekday", "winter", "high", "heavy"],
        ["weekday", "summer", "none", "slight"],
        ["saturday", "spring", "high", "heavy"],
        ["weekday", "summer", "high", "slight"],
        ["saturday", "winter", "normal", "none"],
        ["weekday", "summer", "high", "none"],
        ["weekday", "winter", "normal", "heavy"],
        ["saturday", "autumn", "high", "slight"],
        ["weekday", "autumn", "none", "heavy"],
        ["holiday", "spring", "normal", "slight"],
        ["weekday", "spring", "normal", "none"],
        ["weekday", "spring", "normal", "slight"]
    ]
    y_train_train = ["on time", "on time", "on time", "late", "on time", "very late", "on time",
                    "on time", "very late", "on time", "cancelled", "on time", "late", "on time",
                    "very late", "on time", "on time", "on time", "on time", "on time"] 

    # create instance of model and fit
    model = MyNaiveBayesClassifier()
    model.fit(X_train_train, y_train_train)

    # 3.2 Bramer test case & 3.6 unseen instances
    X_test3 = [["weekday", "winter", "high", "heavy"]]
    actual_class3 = ["very late"]

    model_predictions3 = model.predict(X_test3)
    assert model_predictions3 == actual_class3

    X_test4 = [["weekday", "summer", "high", "heavy"]]
    actual_class4 = ["on time"]

    model_predictions4 = model.predict(X_test4)
    assert model_predictions4 == actual_class4

    X_test5 = [["sunday", "summer", "normal", "slight"]]
    actual_class5 = ["on time"]

    model_predictions5 = model.predict(X_test5)
    assert model_predictions5 == actual_class5
    print("ALL TESTS FOR PREDICT PASSED!!")

def test_decision_tree_classifier_fit():
    # 14 instance interview dataset (test case 1)
    # header_interview = ["level", "lang", "tweets", "phd", "interviewed_well"]
    X_train_interview = [
        ["Senior", "Java", "no", "no"],
        ["Senior", "Java", "no", "yes"],
        ["Mid", "Python", "no", "no"],
        ["Junior", "Python", "no", "no"],
        ["Junior", "R", "yes", "no"],
        ["Junior", "R", "yes", "yes"],
        ["Mid", "R", "yes", "yes"],
        ["Senior", "Python", "no", "no"],
        ["Senior", "R", "yes", "no"],
        ["Junior", "Python", "yes", "no"],
        ["Senior", "Python", "yes", "yes"],
        ["Mid", "Python", "no", "yes"],
        ["Mid", "Java", "yes", "no"],
        ["Junior", "Python", "no", "yes"]
    ]
    y_train_interview = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]

    # note: this tree uses the generic "att#" attribute labels because fit() does not and should not accept attribute names
    # note: the attribute values are sorted alphabetically
    expected_tree_interview = \
            ["Attribute", 0, # splits on attribute at index 0 (e.g. level)
                ["Value", "Junior",
                    ["Attribute", 3,
                        ["Value", "no", 
                            ["Leaf", "True", 3, 5]
                        ],
                        ["Value", "yes",
                            ["Leaf", "False", 2, 5]
                        ]
                    ]
                ],
                ["Value", "Mid",
                    ["Leaf", "True", 4, 14]
                ],
                ["Value", "Senior",
                    ["Attribute", 2,
                        ["Value", "no",
                            ["Leaf", "False", 3, 5]
                        ],
                        ["Value", "yes",
                            ["Leaf", "True", 2, 5]
                        ]
                    ]
                ]
            ]
    
    model = MyDecisionTreeClassifier()
    model.fit(X_train_interview, y_train_interview)

    assert model.tree == expected_tree_interview
    print("DEBUG: DECISION TREE FIT METHOD PASSED TEST 1")

    # 15 instance iPhone training set example
    # header_phone = ["standing", "job_status", "credit_rating", "buys_iphone"]
    X_train_phone = [
        [1, 3, "fair"],
        [1, 3, "excellent"],
        [2, 3, "fair"],
        [2, 2, "fair"],
        [2, 1, "fair"],
        [2, 1, "excellent"],
        [2, 1, "excellent"],
        [1, 2, "fair"],
        [1, 1, "fair"],
        [2, 2, "fair"],
        [1, 2, "excellent"],
        [2, 2, "excellent"],
        [2, 3, "fair"],
        [2, 2, "excellent"],
        [2, 3, "fair"]
    ]

    y_train_phone = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no", "yes"]

    expected_tree_iphone = \
        ["Attribute", 0,  # first split on attribute at index 0 (e.g., Standing)
            ["Value", 1,
                ["Attribute", 1,  # next split is on attribute at index 1 (e.g., Job_Status)
                    ["Value", 1, 
                        ["Leaf", "yes", 1, 5]
                    ],
                    ["Value", 2,
                        ["Attribute", 2,  # next split on attribute at index 2 (e.g., Credit_Rating)
                            ["Value", "excellent",
                                ["Leaf", "yes", 1, 2]

                            ],
                            ["Value", "fair",
                                ["Leaf", "no", 1, 2]
                            ],
                        ]
                    ],
                    ["Value", 3,
                        ["Leaf", "no", 2, 5]
                    ]
                ]
            ],
            ["Value", 2,
                ["Attribute", 2,  # next split for standing 2 is on attribute at index 2 (e.g., Credit_Rating)
                    ["Value", "excellent",
                        ["Leaf", "no", 4, 10]
                    ],
                    ["Value", "fair",
                        ["Leaf", "yes", 6, 10]
                    ]
                ]
            ]
        ]
    
    print(expected_tree_iphone)
    
    model = MyDecisionTreeClassifier()
    model.fit(X_train_phone, y_train_phone)

    print(model.tree)

    assert model.tree == expected_tree_iphone
    print("DEBUG: DECISION TREE FIT METHOD PASSED TEST 2")


def test_decision_tree_classifier_predict():
    # 14 instance interview dataset (test case 1)
    # header_interview = ["level", "lang", "tweets", "phd", "interviewed_well"]
    X_train_interview = [
        ["Senior", "Java", "no", "no"],
        ["Senior", "Java", "no", "yes"],
        ["Mid", "Python", "no", "no"],
        ["Junior", "Python", "no", "no"],
        ["Junior", "R", "yes", "no"],
        ["Junior", "R", "yes", "yes"],
        ["Mid", "R", "yes", "yes"],
        ["Senior", "Python", "no", "no"],
        ["Senior", "R", "yes", "no"],
        ["Junior", "Python", "yes", "no"],
        ["Senior", "Python", "yes", "yes"],
        ["Mid", "Python", "no", "yes"],
        ["Mid", "Java", "yes", "no"],
        ["Junior", "Python", "no", "yes"]
    ]
    y_train_interview = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]

    model = MyDecisionTreeClassifier()
    model.fit(X_train_interview, y_train_interview)

    # need test cases to be a "list of lists" since that is what the predict method expects
    test_case1 = [["Junior", "Java", "yes", "no"]]
    test_case2 = [["Junior", "Java", "yes", "yes"]]

    expected_prediction1 = "True"
    expected_prediction2 = "False"

    model_pred1 = model.predict(test_case1)
    model_pred2 = model.predict(test_case2)

    assert model_pred1 == [expected_prediction1]

    assert model_pred2 == [expected_prediction2]
    print("DEBUG: DECISION TREE PREDICT METHOD PASSED TEST 1")

    # 15 instance iPhone training set example
    # header_phone = ["standing", "job_status", "credit_rating", "buys_iphone"]
    X_train_phone = [
        [1, 3, "fair"],
        [1, 3, "excellent"],
        [2, 3, "fair"],
        [2, 2, "fair"],
        [2, 1, "fair"],
        [2, 1, "excellent"],
        [2, 1, "excellent"],
        [1, 2, "fair"],
        [1, 1, "fair"],
        [2, 2, "fair"],
        [1, 2, "excellent"],
        [2, 2, "excellent"],
        [2, 3, "fair"],
        [2, 2, "excellent"],
        [2, 3, "fair"]
    ]

    y_train_phone = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no", "yes"]

    model = MyDecisionTreeClassifier()
    model.fit(X_train_phone, y_train_phone)

    # need test cases to be a "list of lists" since that is what the predict method expects
    test_case1 = [[2, 2, "fair"]]
    test_case2 = [[1, 1, "excellent"]]

    expected_prediction1 = "yes"
    expected_prediction2 = "yes"

    model_pred1 = model.predict(test_case1)
    model_pred2 = model.predict(test_case2)

    assert model_pred1 == [expected_prediction1]

    assert model_pred2 == [expected_prediction2]
    print("DEBUG: DECISION TREE PREDICT METHOD PASSED TEST 2")