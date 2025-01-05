# pylint: skip-file
import numpy as np
from scipy import stats
from mysklearn import myutils
from mysklearn.mysimplelinearregressor import MySimpleLinearRegressor
from mysklearn.myclassifiers import MySimpleLinearRegressionClassifier,\
    MyKNeighborsClassifier,\
    MyDummyClassifier

# note: order is actual/received student value, expected/solution
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

    

# DEBUGGING PRINT STATEMENTS HERE:
# Call each test function to see the print statements
test_simple_linear_regression_classifier_fit()
test_simple_linear_regression_classifier_predict()
test_dummy_classifier_fit()
test_dummy_classifier_predict()
test_kneighbors_classifier_kneighbors()
test_kneighbors_classifier_predict()
