from mysklearn.myclassifiers import MyRandomForestClassifier

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

def test_random_forest_classifier_fit():
    # N = number of trees, M = features, F = features per split
    # seed random generator here:
    model = MyRandomForestClassifier(N=5, M=3, F=2, seed_random=4)
    model.fit(X_train_interview, y_train_interview)

    # for idx, dt in enumerate(model.trees, start=1):
    #     print(f"Tree {idx}:")
    #     print(dt.tree)
    #     print("-" * 40)

    # check the correct number of trees (N) were created in the ensemble
    assert len(model.trees) == model.M

    # check that M and F are set correctly
    assert model.M == 3
    assert model.F == 2

    # check all trees in the ensemble are unique
    tree_signatures = [str(tree) for tree in model.trees]
    assert len(tree_signatures) == len(set(tree_signatures))

def test_random_forest_classifier_predict():
    model = MyRandomForestClassifier(N =5, M =3, F = 2, seed_random=5)
    model.fit(X_train_interview, y_train_interview)

    # test cases:
    test_case1 = ["Junior", "Java", "yes", "no"] # true
    test_case2 = ["Junior", "Java", "yes", "yes"] # false

    prediction1 = model.predict([test_case1])
    # print(prediction1)
    prediction2 = model.predict([test_case2])
    # print(prediction2)

    assert prediction1 == ["True"]
    assert prediction2 == ["False"]
