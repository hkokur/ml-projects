import pandas as pd
import numpy as np

dataframe = pd.read_csv("play_tenis.csv")
main_column = "Play Tennis"
columns = dataframe.columns.drop([main_column])


def load_dataset(file_name):
    return pd.read_csv(file_name)


def store_model(model, file_name):
    model.to_csv(file_name, index=False)


def one_hot_encoding(dataframe, columns, main_column):
    # One-hot encoding
    dataframe = pd.get_dummies(dataframe, columns=columns)
    for column in dataframe.columns.drop([main_column]):
        dataframe[column] = dataframe[column].map({True: 1, False: 0})
    return dataframe


def compute_distance(instance, instance2, main_column, calculation_type):
    # calculate the distance between two instances
    distance = 0

    for column in instance.drop([main_column]).index:
        if calculation_type == "manhattan":
            distance += abs(instance[column] - instance2[column])
        else:
            distance += (instance[column] - instance2[column]) ** 2

    return distance


def get_k_nearest_neighborhoods(model, instance2, main_column, k, calculation_type):
    # calculate distance
    model["Distance"] = 0.0
    for index, instance in model.iterrows():
        instance = instance.drop(["Distance"])
        distance = compute_distance(instance, instance2, main_column, calculation_type)
        model.loc[index, "Distance"] = distance

    # sort the dataframe by distance
    model = model.sort_values(by="Distance", ascending=True)

    # get the k nearest neighbors
    return model.head(k).copy()


def determine_prediction(neighbors, main_column):
    # Get the majority vote from the nearest neighbors
    prediction = neighbors[main_column].mode()[0]
    return prediction


if __name__ == "__main__":
    # our main column
    main_column = "Play Tennis"

    # train new dataset or load the model
    train_new = input("Train new model(y/n): ")

    if train_new.lower() == "y":
        # get the dataset file name, if not provided, use the default dataset
        input_file = input(
            "Enter the training play tenis json(default: play_tenis.csv): "
        )
        if input_file != "" and input_file:
            file_name = input_file
        else:
            file_name = "play_tenis.csv"

        # load the dataset
        dataset = load_dataset(file_name)

        # convert the dataset to one-hot encoding
        dataset = one_hot_encoding(
            dataset, dataset.columns.drop([main_column]), main_column
        )
        # print(dataset.head(100))
        # store the model
        print("Dataset: \n", dataset.head(20))
        store_model(dataset, "model.csv")

    model = load_dataset("model.csv")
    # print("Model: \n", model.head(20))

    calculation_type = input("Enter the calculation type(euclidean/manhattan): ")
    # set default euclidean
    calculation_type = (
        "euclidean"
        if calculation_type != "euclidean" or calculation_type != "manhattan"
        else calculation_type
    )

    k_number = int(input("Enter the k number: "))

    # i use the same dataset for testing
    test = model.copy()
    # add the prediction column into my model
    model["Play Tennis Prediction"] = "0"  # i give random value initially

    for index, instance in test.iterrows():
        nearest_neighborhoods = get_k_nearest_neighborhoods(
            model.copy().drop(columns=["Play Tennis Prediction"]),
            instance,
            main_column,
            k_number,
            calculation_type,
        )
        print(f"Nearest Neighborhoods for {index}: \n", nearest_neighborhoods.head(100))
        # make a prediction
        prediction = determine_prediction(nearest_neighborhoods, main_column)
        model.at[index, "Play Tennis Prediction"] = prediction

    # print("Model with prediction: \n", model.head(20))

    # add TP, FP, TN, FN
    model["Confusion Matrix"] = None
    for index, row in model.iterrows():
        actual = row["Play Tennis"]
        pre = row["Play Tennis Prediction"]
        if actual == "Yes" and pre == "Yes":
            df = model.at[index, "Confusion Matrix"] = "TP"
        elif actual == "Yes" and pre == "No":
            df = model.at[index, "Confusion Matrix"] = "FP"
        elif actual == "No" and pre == "Yes":
            df = model.at[index, "Confusion Matrix"] = "FN"
        elif actual == "No" and pre == "No":
            df = model.at[index, "Confusion Matrix"] = "TN"

    print("Model with prediction and confusion matrix: \n", model.head(20))
    print(
        "Accuracy: ",
        (model["Play Tennis"] == model["Play Tennis Prediction"]).sum() / len(model),
    )
    print(
        "Precision: ",
        (model["Confusion Matrix"] == "TP").sum()
        / (model["Confusion Matrix"].isin(["TP", "FP"]).sum()),
    )
    print(
        "Recall: ",
        (model["Confusion Matrix"] == "TP").sum()
        / (model["Confusion Matrix"].isin(["TP", "FN"]).sum()),
    )
