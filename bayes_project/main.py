import pandas as pd

dataframe = pd.read_csv("play_tenis.csv")


# counting functio to our main class
def count_cls(df, cls_column):
    count = dict()
    for i in df[cls_column].unique():
        count[i] = len(df[df[cls_column] == i])
    return count


# counting function to features
def count_feature(df, feature_column, cls_column):
    count = dict()
    cls = df[cls_column].unique()
    for i in df[feature_column].unique():
        count[i] = dict()
        for j in cls:
            count[i][j] = len(df[(df[feature_column] == i) & (df[cls_column] == j)])
    return count


# we get count's the class and the features, so we can calculate probability
cls_count = count_cls(dataframe, "Play Tennis")
featurs_count = {}
for i in dataframe.columns[:-1]:
    featurs_count[i] = count_feature(dataframe, i, "Play Tennis")

# calculate the prior probability
prior_prob = pd.DataFrame(columns=["P(Yes)", "P(No)"])
prior_prob.loc[0] = [
    cls_count.get("Yes") / (cls_count.get("Yes") + cls_count.get("No")),
    cls_count.get("No") / (cls_count.get("Yes") + cls_count.get("No")),
]

# calculate the prob for each indepent value(own by feature)
likehoods_prob = pd.DataFrame(
    columns=["Feature", "Value", "P(Value|Yes)", "P(Value|No)"]
)
for feature_key in featurs_count.keys():
    for value_key in featurs_count.get(feature_key).keys():
        # Create a DataFrame for the current feature-value pair with calculated probabilities
        current_prob = pd.DataFrame(
            [
                {
                    "Feature": feature_key,
                    "Value": value_key,
                    # laplace smoothing, adding one numenoter and adding N(unique number of feature value)
                    "P(Value|Yes)": (featurs_count[feature_key][value_key]["Yes"] + 1)
                    / (cls_count["Yes"] + len(featurs_count[feature_key])),
                    "P(Value|No)": (featurs_count[feature_key][value_key]["No"] + 1)
                    / (cls_count["No"] + len(featurs_count[feature_key])),
                }
            ]
        )

        # Concatenate the current probabilities DataFrame with the main likelihoods_prob DataFrame
        likehoods_prob = pd.concat([likehoods_prob, current_prob], ignore_index=True)


# prediction function
def prediction(instance, prior_prob, likehoods_prob):
    cls_probs = dict()
    for cls in prior_prob.columns:
        cls_probs[cls] = prior_prob[cls].values[0]
        for feature in instance.keys():
            # We need to calculate P(Features | Class) which is the product of the probabilities of each feature given the class:
            # P(Feature_1 | Class) * P(Feature_2 | Class) * ... * P(Feature_n | Class)
            lp = likehoods_prob[
                (likehoods_prob["Feature"] == feature)
                & (likehoods_prob["Value"] == instance[feature])
            ]
            # cls P(Yes) or P(No)
            cls_probs[cls] *= lp[
                "P(Value|Yes)" if cls == "P(Yes)" else "P(Value|No)"
            ].values[0]
            # print(lp)
            # print(lp["P(Value|Yes)" if cls == "P(Yes)" else "P(Value|No)"].values[0])

    return max(cls_probs, key=cls_probs.get), cls_probs


## make prediction for whole dataset
df_with_predictions = dataframe.copy()
df_with_predictions["Play Tennis(Prediction)"] = None
for index, row in df_with_predictions.iterrows():
    new_instance = {
        "Outlook": row["Outlook"],
        "Temperature": row["Temperature"],
        "Humidity": row["Humidity"],
        "Wind": row["Wind"],
    }
    cls, cls_probs = prediction(new_instance, prior_prob, likehoods_prob)
    df = df_with_predictions.at[index, "Play Tennis(Prediction)"] = (
        cls.replace("(", "").replace("P", "").replace(")", "")
    )


# print(df_with_predictions.head(100))

# add TP, FP, TN, FN
df_with_predictions["Confusion Matrix"] = None
for index, row in df_with_predictions.iterrows():
    actual = row["Play Tennis"]
    pre = row["Play Tennis(Prediction)"]
    if actual == "Yes" and pre == "Yes":
        df = df_with_predictions.at[index, "Confusion Matrix"] = "TP"
    elif actual == "Yes" and pre == "No":
        df = df_with_predictions.at[index, "Confusion Matrix"] = "FP"
    elif actual == "No" and pre == "Yes":
        df = df_with_predictions.at[index, "Confusion Matrix"] = "FN"
    elif actual == "No" and pre == "No":
        df = df_with_predictions.at[index, "Confusion Matrix"] = "TN"

print(df_with_predictions.head(100))
print(
    "Accuracy: ",
    (
        df_with_predictions["Play Tennis"]
        == df_with_predictions["Play Tennis(Prediction)"]
    ).sum()
    / len(df_with_predictions),
)
