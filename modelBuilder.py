from nltk import NaiveBayesClassifier
from featureSetBuilder import features

def gen_classifier(feature_sets):
    return NaiveBayesClassifier.train(feature_sets)


def main(train_set,test_set):
    classifier = gen_classifier(train_set['Feature_Set'])
    # Testing to see if classifier works.
    test_set['Features'] = test_set.apply(features, axis=1)
    test_set['Pred_Sent'] = test_set['Features'].apply(lambda x: classifier.classify(x))

    # Calculating accuracy.
    test_set['correct'] = test_set['sentiment'] == test_set['Pred_Sent']
    accuracy = sum(test_set['correct'])/len(test_set)

    # Show most informative features.
    classifier.show_most_informative_features(15)
    
    return classifier, accuracy

def tweet_classifier(classifier, df):
    df['Pred_Sent'] = df['Feature_Set'].apply(lambda x: classifier.classify(x))
    return df

