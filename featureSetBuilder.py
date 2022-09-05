from textblob import TextBlob
from preprocessing_cleaning_and_feature_set_updated import punctuations_list, STOPWORDS 


def pos(df, flag):
    pos_dic = {
    'noun' : ['NN','NNS','NNP','NNPS'],
    'pron' : ['PRP','PRP$','WP','WP$'],
    'verb' : ['VB','VBD','VBG','VBN','VBP','VBZ'],
    'adj' :  ['JJ','JJR','JJS'],
    'adv' : ['RB','RBR','RBS','WRB']
    }

    cnt = 0
    try:
        TEXT = TextBlob(df)
        for tup in TEXT.tags:
            #print(tup)
            asi = list(tup)[1]
            #print(asi)
            if asi in pos_dic[flag]:
                cnt += 1
    except:
        pass
    return cnt

def features(text, num=0, inc_tag=False):
    char_count = len(text[num+2])
    word_count = len(text[num+2].split())
    word_density = char_count / word_count + 1
    punctuation_count = len("".join(p for p in text[num+2] if p in punctuations_list))
    title_word_count = len([wrd for wrd in text[num+2].split() if wrd.istitle()])
    upper_case_word_count = len([wrd for wrd in text[num+2].split() if wrd.isupper()])
    stopword_count = len([wrd for wrd in text[num+2].split() if wrd.lower() in STOPWORDS])
    avg_wordlength = sum([len(wrd) for wrd in text[num+2].split()]) / word_count
    
    # True/False, Whether the tweet was posted on SUnday or Saturday
    weekend = text[num] in ['Sunday', 'Saturday']
    len_user = len(text[num+1]) # LEngth of the username
    len_cleaned = len(text[num+3]) # Number of words left after being cleaned
    len_change = word_count - len_cleaned # Difference in length between cleaned and original text
    # Longest word of the cleaned words
    largest_cleaned = max([len(wrd) for wrd in text[num+3]]) if type(text[num+3]) == 'list' else 0 
    num_ats = text[num+2].count('@') # Number of @'s used in the original tweet.
    num_hash = text[num+2].count('#') # Number of hashtags used.
    # Number of non-alphabetic words in the username, so like 12, -, ...
    cnt_non_alpha = len([c for c in text[num+1] if not c.isalpha()]) 
    
    
    feature_set = {'char_count':char_count, 'word_count':word_count, 'word_density':word_density, 
                   'punctuation_count':punctuation_count, 'title_word_count':title_word_count, 
                   'upper_case_word_count':upper_case_word_count, 'stopword_count':stopword_count, 
                   'avg_wordlength':avg_wordlength, 'noun_count':pos(text[num+2], 'noun'), 
                   'verb_count':pos(text[num+2], 'verb'), 'adj_count':pos(text[num+2], 'adj'), 
                   'adv_count':pos(text[num+2], 'adv'), 'pron_count':pos(text[num+2], 'pron'), 
                   'weekend':weekend, 'len_user':len_user, 'len_cleaned':len_cleaned,
                   'len_change':len_change, 'largest_cleaned':largest_cleaned, 
                   'num_ats':num_ats, 'num_hash':num_hash, 'cnt_non_alpha':cnt_non_alpha}
    
    return (feature_set, text[num-1]) if inc_tag else feature_set


def main(df):
    # Variable to decide how to split dataframe into train and test sets, currently at 70:30 split.
    frac = int(len(df) * 0.7)

    # Splitting dataset
    train_set, test_set = df.iloc[:frac], df.iloc[frac:]
    train_set['Feature_Set'] = train_set.apply(features, axis=1, num=1, inc_tag=True)

    return train_set, test_set

def tweet_features(df):
    df['Feature_Set'] = df.apply(features, axis=1)
    return df
    