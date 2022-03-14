import pandas as pd

# pd.set_option('display.max_columns',20)
# pd.set_option('display.width', 2000)
import re


# word count in every label
'''
def label_word(data, label=None):
    if label == None:
        for col in data.drop(['comment_text', 'id'], axis=1):
            data = data[data[col] == 0]
    else:
        for col in data.drop(['comment_text', 'id', label], axis=1):
            data = data[data[col] == 0]
            data = data[data[label] == 1]
    col_freq = FreqDist([w for row in data['comment_text'] for w in row.split()])

    return data, col_freq

for word in ['fuck', 'ass', 'sex', 'gay', 'bitch', 'bullshit', 'kill', 'homosexual', 'dick',
             'motherfucker', 'anus', 'vagina', 'jewish', 'muslim', 'big', 'black', 'white', 'rape', 'die',
             'slut', 'hell', 'jew', 'nigga', 'kidnap', 'homosexual', 'murder', 'dumb', 'violate',
             'sever', 'blow', 'target', 'hate', 'stalk', 'thy', 'vandalism', 'beat', 'ticket', 'argument',
             'perhaps', 'something', 'ticket', 'buy', 'delete', 'may', 'year', 'month', 'username', 'email', 'good',
             'luck', 'bad', 'nice', 'night', 'hour', 'ago', 'yesterday', 'long', 'short', 'post', 'name',
             'enjoy', 'way', 'no', 'yes', 'thanks', 'thank', 'easy', 'make', 'would', 'notice', 'result', 'article',
             'people', 'among', 'legal', 'know', 'case', 'even', 'old', 'new', 'say', 'claim', 'best', 'happy',
             'sad', 'soon', 'sure', 'wrong', 'right', 'place', 'get', 'able', 'link', 'url', 'avoid', 'need', 'use',
             'wrong',
             'add', 'delete', 'major', 'person', 'war', 'fight', 'fire', 'porn', 'speak', 'talk',
             'massacre', 'nigger', 'woman', 'man', 'race', 'hit', 'block', 'keep',
             'edit', 'stop', 'please', 'cat', 'refers', 'picture', 'video', 'image', 'play', 'well',
             'gold', 'many', 'luck', 'fat', 'pussy', 'lick', 'suck', 'worst', 'best',
             'include', 'player', 'message', 'see', 'add', 'top', 'suck', 'worst', 'best', 'information',
             'data', 'computer', 'science', 'junk', 'company', 'check', 'go', 'try', 'one', 'second',
             'page', 'usually', 'side', 'find', 'instead', 'hello', 'correct', 'book', 'someone', 'might',
             'nazi', 'nothing', 'bias', 'skill', 'citizens', 'upset', 'sorry', 'agree', 'source', 'reason',
             'chapter', 'history', 'christian', 'read', 'anything', 'doubt', 'start', 'end', 'although', 'rather',
             'metal', 'death', 'rock', 'similar', 'problem', 'oil', 'work', 'click', 'work', 'also',
             'room', 'growth', 'enough', 'biology', 'productive', 'attack', 'law', 'call', 'fascism', 'attack',
             'stupid', 'lose', 'kick', 'expire', 'body', 'blind', 'force', 'betrayal', 'wife', 'love',
             'probably', 'revenge', 'daily', 'sound', 'body', 'think', 'select', 'another', 'notice', 'else',
             'admin', 'complete', 'honest', 'responsibility', 'suck', 'cock', 'pay', 'attention', 'ban', 'give',
             'cocksucking', 'bastard', 'meet', 'artist', 'invite', 'appreciate', 'view', 'ask', 'real', 'background',
             'refer', 'stuff', 'locate', 'moment', 'break', 'ahead', 'type', 'write', 'smarter', 'fun',
             'rag', 'faggot', 'understand', 'encyclopedia', 'business', 'directory', 'understand', 'development',
             'mention', 'relevant',
             'copyright', 'touch', 'ugly', 'butt', 'wise', 'defined', 'fugly', 'development', 'mention', 'step',
             'behaviour',
             'police', 'political', 'israeli', 'gaza', 'nationalist', 'crap', 'arrest', 'prison', 'mention', 'assist',
             'welcome', 'nice', 'lady', 'featured', 'accept', 'condition', 'discussion', 'prison', 'evidence', 'kind',
             'list', 'product', 'script', 'lead', 'cannot', 'sexual', 'search', 'reliable', 'source',
             'conflict', 'alone', 'others', 'grammer', 'typo', 'mistake', 'search', 'forget', 'password',
             'dear', 'die', 'son', 'bich', 'substantive', 'agenda', 'come', 'check', 'put', 'project',

             'percentage', 'never', 'biographical', 'idea', 'accuracy', 'conclusion', 'support', 'percentage', 'never',
             'fiction', 'literary', 'content', 'criticism', 'help', 'everyone', 'warn', 'interest', 'later',
             'judge', 'movie', 'invent', 'number', 'free', 'discount', 'skin', 'gun', 'fan', 'stop', 'peace', 'read',

             'station', 'result', 'fix', 'budget', 'billion', 'follow', 'estimate', 'live', 'event', 'first', 'random',
             'hello', 'let', 'parent', 'date', 'actually', 'time', 'discussion', 'wikipedia', 'either', 'neither']:
    none, freq_none = label_word(data)
    toxic, freq_toxic = label_word(data, 'toxic')
    severe_toxic, freq_severe_toxic = label_word(data, 'severe_toxic')
    obscene, freq_obscene = label_word(data, 'obscene')
    threat, freq_threat = label_word(data, 'threat')
    insult, freq_insult = label_word(data, 'insult')
    identity_hate, freq_identity_hate = label_word(data, 'identity_hate')
    for col, name in zip([freq_toxic,
                          freq_severe_toxic, freq_obscene, freq_threat, freq_insult, freq_identity_hate, freq_none],
                         ['toxic',
                          'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate', 'none']):
        print(f'{name}: {word}={col[word]}')
    print()
    print()'''

extracted_words = pd.Series(['fuck', 'ass', 'sex', 'gay', 'bitch', 'bullshit', 'kill', 'homosexual', 'dick',
                             'motherfucker', 'anus', 'vagina', 'jewish', 'muslim', 'big', 'black', 'white', 'rape',
                             'die',
                             'slut', 'hell', 'jew', 'nigga', 'kidnap', 'homosexual', 'murder', 'dumb', 'violate',
                             'sever', 'blow', 'target', 'hate', 'stalk', 'thy', 'vandalism', 'beat', 'ticket',
                             'argument',
                             'perhaps', 'something', 'ticket', 'buy', 'delete', 'may', 'year', 'month', 'username',
                             'email',
                             'good',
                             'luck', 'bad', 'nice', 'night', 'hour', 'ago', 'yesterday', 'long', 'short', 'post',
                             'name',
                             'enjoy', 'way', 'no', 'yes', 'thanks', 'thank', 'easy', 'make', 'would', 'notice',
                             'result',
                             'article',
                             'people', 'among', 'legal', 'know', 'case', 'even', 'old', 'new', 'say', 'claim', 'best',
                             'happy',
                             'sad', 'soon', 'sure', 'wrong', 'right', 'place', 'get', 'able', 'link', 'url', 'avoid',
                             'need',
                             'use',
                             'wrong',
                             'add', 'delete', 'major', 'person', 'war', 'fight', 'fire', 'porn', 'speak', 'talk',
                             'massacre', 'nigger', 'woman', 'man', 'race', 'hit', 'block', 'keep',
                             'edit', 'stop', 'please', 'cat', 'refers', 'picture', 'video', 'image', 'play', 'well',
                             'gold', 'many', 'luck', 'fat', 'pussy', 'lick', 'suck', 'worst', 'best',
                             'include', 'player', 'message', 'see', 'add', 'top', 'suck', 'worst', 'best',
                             'information',
                             'data', 'computer', 'science', 'junk', 'company', 'check', 'go', 'try', 'one', 'second',
                             'page', 'usually', 'side', 'find', 'instead', 'hello', 'correct', 'book', 'someone',
                             'might',
                             'nazi', 'nothing', 'bias', 'skill', 'citizens', 'upset', 'sorry', 'agree', 'source',
                             'reason',
                             'chapter', 'history', 'christian', 'read', 'anything', 'doubt', 'start', 'end', 'although',
                             'rather',
                             'metal', 'death', 'rock', 'similar', 'problem', 'oil', 'work', 'click', 'work', 'also',
                             'room', 'growth', 'enough', 'biology', 'productive', 'attack', 'law', 'call', 'fascism',
                             'attack',
                             'stupid', 'lose', 'kick', 'expire', 'body', 'blind', 'force', 'betrayal', 'wife', 'love',
                             'probably', 'revenge', 'daily', 'sound', 'body', 'think', 'select', 'another', 'notice',
                             'else',
                             'admin', 'complete', 'honest', 'responsibility', 'suck', 'cock', 'pay', 'attention', 'ban',
                             'give',
                             'cocksucking', 'bastard', 'meet', 'artist', 'invite', 'appreciate', 'view', 'ask', 'real',
                             'background',
                             'refer', 'stuff', 'locate', 'moment', 'break', 'ahead', 'type', 'write', 'smarter', 'fun',
                             'rag', 'faggot', 'understand', 'encyclopedia', 'business', 'directory', 'understand',
                             'development',
                             'mention', 'relevant',
                             'copyright', 'touch', 'ugly', 'butt', 'wise', 'defined', 'fugly', 'development', 'mention',
                             'step',
                             'behaviour',
                             'police', 'political', 'israeli', 'gaza', 'nationalist', 'crap', 'arrest', 'prison',
                             'mention',
                             'assist',
                             'welcome', 'nice', 'lady', 'featured', 'accept', 'condition', 'discussion', 'prison',
                             'evidence',
                             'kind',
                             'list', 'product', 'script', 'lead', 'cannot', 'sexual', 'search', 'reliable', 'source',
                             'conflict', 'alone', 'others', 'grammer', 'typo', 'mistake', 'search', 'forget',
                             'password',
                             'dear', 'die', 'son', 'bich', 'substantive', 'agenda', 'come', 'check', 'put', 'project',

                             'percentage', 'never', 'biographical', 'idea', 'accuracy', 'conclusion', 'support',
                             'percentage',
                             'never',
                             'fiction', 'literary', 'content', 'criticism', 'help', 'everyone', 'warn', 'interest',
                             'later',
                             'judge', 'movie', 'invent', 'number', 'free', 'discount', 'skin', 'gun', 'fan', 'stop',
                             'peace',
                             'read',

                             'station', 'result', 'fix', 'budget', 'billion', 'follow', 'estimate', 'live', 'event',
                             'first',
                             'random',
                             'hello', 'let', 'parent', 'date', 'actually', 'time', 'discussion', 'wikipedia', 'either',
                             'neither'])

drop = pd.Series(
    ['fuck', 'ass', 'bitch', 'bullshit', 'kill', 'homosexual', 'dick', 'motherfucker', 'anus', 'vagina', 'jewish',
     'muslim', 'big', 'rape', 'die', 'slut', 'hell', 'jew', 'nigga', 'nigga', 'homosexual', 'murder', 'dumb', 'sever',
     'blow',
     'stalk', 'thy', 'sad', 'soon', 'porn', 'speak', 'massacre', 'nigga', 'cat', 'gold', 'luck', 'fat', 'hello',
     'pussy', 'lick', 'suck', 'worst', 'computer', 'science', 'junk', 'company', 'hello', 'nazi', 'skill',
     'citizens',
     'upset', 'christian', 'metal', 'rock', 'room', 'growth', 'biology', 'productivem', 'fascism', 'stupid', 'lose',
     'kick', 'expire', 'blind', 'betrayal', 'wife', 'revenge', 'daily', 'select', 'responsibility', 'suck', 'cock',
     'cocksucking', 'bastard', 'artist', 'background', 'locate', 'moment', 'smarter', 'fun', 'rag', 'faggot', 'butt'
        , 'wise', 'fugly', 'israeli', 'gaza', 'nationalist', 'crap', 'arrest', 'prison', 'assist', 'grammer', 'typo',
     'mistake', 'forget', 'password', 'password', 'dear', 'die', 'son', 'bich', 'substantive', 'percentage',
     'fiction'
        , 'discount', 'skin', 'gun', 'literary', 'warn', 'judge', 'movie', 'estimate', 'parent'
     ])
drop_50 = ['sexual', 'conflict', 'alone', 'accuracy', 'conclusion', 'criticism', 'fan', 'budget', 'billion', 'event',
           'neither'
           'featured', 'lady', 'condition', 'product', 'script', 'sexual', 'agenda', 'cannot', 'reliable', 'source',
           'others'
           'break', 'ahead', 'write', 'directory', 'development', 'touch', 'ugly', 'development', 'step', 'behaviour',
           'police',
           'sound', 'body', 'complete', 'honest', 'pay', 'ban', 'meet', 'artist', 'invite', 'view', 'ask', 'real',
           'business'
           'sex', 'blow', 'hate', 'beat', 'ticket', 'buy', 'night', 'yesterday', 'url', 'fight', 'fire', 'woman', 'man',
           'station'
           'race', 'hit', 'refers', 'video', 'play', 'player', 'data', 'bias', 'chapter', 'death', 'oil', 'click',
           'force']
important = ['come', 'check', 'put', 'never', 'idea', 'support', 'never', 'content', 'help', 'judge', 'movie', 'number',
             'free', 'random', 'let', 'time', 'discussion', 'wikipedia', 'either', 'actually'
                                                                                   'understand', 'mention', 'relevant',
             'copyright', 'welcome', 'discussion', 'evidence', 'kind', 'list', 'stop'
                                                                               'problem', 'work', 'also', 'probably',
             'think', 'else', 'another', 'admin', 'give', 'write', 'read',
             'would', 'notice', 'make', 'no', 'way', 'name', 'post', 'long', 'good', 'may', 'delete', 'something',
             'vandalism', 'argument', 'perhaps', 'year', 'username', 'email', 'bad', 'enjoy', 'thanks', 'article',
             'people', 'know', 'case', 'even', 'new', 'sure', 'say', 'claim', 'best', 'wrong', 'right', 'place',
             'get', 'able', 'link', 'need', 'use', 'wrong', 'add', 'delete', 'person', 'talk', 'keep', 'image', 'well',
             'edit', 'stop', 'many', 'please', 'include', 'message', 'see', 'top', 'information', 'check', 'go',
             'try', 'one', 'second', 'page', 'find', 'instead', 'correct', 'book', 'book', 'someone', 'might',
             'nothing',
             'read', 'anything', 'start', 'start', 'rather', 'agree', 'source', 'reason', 'history', 'enough', 'call',
             ]

print('extracted_words:', len(extracted_words))
print('drop:', len(drop))
print('drop_50:', len(drop_50))
print('important:', len(important))
data = pd.read_csv('../../inputs/pro_train.csv')
test = pd.read_csv('../../inputs/pro_test.csv')


def is_word(text, target_word):
    i = 0
    for word in text:
        if word == target_word:
            i += 1
    return i


print(data.shape)

for i, word in enumerate(extracted_words, start=1):
    if word not in drop:
        data[str(i)] = data['comment_text'].apply(lambda x: is_word(x, word))
        test[str(i)] = test['comment_text'].apply(lambda x: is_word(x, word))

print(data.shape)
print(test.shape)



data.to_csv('../inputs/pro_train_FE_drop.csv', index=False)
test.to_csv('../inputs/pro_test_FE_drop.csv', index=False)
print('done')
