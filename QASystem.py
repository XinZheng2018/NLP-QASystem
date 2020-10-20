import nltk

path = './training/qadata/questions.txt'
stop_words = set(nltk.corpus.stopwords.words('english'))
def preprocessing(filename):
    """
    Preprocessing the question file
    @param filename:
    @type filename:
    @return:
    @rtype:
    """
    tokenizer = nltk.RegexpTokenizer(r'\w+')
    question_training = []
    with open(filename, 'rt',encoding = 'utf-8-sig') as file:
        for line in file:
            current_line = next(file)
            #if line[0]!='Number':
            word_tokens = tokenizer.tokenize(current_line)
            filter_sent = [word for word in word_tokens if not word in stop_words]
            question_training.append(filter_sent)
            next(file)
    return question_training

a = preprocessing(path)
print(a)

#todo: passage retrieval
"""
data structre:  a list of document for each qid , so it will be 50*20 dimension list [[[tokenblock1],[tokenblock2].....20 tokenblocks ],[].....50documents[]]
1. skip qid, read in as xmlfile
2. combine all sentences in <p></p> ( we can do tokenize then extend the list) current the list is only 1d
3. split into 20 blocks --> append each block to the list of document of token blocks 
4. create bag-of-word for each token block
5. computer  the dot similarity for each document mean while find the max number & related index 
6.return the most similar passage (list of 20 tokens)

"""



