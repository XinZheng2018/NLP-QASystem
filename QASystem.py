import nltk
import spacy
import re


from sklearn.preprocessing import Binarizer
path = './training/qadata/questions.txt'
stop_words = set(nltk.corpus.stopwords.words('english'))
def preprocessing_question(filename):
    """
    Preprocessing the question file
    @param filename:
    @type filename:
    @return:
    @rtype:
    """
    tokenizer = nltk.RegexpTokenizer(r'\w+')
    #question_training = []
    question_training = {}
    q_num = 0
    with open(filename, 'rt',encoding = 'utf-8-sig') as file:
        for line in file:
            if "Number" in line:
                for word in line.split():
                    if word.isdigit():
                        q_num = int(word)
            else:
                # current_line = next(file)
            #if line[0]!='Number':
                word_tokens = tokenizer.tokenize(line)
                filter_sent = [word for word in word_tokens if not word in stop_words]
                question_training[q_num] = filter_sent
                next(file)
    return question_training

# a = preprocessing_question(path)
# print(a)

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
new_added_stop_words = ['DATE','SECTION','P','LENGTH','HEADLINE','BYLINE','TEXT','DNO','TYPE','SUBJECT','DOC','DOCID','COUNTRY','EDITION','NAME','PUBDATE',
               'DAY','MONTH','PG','COL','PUBYEAR','REGION','FEATURE','STATE','WORD','CT','DATELINE','COPYRIGHT','LIMLEN','LANGUAGE','FILEID','FIRST','SECOND',
                        'HEAD','BYLINE','HL']
for ele in new_added_stop_words:
    stop_words.add(ele)

filename = './training/topdocs/top_docs.3'
#test_filename =
def chunks(list, n):
    final_list =[]
    print("start chunks")
    for i in range(0, len(list), n):
        temp_list = list[i:i+n]
        final_list.append(temp_list)
        print("in the chunk")
    return final_list

def document_sep(filename,n=20):
    rank = 0
    doc = []
    candidate_passage = []
    bow = set()

    #dict_document_tokenblocks ={}
    with open(filename, 'r', encoding='utf-8-sig') as f:
        #large_scale_tokenizer = nltk.RegexpTokenizer(r'\d+,?\d+|\s\w+|\w+\s')
        large_scale_tokenizer = nltk.RegexpTokenizer(r'\d+\smillion|\d+,?\d+|\w+')
        try:
            current_line = next(f)
            while True:

                #print(current_line)
                if "Rank" in current_line:
                    print("line contains rank: ")
                    print(current_line)
                    # numbers = [int(word) for word in current_line.split() if word.isdigit()]
                    # if rank != numbers[1]:
                    #     doc = []
                    #     rank = numbers[1]
                    #skip <doc>
                    next(f)
                    next(f)
                    #current_line = next(f)
                    #current_docno = current_line.split()[1]
                    #dict_document_tokenblocks[current_docno] = []
                    current_line = next(f)


                    while "Rank" not in current_line:
                        current_tokens = large_scale_tokenizer.tokenize(current_line)
                        filter_token = [word for word in current_tokens if not word in stop_words]
                        for ele in filter_token:
                            bow.add(ele)
                        doc.extend(filter_token)
                        current_line=next(f)


                    # candidate_passage = chunks(doc,n)
                    # dict_document_tokenblocks[current_docno]=candidate_passage
        except StopIteration:
            print('EOF!')
            candidate_passage = chunks(doc, n)
            voc_list = sorted(list(bow))
            #dict_document_tokenblocks[current_docno] = candidate_passage
            return (candidate_passage,voc_list)


#result = document_sep(filename)
#print(result)
def vectorize(candidate_passages,voc_list,question):
    bow = []
    for p in candidate_passages:
        temp = [1 if x in p else 0 for x in voc_list]
        bow.append(temp)
    question_vectors = [1 if x in question[0] else 0 for x in voc_list]
        # transformer = Binarizer().fit_transform(p,voc_list)
        # print(transformer)
    return (bow,question_vectors)
import math
import numpy as np
import copy
def compute_similarity_find_max(bow,question_vectors,N):
    max_similariy = -math.inf
    #backpointer = 0
    i = 0
    sim_list = []
    for vectors in bow:
        temp_sim = np.dot(vectors,question_vectors)
        sim_list.append(temp_sim)

        # if temp_sim > max_similariy:
        #     max_similariy=temp_sim
        #     backpointer = i
        # i+=1
    #copy_sim_list = copy.deepcopy(sim_list)
    #res = sorted(copy_sim_list, key=lambda x: x, reverse=True)[:N]
    #same value ?"????
    #top_n_list = [bow[sim_list.index(x)]for x in res]
    top_n_indices = np.argsort(sim_list)[-N:]
    return top_n_indices

question = preprocessing_question(path)
candidate_passage, voc_list = document_sep(filename)
bow,question_vectors = vectorize(candidate_passage,voc_list,question)
top_n_indices = compute_similarity_find_max(bow,question_vectors, 10)

#print(bow)
for index in top_n_indices:
    print(candidate_passage[index])
    print(len(candidate_passage[index]))
#print(top_n_indices)


# use spacy to assign labels to each word in candidate passage
#sp = spacy.load('/opt/anaconda3/lib/python3.7/site-packages/en_core_web_sm')

passage_with_label = []

# for i in top_n_indices:
#     passage = sp(candidate_passage[i])
#     entity = [ent for ent in passage.ents]
#     passage_with_label.append(entity)
#
# for passage in passage_with_label:
#     for word in passage:
#         print(word.text + "-" + word.label_)

answer_key = []
# for query in question:
#     q_key = re.search('Where', query, flags=re.IGNORECASE)
#     if q_key:
#         answer_key = []
#         answer_key.append("PLACE")
#     q_key = re.search('When', query, flags=re.IGNORECASE)
#     if q_key:
#         answer_key = []
#         answer_key.append("DATE")
#         answer_key.append("TIME")
#     q_key = re.search('Who', query, flags=re.IGNORECASE)
#     if q_key:
#         answer_key = []
#         answer_key.append("PERSON")


