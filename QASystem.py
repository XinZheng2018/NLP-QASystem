import nltk
import spacy
import re


path = './training/qadata/questions.txt'
stop_words = set(nltk.corpus.stopwords.words('english'))
print(stop_words)
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
    print(sim_list)

        # if temp_sim > max_similariy:
        #     max_similariy=temp_sim
        #     backpointer = i
        # i+=1
    #copy_sim_list = copy.deepcopy(sim_list)
    #res = sorted(copy_sim_list, key=lambda x: x, reverse=True)[:N]
    #same value ?"????
    #top_n_list = [bow[sim_list.index(x)]for x in res]
    print("the top n indicies is ")
    x= np.sort(sim_list)[-N:]
    #x.reverse()
    # print(x)
    top_n_indices = np.argsort(sim_list)[-N:]
    result = top_n_indices.tolist()
    result.reverse()
    return result


question = preprocessing_question(path)
candidate_passage, voc_list = document_sep(filename)
bow,question_vectors = vectorize(candidate_passage,voc_list,question)
top_n_indices = compute_similarity_find_max(bow,question_vectors, 10)



for index in top_n_indices:
    print(candidate_passage[index])


# add pos tag for questions
question_with_tag = {}
for ques_number in question:
    question_with_tag[ques_number] = nltk.pos_tag(question[ques_number])
#print(question_with_tag)

# dicide answer types based on different questions
answer_key = {}
nl = False
spa = False
for num in question_with_tag:
    quest = question_with_tag[num]
    for i in range(0,len(quest)):
            if re.search('Where',quest[i][0], flags=re.IGNORECASE):
                answer_key[num] = ["ORG", "COUNTRIES"]
                spa = True
                break
            elif re.search('When',quest[i][0], flags=re.IGNORECASE):
                answer_key[num] = ["DATE", "TIME"]
                spa = True
                break
            elif re.search('Who',quest[i][0], flags=re.IGNORECASE):
                answer_key[num] = ["PERSON", "ORG"]
                spa = True
                break
            elif re.search('What',quest[i][0], flags=re.IGNORECASE):
                nl = True
                if quest[i+1][1] == "NN" or quest[i+1][1] == "NNS" or  quest[i+1][1] == "VB" or quest[i+1][1] == "VBD" or \
                        quest[i+1][1] == "VBZ" or quest[i+1][1] == "VBN" or quest[i+1][1] == "JJ" or \
                        quest[i+1][1] == "JJR" or quest[i+1][1] == "DT":
                    answer_key[num] = ["NNP", "NN", "NNS"]
                    if quest[i+1][0] == "continent" or quest[i+1][0] == "nationality" or quest[i+1][0] == "city" or quest[i+1][0] == "province":
                        nl = False
                        spa = True
                        answer_key[num] = ["ORG", "COUNTRIES"]
                        break
                    elif quest[i+1][0] == "population":
                        answer_key[num] = ["CD"]
                        break
                    elif quest[i+1][0] == "year":
                        nl = False
                        spa = True
                        answer_key[num] = ["DATE", "TIME"]
                        break
                    elif quest[i+1][0] == "zip":
                        answer_key[num] = ["CD"]
                        break
                    break
                else:
                    answer_key[num] = ["UNDEFINED"]
                    break
            elif quest[0][1] == "IN":
                nl = True
                answer_key[num] = ["VB", "VBZ", "VBD", "VBN"]
                if quest[1][1] == "JJ":
                    answer_key[num] = ["NNP", "NN", "NNS"]
                break
            elif quest[0][1] == "NN" or quest[0][1] == "NNP":
                nl = True
                answer_key[num] = ["NNP", "NN", "NNS"]
                if quest[1][0] == "city":
                    nl = False
                    spa = True
                    answer_key[num] = ["ORG", "COUNTRIES"]
                break
            elif quest[0][0] == "How":
                nl = True
                answer_key[num] = ["NNP", "NN", "NNS"]
                if quest[1][0] == "many":
                    nl = True
                    answer_key[num] = ["CD"]
                    break
            elif quest[0][1] == "MD":
                nl = True
                answer_key[num] = ["NNP", "NN", "NNS"]
                break
            else:
                nl = True
                answer_key[num] = ["UNDEFINED"]
print(answer_key)
print(question_with_tag)
print(candidate_passage)



passage_with_label = []

# add tags using nltk
for i in top_n_indices:
    passage = nltk.pos_tag(candidate_passage[i])
    print(passage)
    ne_tree = nltk.ne_chunk(passage)
    for child in ne_tree:
        if type(child) == nltk.tree.Tree:
            ''.join(x[0] for x in child.leaves())
    # pattern = 'NP: {<NNP.*>*}'
    # pattern = 'NP: {<NNP.*><VBD|VB|VBZ|VBN|VBG><DT>?<NNP>}'
    pattern = 'NP: {<NNP.*><VB|VBD|VBZ|VBN|VBG><NNP.*>?<DT>?<NNP.*>}'
    np_parser = nltk.RegexpParser(pattern)
    np_parser.parse(passage)
    t = np_parser.parse(passage)
    print(t)

# add tags using spacy
import spacy
sp = spacy.load('en_core_web_sm')

for i in top_n_indices:
    passage_str =' '.join([str(elem) for elem in candidate_passage[i]])
    passage = sp(passage_str)
    passage_with_label.append(passage)

answer_list = set([])

for passage in passage_with_label:
    for entity in passage.ents:
        print(entity.text + ' - ' + entity.label_ + ' - ' + str(spacy.explain(entity.label_)))
        if entity.label_ == "PERSON" or entity.label == "ORG":
            answer = entity.text
            answer_list.add(answer)
print(answer_list)


#print(len(question))




