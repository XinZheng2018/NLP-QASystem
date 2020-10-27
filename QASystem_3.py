import nltk
import spacy
import re

from gensim import corpora

#new location  features
path = './training/qadata/questions.txt'
stop_words = set(nltk.corpus.stopwords.words('english'))
kept_words = ['when','what','why','who','where','which','whom','how']
for ele in kept_words:
    stop_words.remove(ele)
kept_words.append("name")
def preprocessing_question(filename):
    """
    #TODO SENTENCE TOKENIZE USE WORD2VEC TO FIND VECTORIZE EACH SENTENCE AND FIND THE TOP10 MOST
    SIMILAR ONE FOR THE CANDIDATE PASSAGE

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
        try:
            while True:
                current_line = next(file)
                if "Number" in current_line:
                    word_list = current_line.split(": ")
                    q_num = int(word_list[1])
                else:
                    temp=[]
                    word_tokens = tokenizer.tokenize(current_line)
                    filter_sent = [word for word in word_tokens if not word in stop_words]
                    temp.append(filter_sent)
                    question_training[q_num] = temp
                    question_training[q_num].append(current_line)
                    next(file)
        except StopIteration:
            return question_training

    #return question_training

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
               'DAY','MONTH','PG','COL','PUBYEAR','REGION','FEATURE','STATE','WORD','CT','DATELINE','COPYRGHT','LIMLEN','LANGUAGE','FILEID','FIRST','SECOND',
                        'HEAD','BYLINE','HL','DOCNO']
for ele in new_added_stop_words:
    stop_words.add(ele)



#test_filename =
def chunks(list, n):
    final_list =[]

    for i in range(0, len(list), n):
        temp_list = list[i:i+n]
        final_list.append(temp_list)

    return final_list
#todo remove all book information
#todo think about different features to find candidate passages
def document_sep(filename,question,id,n=20,num_chunks=20):
    rank = 0
    doc = []
    candidate_passage = []
    # bow = set()
    # raw = open(filename, 'r', encoding='latin-1').read()
    # sentences = nltk.sent_tokenize(raw)
    # tokenizer = nltk.RegexpTokenizer(r'\w+')
    # filtered_sentences = []
    # for sent in sentences:
    #     if sent.startswith("Qid:"):
    #         continue
    #     elif sent.endswith("</DOC>"):
    #         continue
    #     else:
    #         temp_list = sent.split()
    #         for ele in temp_list:
    #             if ele.startswith("<"):
    #                 temp_list.remove(ele)
    #         new_sent = " ".join(temp_list)
    #         filter_punct_sent = re.sub('[^\w\s]', '', new_sent)
    #         filtered_sentences.append(tokenizer.tokenize(filter_punct_sent))
    # frequency = defaultdict(int)
    # for text in filtered_sentences:
    #     for token in text:
    #         frequency[token] += 1
    # dictionary = corpora.Dictionary(filtered_sentences)
    # corpus = [dictionary.doc2bow(text)for text in filtered_sentences]
    # from gensim import models
    # lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=2)
    # current_question = question[id][0]
    # vec_bow = dictionary.doc2bow(current_question)
    # vec_lsi = lsi[vec_bow]
    # from gensim import similarities
    # index = similarities.MatrixSimilarity(lsi[corpus])
    # index.save('/tmp/deerwester.index')
    # index = similarities.MatrixSimilarity.load('/tmp/deerwester.index')
    # sims = index[vec_lsi]
    # sims = sorted(enumerate(sims), key=lambda item: -item[1])
    # i = 0
    # top_n_passages = []
    # for i, s in enumerate(sims):
    #     if i<n:
    #         top_n_passages.append(filtered_sentences[i])
    #         #print(s, filtered_sentences[i])
    #     else:
    #         break
    #     i+=1
    # return top_n_passages






    # dict_document_tokenblocks ={}
    with open(filename, 'r', encoding='latin-1') as f:
        #large_scale_tokenizer = nltk.RegexpTokenizer(r'\d+,?\d+|\s\w+|\w+\s')
        large_scale_tokenizer = nltk.RegexpTokenizer(r'\d+\smillion|\d+,?\d+|\w+')
        start_tag = ("<DOCNO>", "<DOCID>", "<FILEID>", "<FIRST>", "<SECOND>", "<HEAD>",
                     "<NOTE>", "<BYLINE>", "<DATELINE>", "<DATE>", "<SECTION>",
                     "<LENGTH>", "<HEADLINE>", "<ACCESS>", "<CAPTION>", "<DESCRIPT>",
                     "<LEADPARA>", "<MEMO>", "<COUNTRY>", "<CITY>", "<EDITION>", "<CODE>",
                     "<NAME>", "<PUBDATE>", "<DAY>", "<MONTH>", "<PG.COL>", "<PUBYEAR>",
                     "<REGION>", "<FEATURE>", "<STATE>", "<WORD.CT>", "<COPYRGHT>",
                     "<LIMLEN>", "<LANGUAGE>", "<EDITION>", "<DNO>", "<TYPE>", "<SUBJECT>", "<HL>","<GRAPHIC>")
        end_tag = ("</DOCNO>", "</DOCID>", "</FILEID>", "</FIRST>", "</SECOND>", "</HEAD>",
                   "</NOTE>", "</BYLINE>", "</DATELINE>", "</DATE>", "</SECTION>",
                   "</LENGTH>", "</HEADLINE>", "</ACCESS>", "</CAPTION>", "</DESCRIPT>",
                   "</LEADPARA>", "</MEMO>", "</COUNTRY>", "</CITY>", "</EDITION>", "</CODE>",
                   "</NAME>", "</PUBDATE>", "</DAY>", "</MONTH>", "</PG.COL>", "</PUBYEAR>",
                   "</REGION>", "</FEATURE>", "</STATE>", "</WORD.CT>", "</COPYRGHT>",
                   "</LIMLEN>", "</LANGUAGE>", "</EDITION>", "</DNO>", "</TYPE>", "</SUBJECT>", "</HL>","</GRAPHIC>")
        try:
            current_line = next(f)
            while True:
                current_line = next(f)
                #print(current_line)
                if "Rank" in current_line:

                    next(f)

                else:

                    if current_line.startswith(start_tag):

                        if any(s in current_line for s in end_tag):
                            continue
                        else:
                            while True:
                                current_line = next(f)

                                if any(s in current_line for s in end_tag):
                                    break
                                else:
                                    continue
                    else:

                        current_tokens = large_scale_tokenizer.tokenize(current_line)
                        filter_token = [word for word in current_tokens if not word in stop_words]
                        # for ele in filter_token:
                        #     bow.add(ele)
                        doc.extend(filter_token)
                        #current_line = next(f)

                   # while "Rank" not in current_line:


                    # candidate_passage = chunks(doc,n)
                    # dict_document_tokenblocks[current_docno]=candidate_passage
        except StopIteration:
            # print('EOF!')
            filtered_sentences = chunks(doc, num_chunks)
            dictionary = corpora.Dictionary(filtered_sentences)
            corpus = [dictionary.doc2bow(text)for text in filtered_sentences]
            from gensim import models
            lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=2)
            current_question = question[id][0]
            vec_bow = dictionary.doc2bow(current_question)
            vec_lsi = lsi[vec_bow]
            from gensim import similarities
            index = similarities.MatrixSimilarity(lsi[corpus])
            index.save('/tmp/deerwester.index')
            index = similarities.MatrixSimilarity.load('/tmp/deerwester.index')
            sims = index[vec_lsi]
            sims = sorted(enumerate(sims), key=lambda item: -item[1])
            i = 0
            top_n_passages = []
            for i, s in enumerate(sims):
                if i<n:
                    top_n_passages.append(filtered_sentences[i])
                    #print(s, filtered_sentences[i])
                else:
                    break
                i+=1
            return top_n_passages
            # voc_list = sorted(list(bow))
            # #dict_document_tokenblocks[current_docno] = candidate_passage
            # return (candidate_passage,voc_list)


# def vectorize(candidate_passages,voc_list,question, question_num):
#
#     bow = []
#     for p in candidate_passages:
#         temp = [1 if x in p else 0 for x in voc_list]
#         #index_list = [if x in p for x in voc_list]
#         bow.append(temp)
#
#
#     question_vectors = [1 if x in question[question_num][0] else 0 for x in voc_list]
#
#     return (bow,question_vectors)
#
# import math
# import numpy as np
# import copy
# def compute_similarity_find_max(bow,question_vectors,N):
#
#     i = 0
#     sim_list = []
#     for vectors in bow:
#         temp_sim = np.dot(vectors,question_vectors)
#         sim_list.append(temp_sim)
#
#     top_n_indices = np.argsort(sim_list)[-N:]
#     result = top_n_indices.tolist()
#     result.reverse()
#     return result
#
#
#
# def get_top_n_passages(candidate_passage,top_n_indices):
#     top_n_passages=[]
#     for num in top_n_indices:
#         top_n_passages.append(candidate_passage[num])
#     return top_n_passages
#


# for index in top_n_indices:

#     print(candidate_passage[index])


import copy
#add pos tag for questions
def get_question_with_tag(question):
    # new_question = copy.deepcopy(question)
    question_with_tag = {}
    for ques_number in question.keys():
        question_with_tag[ques_number] = nltk.pos_tag(question[ques_number][0])
    return question_with_tag

#todo simplity code currently thinking: dictionary
# dicide answer types based on different questions
def question_extraction(question, question_with_tag):
    # answer_key = {}
    # np = {}
    answer_key = []
    np=True
    for num in question_with_tag.keys():
        quest = question_with_tag[num]
        for i in range(0,len(quest)):
                if re.search('Where',quest[i][0], flags=re.IGNORECASE):
                    answer_key = ["ORGANIZATION", "GPE"]
                    np= False
                    #question[num][0].remove("Where")
                    break
                elif re.search('name',quest[i][0].lower(),flags=re.IGNORECASE):

                    answer_key = ['NNP_Pattern']
                    np = True

                    break
                elif re.search('When',quest[i][0], flags=re.IGNORECASE):
                    answer_key = ["CD"]
                    np= True
                    break
                elif re.search('Who',quest[i][0], flags=re.IGNORECASE):
                    answer_key= ["PERSON"]
                    np = False
                    break
                elif re.search('what',quest[i][0].lower(), flags=re.IGNORECASE):
                    # print("hi")
                    #np = True
                    # if quest[i+1][1] == "NN" or quest[i+1][1] == "NNS" or  quest[i+1][1] == "VB" or quest[i+1][1] == "VBD" or \
                    #         quest[i+1][1] == "VBZ" or quest[i+1][1] == "VBN" or quest[i+1][1] == "JJ" or \
                    #         quest[i+1][1] == "JJR" or quest[i+1][1] == "DT":

                    if quest[i+1][0] == "continent" or quest[i+1][0] == "nationality" or quest[i+1][0] == "city" or quest[i+1][0] == "province":
                        np = False
                        answer_key = ["ORGANIZATION", "GPE"]
                        break
                    elif quest[i+1][0] == "name":
                        answer_key = ['NNP_Pattern']
                        np = True
                        break
                    elif quest[i+1][0] == "population":
                        answer_key = ["CD"]
                        np = True
                        break
                    elif quest[i+1][0] == "year":
                        np = True
                        answer_key = ["CD"]
                        break
                    elif quest[i+1][0] == "zip":
                        answer_key = ["CD"]
                        np = True
                        break
                        #break
                    else:
                        answer_key = ["NNP","NN","NNS"] #TODO NP
                        np = True
                        break
                elif quest[0][1] == "IN":
                    np = True
                    if quest[1][1] == "JJ":
                        answer_key = ["NNP", 'NN'] #TODO NP
                        break
                    answer_key = ["VB", "VBZ", "VBD", "VBN"]
                    break
                elif quest[0][1] == "NN" or quest[0][1] == "NNP":
                    #print(quest)
                    if quest[1][0] == "city":
                        np = False
                        answer_key = ["GPE"]
                    break
                    np = True
                    answer_key = ["NNP", 'NN','NNS']#TODO NP
                    break
                elif quest[0][0] == "How":
                    if quest[1][0] == "many":
                        np = True
                        answer_key = ["CD"]
                        break
                    np = True
                    answer_key = ["NNP", 'NN','NNS'] #TODO NP
                    break
                elif quest[0][1] == "MD":
                    np = True
                    answer_key = ["NNP", 'NN','NNS'] #TODO NP
                    break
                else:
                    np = True
                    answer_key = ["NNP", 'NN','NNS'] #TODO NP
                    break
        #print(answer_key[num])
        # print("current num is " +str(num))
        # print(answer_key)
        question[num].append(answer_key)
        question[num].append(np)
        # question[num][0].remove(ele for ele in )
        #todo check data structure?
        #to elimiate nl and answerkey
    return question
def clean_question(question):
    new_ques = copy.deepcopy(question)
    for quest in new_ques.values():
        for ele in quest[0]:
            #print(quest[0])
            if ele.lower() in kept_words:
                quest[0].remove(ele)
    return new_ques
#passage_with_label = []
import math
#def generate_candidate_passages_each_question(question_with_tag, )
def find_location (list_index):
    return sum(list_index)/len(list_index)

def find_closest_distance_to_any_keyword(loc_ans, list_index):
    distance = math.inf
    for ele in list_index:
        current_distance = abs(loc_ans-ele)
        if distance > current_distance:
            distance = current_distance
    return distance

def answer_extract(top_n_passages, single_question):
    """

    @param top_n_passages:
    @type top_n_passages:
    @param single_question:
    @type single_question:list of lists element 1: all key words, element two: answer type, element three: nl or not
    @return: answer_list, list of answer_type(0/1), list of locality
    @rtype:
    """

    answer_list = []
    list_answer_type =[] #0 means not match answer type of question, 1 means matching answer type of question 1
    list_of_locality = [] #the relevant distance between key words in the candidate passage and the answer of the key words
    np = single_question[2]
    tag_looing_for = single_question[1]
    keyword_list = single_question[0]

    for ele in top_n_passages:

        #calculate location
        i = 0
        list_index = []
        find_tag = False
        #find the location of question key word
        #print(ele)
        for word in ele:
            if word in keyword_list:
                #print("the word is " +word+" and the current index is " + str(i))
                # print(keyword_list)
                list_index.append(i)

            i+=1

        #todo what if no key word?
        if len(list_index) == 0:
            loc_keyword = 0
        else:
            loc_keyword = find_location(list_index)
            #print("the location of keyword is " + str(loc_keyword))
        #todo, deal with undefined
        #find pos tag
        if np:

            passage_with_tag = nltk.pos_tag(ele)
            tracker = 0
            if tag_looing_for[0] == "CD":
                for token in passage_with_tag:
                    if token[1] in tag_looing_for:
                        # todo: check assumption here only one word ?is np really bad?
                        # todo after testing performance change to token[0]
                        if token[0] in keyword_list:
                            continue

                        answer_list.append(token[0])
                        list_answer_type.append(1.0)
                        #print answer loc

                        loc_answer = tracker
                        distance = find_closest_distance_to_any_keyword(loc_answer, list_index)
                        #print("current anewser is " + str(token[0]) + "current location is " + str(tracker) + "the distance is " + str(distance))
                        list_of_locality.append(distance)
                        find_tag = True
                    tracker += 1


            elif tag_looing_for[0] == "NNP_Pattern":

                np_parser_list = []
                t_list = []
                # pattern_list = ['NP: {<NNP|NN.*>*}', 'NP: {<DT>?<JJ|PR.*>*<NNP>+}', 'NP: {<NNP.*>*<VB|VBD|VBZ|VBN|VBG><NNP.*>?<DT>?<NNP.*>}']
                # pattern = 'NP: {<DT>?<JJ|PR.*>*<NNP>+}'
                # pattern_list = ['NP: {<DT>?<JJ|PR.*>*<NNP>*}','NP: {<DT>?<NNP>*}']
                pattern_list = ['NP: {<DT>?<JJ|PR.*>*<NNP>+}']
                for pattern in pattern_list:
                    np_parser_list.append(nltk.RegexpParser(pattern))

                for np_parser in np_parser_list:
                    t_list.append(np_parser.parse(nltk.pos_tag(ele)))

                for t in t_list:
                    tracker = 0
                    for child in t:
                        if type(child) == nltk.tree.Tree:
                            # todo simplify
                            current_list = []
                            for x in child.leaves():
                                if x[0] in keyword_list:
                                    current_list = []
                                    break
                                else:
                                    current_list.append(x[0])


                            if len(current_list)>0:
                                x = ' '.join(current_list)
                                answer_list.append(x)
                                num_token = len(child.leaves())
                                list_answer_type.append(1.0)
                                loc_answer = (tracker + tracker + num_token - 1) / 2
                                # print("current anewser is " + str(x) + "current location is " + str(tracker) + "the distance is " + str(abs(loc_keyword - loc_answer)))
                                distance = find_closest_distance_to_any_keyword(loc_answer, list_index)
                                list_of_locality.append(distance)
                                tracker += num_token
                                find_tag = True
                        else:
                            tracker += 1

            else:
                #for token in passage_with_tag:

                np_parser_list = []
                t_list = []
                # pattern_list = ['NP: {<NNP|NN.*>*}', 'NP: {<DT>?<JJ|PR.*>*<NNP>+}', 'NP: {<NNP.*>*<VB|VBD|VBZ|VBN|VBG><NNP.*>?<DT>?<NNP.*>}']
                # pattern = 'NP: {<DT>?<JJ|PR.*>*<NNP>+}'
                # pattern_list = ['NP: {<DT>?<JJ|PR.*>*<NN|NNP|NNS>}']
                pattern_list = ['NP: {<DT>?<JJ|PR.*>*<NN|NNP|NNS>+}', 'NP: {<DT>?<NN|NNP|NNS>+}']
                # pattern = 'NP: {<NNP.*><VBD|VB|VBZ|VBN|VBG><DT>?<NNP>}'
                # pattern = 'NP: {<NNP.*>*<VB|VBD|VBZ|VBN|VBG><NNP.*>?<DT>?<NNP.*>}'
                for pattern in pattern_list:
                    np_parser_list.append(nltk.RegexpParser(pattern))
                for np_parser in np_parser_list:
                    t_list.append(np_parser.parse(nltk.pos_tag(ele)))

                for t in t_list:
                    tracker =0
                    for child in t:
                        if type(child) == nltk.tree.Tree:
                            # todo simplify

                            x = ' '.join(x[0] for x in child.leaves())
                            if x in keyword_list:
                                continue
                            #if x not in answer_list:
                            answer_list.append(x)
                            num_token = len(child.leaves())
                            list_answer_type.append(1.0)
                            loc_answer = (tracker + tracker + num_token-1) / 2
                            #print("current anewser is " + str(x) + "current location is " + str(tracker) + "the distance is " + str(abs(loc_keyword - loc_answer)))
                            distance =find_closest_distance_to_any_keyword(loc_answer,list_index)
                            list_of_locality.append(distance)
                            tracker += num_token
                            find_tag = True
                        else:
                            tracker += 1
                #todo: if NP : then directly go to find NP pattern
                # if token[1] in tag_looing_for:
                #     # todo: check assumption here only one word ?is np really bad?
                #     #todo after testing performance change to token[0]
                #     answer_list.append(token)
                #     list_answer_type.append(1.0)
                #     list_of_locality.append(abs(loc_keyword - tracker))
                #     find_tag = True
                # tracker += 1
        else:
            #find ne tag!
            token_pos_list = nltk.pos_tag(ele)
            passage_with_tag = nltk.ne_chunk(token_pos_list)
            tracker = 0
            for child in passage_with_tag:

                if type(child) == nltk.tree.Tree and child.label() in tag_looing_for:


                    ans = ' '.join(x[0] for x in child.leaves())
                    # if ans not in answer_list:
                    if ans in keyword_list:
                        continue
                    answer_list.append(ans)
                    num_token = len(child.leaves())
                    loc_answer = (tracker + tracker + num_token-1) / 2
                    list_answer_type.append(1.0)

                    distance = find_closest_distance_to_any_keyword(loc_answer, list_index)
                    list_of_locality.append(distance)

                    tracker += num_token
                    find_tag=True
                else:
                    tracker +=1
        #BACK UP PLAN NP
        if find_tag==False:
            #print("enter back up ")
            #todo what's the pattern we are looking for?
            pattern ='NP: {<DT>?<JJ|PR.*>*<NN|NNP|NNS>}'
            #pattern = ['NP: {<DT>?<JJ|PR.*>*<NN|NNP|NNS>+}', 'NP: {<DT>?<NN|NNP|NNS>+}']
            np_parser = nltk.RegexpParser(pattern)

            t = np_parser.parse(nltk.pos_tag(ele))
            tracker = 0
            for child in t:
                if type(child) == nltk.tree.Tree:
                    #todo simplify
                    x = ' '.join(x[0] for x in child.leaves())
                    if x in keyword_list:
                        continue
                    answer_list.append(x)
                    num_token = len(child.leaves())
                    list_answer_type.append(0.0)
                    loc_answer = (tracker+tracker+num_token-1)/2
                    distance = find_closest_distance_to_any_keyword(loc_answer, list_index)
                    list_of_locality.append(distance)
                    tracker+= num_token
                else:
                    tracker+=1
    #print(list_of_locality)
    return answer_list,list_answer_type,list_of_locality
import numpy as np
def ranking(list_answer_type, list_of_locality,N=30):
    weighted_ans_type = [ele*0.9 for ele in list_answer_type]
    weighted_locality = [ele*0.1 for ele in list_of_locality]
    weighted_value = np.subtract(weighted_ans_type,weighted_locality)
    top_n_indices = np.argsort(weighted_value.tolist())[-N:]
    result = top_n_indices.tolist()
    result.reverse()
    return result

#todo :evaluation_results = 0.9*list_answer_type + 0.1* list_of_locality
# add tags using spacy

def write_file(answer):
    f = open("prediction_file.txt", "w")
    for elem in answer:
        f.write("qid " + str(elem))
        f.write("\n")
        for ans in answer[elem]:
            f.write(ans)
            f.write("\n")
    f.close()

if __name__ == "__main__":
    #todo 8 problem

    question = preprocessing_question(path)

    n_list = list(question.keys())

    # n_list = [0]

    answer = {}
    for n in n_list:
        # print(n)
        filename = './training/topdocs/top_docs.' + str(n)
        #filename = './training/top_docs.' + str(n)
        filter_question = clean_question(question)
        top_n_passages = document_sep(filename,filter_question,n,30,40)
        # bow,question_vectors = vectorize(candidate_passage,voc_list,question, n)
        #
        # top_n_indices = compute_similarity_find_max(bow,question_vectors, 10)
        # top_n_passages = get_top_n_passages(candidate_passage,top_n_indices)
        # for ele in top_n_passages:
        #     print(ele)

        f = open("candidatepassage.txt", "w")
        for elem in top_n_passages:
            for ele in elem:
                f.write(ele)
                f.write("\t")
            f.write("\n")
        f.close()

        question_with_tag = get_question_with_tag(question)
        #print(question_with_tag)
        #todo check pointer stuff

        full_question_info = question_extraction(filter_question,question_with_tag)

        anwer_list_0, list_answer_type_0, list_locality_0 = answer_extract(top_n_passages,full_question_info[n])

        ranking_top_n_indices = ranking(list_answer_type_0,list_locality_0,10)
        final_answer = [anwer_list_0[ele] for ele in ranking_top_n_indices]
        # print()
        # print("final answer is: ")
        # for ans in final_answer:
        #     print(ans)
        answer[n] = final_answer
    write_file(answer)



