import nltk
import spacy
import re


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
def document_sep(filename,n=20):
    rank = 0
    doc = []
    candidate_passage = []
    bow = set()

    #dict_document_tokenblocks ={}
    with open(filename, 'r', encoding='utf-8-sig') as f:
        #large_scale_tokenizer = nltk.RegexpTokenizer(r'\d+,?\d+|\s\w+|\w+\s')
        large_scale_tokenizer = nltk.RegexpTokenizer(r'\d+\smillion|\d+,?\d+|\w+')
        start_tag = ("<DOCNO>", "<DOCID>", "<FILEID>", "<FIRST>", "<SECOND>", "<HEAD>",
                     "<NOTE>", "<BYLINE>", "<DATELINE>", "<DATE>", "<SECTION>",
                     "<LENGTH>", "<HEADLINE>", "<ACCESS>", "<CAPTION>", "<DESCRIPT>",
                     "<LEADPARA>", "<MEMO>", "<COUNTRY>", "<CITY>", "<EDITION>", "<CODE>",
                     "<NAME>", "<PUBDATE>", "<DAY>", "<MONTH>", "<PG.COL>", "<PUBYEAR>",
                     "<REGION>", "<FEATURE>", "<STATE>", "<WORD.CT>", "<COPYRGHT>",
                     "<LIMLEN>", "<LANGUAGE>", "<EDITION>", "<DNO>", "<TYPE>", "<SUBJECT>", "<HL>")
        end_tag = ("</DOCNO>", "</DOCID>", "</FILEID>", "</FIRST>", "</SECOND>", "</HEAD>",
                   "</NOTE>", "</BYLINE>", "</DATELINE>", "</DATE>", "</SECTION>",
                   "</LENGTH>", "</HEADLINE>", "</ACCESS>", "</CAPTION>", "</DESCRIPT>",
                   "</LEADPARA>", "</MEMO>", "</COUNTRY>", "</CITY>", "</EDITION>", "</CODE>",
                   "</NAME>", "</PUBDATE>", "</DAY>", "</MONTH>", "</PG.COL>", "</PUBYEAR>",
                   "</REGION>", "</FEATURE>", "</STATE>", "</WORD.CT>", "</COPYRGHT>",
                   "</LIMLEN>", "</LANGUAGE>", "</EDITION>", "</DNO>", "</TYPE>", "</SUBJECT>", "</HL>")
        try:
            current_line = next(f)
            while True:

                #print(current_line)
                if "Rank" in current_line:
                    # print("line contains rank: ")
                    # print(current_line)
                    # numbers = [int(word) for word in current_line.split() if word.isdigit()]
                    # if rank != numbers[1]:
                    #     doc = []
                    #     rank = numbers[1]
                    #skip <doc>
                    # next(f)
                    next(f)
                    while True:


                    #current_line = next(f)
                    #current_docno = current_line.split()[1]
                    #dict_document_tokenblocks[current_docno] = []
                        current_line = next(f)
                        # if current_line.startswith("<DOC>"):
                        #     current_line = next(f)
                        if current_line.startswith(start_tag):
                            if any(s in current_line for s in end_tag):
                                continue
                            else:
                                while True:
                                    current_line = next(f)
                                    if any(s in current_line for s in end_tag):
                                        break
                        else:
                            break

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
            # print('EOF!')
            candidate_passage = chunks(doc, n)
            voc_list = sorted(list(bow))
            #dict_document_tokenblocks[current_docno] = candidate_passage
            return (candidate_passage,voc_list)


def vectorize(candidate_passages,voc_list,question, question_num):

    bow = []
    for p in candidate_passages:
        temp = [1 if x in p else 0 for x in voc_list]
        #index_list = [if x in p for x in voc_list]
        bow.append(temp)


    question_vectors = [1 if x in question[question_num] else 0 for x in voc_list]

    return (bow,question_vectors)

import math
import numpy as np
import copy
def compute_similarity_find_max(bow,question_vectors,N):

    i = 0
    sim_list = []
    for vectors in bow:
        temp_sim = np.dot(vectors,question_vectors)
        sim_list.append(temp_sim)

    top_n_indices = np.argsort(sim_list)[-N:]
    result = top_n_indices.tolist()
    result.reverse()
    return result



def get_top_n_passages(candidate_passage,top_n_indices):
    top_n_passages=[]
    for num in top_n_indices:
        top_n_passages.append(candidate_passage[num])
    return top_n_passages



# for index in top_n_indices:

#     print(candidate_passage[index])



#add pos tag for questions
def get_question_with_tag(question):
    question_with_tag = {}
    for ques_number in question.keys():
        question_with_tag[ques_number] = nltk.pos_tag(question[ques_number][0])
    # print("question is :")
    # print(question_with_tag)
    return question_with_tag


#todo simplity code currently thinking: dictionary
# dicide answer types based on different questions
def question_extraction(question, question_with_tag):
    # answer_key = {}
    # np = {}
    for num in question_with_tag.keys():
        quest = question_with_tag[num]
        for i in range(0,len(quest)):
                if re.search('Where',quest[i][0], flags=re.IGNORECASE):
                    answer_key = ["ORGANIZATION", "GPE"]
                    np= False
                    break
                elif re.search('When',quest[i][0], flags=re.IGNORECASE):
                    answer_key = ["DATE"]
                    np= False
                    break
                elif re.search('Who',quest[i][0], flags=re.IGNORECASE):
                    answer_key= ["PERSON"]
                    np = False
                    break
                elif re.search('What',quest[i][0], flags=re.IGNORECASE):
                    np = True
                    if quest[i+1][1] == "NN" or quest[i+1][1] == "NNS" or  quest[i+1][1] == "VB" or quest[i+1][1] == "VBD" or \
                            quest[i+1][1] == "VBZ" or quest[i+1][1] == "VBN" or quest[i+1][1] == "JJ" or \
                            quest[i+1][1] == "JJR" or quest[i+1][1] == "DT":
                        answer_key = ["NNP","NN"]
                        if quest[i+1][0] == "continent" or quest[i+1][0] == "nationality" or quest[i+1][0] == "city" or quest[i+1][0] == "province":
                            np = False
                            answer_key = ["ORGANIZATION", "GPE"]
                            break
                        elif quest[i+1][0] == "population":
                            answer_key = ["CD"]
                            np = True
                            break
                        elif quest[i+1][0] == "year":
                            np = False
                            answer_key = ["DATE"]
                            break
                        elif quest[i+1][0] == "zip":
                            answer_key = ["CD"]
                            np = True
                            break
                        break
                    else:
                        answer_key = ["UNDEFINED"] #TODO NP
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
                    if quest[1][0] == "city":
                        np = False
                        answer_key = ["GPE"]
                    break
                    np = True
                    answer_key = ["NNP", 'NN']#TODO NP
                    break
                elif quest[0][0] == "How":
                    if quest[1][0] == "many":
                        np = True
                        answer_key = ["CD"]
                        break
                    np = True
                    answer_key = ["NNP", 'NN'] #TODO NP
                    break
                elif quest[0][1] == "MD":
                    np = True
                    answer_key = ["NNP", 'NN'] #TODO NP
                    break
                else:
                    np = True
                    answer_key = ["UNDEFINED"] #TODO NP
        #print(answer_key[num])
        question[num].append(answer_key)
        question[num].append(np)
        #todo check data structure?
        #to elimiate nl and answerkey
    return question


#import spacy

passage_with_label = []
#def generate_candidate_passages_each_question(question_with_tag, )
def find_location (list_index):
    return sum(list_index)/len(list_index)

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
        for word in ele:
            if word in keyword_list:
                # print("the word is " +word)
                # print(keyword_list)
                list_index.append(i)
            i+=1

        #todo what if no key word?
        if len(list_index) == 0:
            loc_keyword = 0
        else:
            loc_keyword = find_location(list_index)
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
                        answer_list.append(token[0])
                        list_answer_type.append(1.0)
                        list_of_locality.append(abs(loc_keyword - tracker))
                        find_tag = True
                    tracker += 1


            else:
                #for token in passage_with_tag:

                np_parser_list = []
                t_list = []
                # pattern_list = ['NP: {<NNP|NN.*>*}', 'NP: {<DT>?<JJ|PR.*>*<NNP>+}', 'NP: {<NNP.*>*<VB|VBD|VBZ|VBN|VBG><NNP.*>?<DT>?<NNP.*>}']
                # pattern = 'NP: {<DT>?<JJ|PR.*>*<NNP>+}'
                pattern_list = ['NP: {<DT>?<JJ|PR.*>*<NNP>+}']
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
                            if x not in answer_list:
                                answer_list.append(x)
                                num_token = len(child.leaves())
                                list_answer_type.append(1.0)
                                loc_answer = (tracker + tracker + num_token-1) / 2
                                list_of_locality.append(abs(loc_answer - loc_answer))
                                tracker += num_token
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
                    if ans not in answer_list:
                        answer_list.append(ans)
                        num_token = len(child.leaves())
                        loc_answer = (tracker + tracker + num_token-1) / 2
                        list_answer_type.append(1.0)
                        list_of_locality.append(abs(loc_keyword-loc_answer))
                        tracker += num_token
                        find_tag=True
                else:
                    tracker +=1
        #BACK UP PLAN NP
        if find_tag==False:
            #todo what's the pattern we are looking for?
            pattern ='NP: {<DT>?<JJ|PR.*>*<NN|NNS>}'
            #pattern = 'NP: {<NNP.*>*}'
            # pattern = 'NP: {<NNP.*><VBD|VB|VBZ|VBN|VBG><DT>?<NNP>}'
            # pattern = 'NP: {<NNP.*><VB|VBD|VBZ|VBN|VBG><NNP.*>?<DT>?<NNP.*>}'
            np_parser = nltk.RegexpParser(pattern)

            t = np_parser.parse(nltk.pos_tag(ele))
            tracker = 0
            for child in t:
                if type(child) == nltk.tree.Tree:
                    #todo simplify
                    x = ' '.join(x[0] for x in child.leaves())
                    answer_list.append(x)
                    num_token = len(child.leaves())
                    list_answer_type.append(0.0)
                    loc_answer = (tracker+tracker+num_token-1)/2
                    list_of_locality.append(abs(loc_answer-loc_answer))
                    tracker+= num_token
                else:
                    tracker+=1

    return answer_list,list_answer_type,list_of_locality
import numpy as np
def ranking(list_answer_type, list_of_locality,N=10):
    weighted_ans_type = [ele*0.9 for ele in list_answer_type]
    weighted_locality = [ele*0.1 for ele in list_of_locality]
    weighted_value = np.add(weighted_ans_type,weighted_locality)
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
    # n_list = list(question.keys())
    #     # n_list.remove(8)
    #     # n_list.remove(14)
    n_list = [17]
    answer = {}
    for n in n_list:
        print(n)
        filename = './training/topdocs/top_docs.' + str(n)
        candidate_passage, voc_list = document_sep(filename,40)
        bow,question_vectors = vectorize(candidate_passage,voc_list,question, 10)

        top_n_indices = compute_similarity_find_max(bow,question_vectors, 10)
        top_n_passages = get_top_n_passages(candidate_passage,top_n_indices)

        f = open("candidatepassage.txt", "w")
        for elem in top_n_passages:
            for ele in elem:
                f.write(ele)
                f.write("\t")
            f.write("\n")
        f.close()

        question_with_tag = get_question_with_tag(question)
        #todo check pointer stuff
        full_question_info = question_extraction(question,question_with_tag)

        anwer_list_0, list_answer_type_0, list_locality_0 = answer_extract(top_n_passages,full_question_info[n])
        print("answer_list is: ")
        for ele in anwer_list_0:
            print(ele)
        ranking_top_n_indices = ranking(list_answer_type_0,list_locality_0)
        final_answer = [anwer_list_0[ele] for ele in ranking_top_n_indices]
        print("final answer is: ")
        for ans in final_answer:
            print(ans)
        answer[n] = final_answer
    write_file(answer)



