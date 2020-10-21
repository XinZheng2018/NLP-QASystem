import nltk
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

# a = preprocessing(path)
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

filename = './training/qadata/top_docs_test.0'
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
    question_vectors = [1 if x in question else 0 for x in voc_list]
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
    copy_sim_list = copy.deepcopy(sim_list)
    res = sorted(copy_sim_list, key=lambda x: x, reverse=True)[:N]
    #same value ?"????
    top_n_list = [bow(sim_list.index(x))for x in res]
    return top_n_list




    # document separated to lines
            # elif line.startswith("<P>"):
            #     for line in f:
            #         if line.startswith("</P>"):
            #             break
            #         elif line.startswith("<"):
            #             continue
            #         else:
            #             for word in line.split():
            #                 if not word in stop_words:
            #                     doc.append(word)

            # document stored as an entire text
            # there are some other attributes that I did not cover
            # in rank 4/5:
            # <first> <second>, etc.
            # in rank 21:
            # <memo> <Caption> <LeadPara> <Section> needs to be cover
            # also, I think the case where header goes to the next line need to be covered (not sure)
            # code below is covered in the function


<<<<<<< HEAD
filename = 'training/topdocs/top_docs.0'
def document_sep(filename):
    rank = 0
    doc = []
    token_blocks = []

    with open(filename, 'r', encoding='utf-8-sig') as f:
        for line in f:
            if "Rank" in line:
                print("line contains rank: ")
                print(line)
                numbers = [int(word) for word in line.split() if word.isdigit()]
                if rank != numbers[1]:
                    doc = []
                    rank = numbers[1]

            elif line.startswith("<FIELDID>"):
                # function to preprocess other tags
                process_data()

            # document separated to lines
            elif line.startswith("<P>"):
                for line in f:
                    if line.startswith("</P>"):
                        break
                    elif line.startswith("<"):
                        continue
                    else:
                        for word in line.split():
                            if not word in stop_words:
                                doc.append(word)

            # document stored as an entire text
            # there are some other attributes that I did not cover
            # in rank 4/5:
            # <first> <second>, etc.
            # in rank 21:
            # <memo> <Caption> <LeadPara> <Section> needs to be cover
            # also, I think the case where header goes to the next line need to be covered (not sure)
            # code below is covered in the function


=======
>>>>>>> jing
            # elif line.startswith("<HL>") or line.startswith("<HEAD>"):
            #     for word in line.split():
            #         if not word in stop_words and (word != "</HL>" or word != "</HEAD>"):
            #             doc.append(word)
            # elif line.startswith("<AUTHOR>"):
            #     for word in line.split():
            #         if not word in stop_words and word != "</AUTHOR>":
            #             doc.append(word)
<<<<<<< HEAD
            elif line.startswith("<TEXT>"):
                next_line = next(f)
                # repeated code
                # need to make it cleaner if possible
                if next_line.startswith("<P>"):
                    for line in f:
                        if line.startswith("</TEXT>"):
                            break
                        elif line.startswith("</P>"):
                            break
                        elif line.startswith("<"):
                            continue
                        else:
                            for word in line.split():
                                if not word in stop_words:
                                    doc.append(word)

                else:
                    for line in f:
                        break_flag = False
                        if line.startswith("</TEXT>"):
                            break
                        elif break_flag:
                            break
                        else:
                            for word in line.split():
                                if word == "</TEXT>":
                                    break_flag = True
                                    break
                                else:
                                    if not word in stop_words:
                                        doc.append(word)
                print(rank)
                print(doc)
            else:
                continue

document_sep(filename)

=======
            # elif line.startswith("<TEXT>"):
            #     next_line = next(f)
            #     # repeated code
            #     # need to make it cleaner if possible
            #     if next_line.startswith("<P>"):
            #         for line in f:
            #             if line.startswith("</TEXT>"):
            #                 break
            #             elif line.startswith("</P>"):
            #                 break
            #             elif line.startswith("<"):
            #                 continue
            #             else:
            #                 for word in line.split():
            #                     if not word in stop_words:
            #                         doc.append(word)
            #
            #     else:
            #         for line in f:
            #             break_flag = False
            #             if line.startswith("</TEXT>"):
            #                 break
            #             elif break_flag:
            #                 break
            #             else:
            #                 for word in line.split():
            #                     if word == "</TEXT>":
            #                         break_flag = True
            #                         break
            #                     else:
            #                         if not word in stop_words:
            #                             doc.append(word)
            #     print(rank)
            #     print(doc)
            # else:
            #     continue
>>>>>>> jing


