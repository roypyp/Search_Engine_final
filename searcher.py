import math

from nltk.corpus import wordnet,lin_thesaurus as thes
import numpy as np
from ranker import Ranker
import utils
import gensim
import pandas as pd



# DO NOT MODIFY CLASS NAME
class Searcher:
    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit. The model 
    # parameter allows you to pass in a precomputed model that is already in 
    # memory for the searcher to use such as LSI, LDA, Word2vec models. 
    # MAKE SURE YOU DON'T LOAD A MODEL INTO MEMORY HERE AS THIS IS RUN AT QUERY TIME.
    def __init__(self, parser, indexer, model=None, dict=None, inx=None):
        self._parser = parser
        self._indexer = indexer
        self._ranker = Ranker()
        self._model = model
        self._dict=dict
        self.index=inx
        self.Docment_info=utils.load_obj("Docment_info")

    def dot(self,A, B):
        return (sum(a * b for a, b in zip(A, B)))

    def cosine_similarity(self,a, b):
        return self.dot(a, b) / ((self.dot(a, a) ** .5) * (self.dot(b, b) ** .5))
    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def search(self, query, k=None,engine=0):
        """ 
        Executes a query over an existing index and returns the number of 
        relevant docs and an ordered list of search results (tweet ids).
        Input:
            query - string.
            k - number of top results to return, default to everything.
            engine - a var that decide on what search engine we use
            0 - lsi
            1 - wordnet
            2 - spelling correction
            3 - thesaurus
            4 - advanced parser
            best - the best engine
        Output:
            A tuple containing the number of relevant search results, and 
            a list of tweet_ids where the first element is the most relavant 
            and the last is the least relevant result.
        """
        '''query_as_list = self._parser.parse_sentence(query)

        relevant_docs = self._relevant_docs_from_posting(query_as_list)
        n_relevant = len(relevant_docs)
        ranked_doc_ids = Ranker.rank_relevant_docs(relevant_docs)
        return n_relevant, ranked_doc_ids'''
        query_as_list = self._parser.parse_sentence(query)
        if(engine==0):
            """
            lsi engine
            """
            """queryexsp = []
            for word in query_as_list:
                for syn in wordnet.synsets(word):
                    w = wordnet.synsets(word)[0]
                    for l in syn.lemmas():

                        if l.name() == word or l.name().lower() == word or word == l.name():
                            # queryexsp+=[lname]
                            continue
                        else:
                            w2 = wordnet.synsets(l.name())[0]
                            score = w.wup_similarity(w2)
                            if (score != None and score > 0.8):
                                if (len(l.name().replace("_", " ").replace("-", " ").split()) == 1):
                                    queryexsp.append(l.name())
                # queryexsp = [t for t in queryexsp if t[1] != None]
            queryexsp = set(queryexsp)
            query_as_list += list(queryexsp)"""
            """
            dict for position of the doc to the tweet ID
            """
            tweet2doc = utils.load_obj("tweet2doc")
            """
            create the vec of the query from the lsi model
            """
            vec_bow = self._dict.doc2bow(query_as_list)
            vec_lsi = self._model[vec_bow]

            """
            find the similarity between the docs to the query with the cosin matrix of the lsi model
            """
            sims = self.index[vec_lsi]
            # print(list(enumerate(sims)))
            """
            combine the tweetid with the cosin similarity score
            """
            sims = sorted(zip(list(tweet2doc.values()), sims), key=lambda item: -item[1])
            sims=np.array(sims)
            a=sims[0:,1]
            a = a.astype(np.float)
            a=a[a > 0]
            sims=sims[0:len(a)]
            ranked_doc_ids = Ranker.rank_relevant_docs(sims)
            n_relevant = len(sims)
            return n_relevant,ranked_doc_ids
        elif(engine==1):
            """
            wordnet engine
            """
            # query expansion
            queryexsp = []
            for word in query_as_list:

                queryexsp += [(word, 1)]
                for syn in wordnet.synsets(word):
                    w = wordnet.synsets(word)[0]
                    for l in syn.lemmas():
                        lname = l.name()
                        if l.name() == word or l.name().lower() == word or word == lname:
                            queryexsp += [(lname, 1)]
                        else:
                            w2 = wordnet.synsets(l.name())[0]
                            score = w.wup_similarity(w2)
                            if (score != None and score > 0.60):#the score we dicide that bring better results
                                if (len(l.name().replace("_", " ").replace("-", " ").split()) == 1):
                                    queryexsp.append((lname.lower(), score * 0.8))#make the expansion word rank less

                # queryexsp = [t for t in queryexsp if t[1] != None]

            queryexsp = set(queryexsp)
            queryexsp = sorted(queryexsp, reverse=True, key=lambda item: item[0])
            #print(queryexsp)
            # sort_query = sorted(query)
            relevant_docs = {}
            tfidfdic = {}
            i, t = 0, 0
            # listForName = list(tempdict.keys())
            """
            calculate tf-idf with the score of the word with the improve tf-idf
            """
            while t < len(queryexsp):
                inv_ind=self._indexer.inverted_idx

                if(not inv_ind.get(queryexsp[t][0])):
                    t+=1
                    continue
                tempdic=inv_ind[queryexsp[t][0]][2]
                for tweet in tempdic.items():
                    if (tfidfdic.get(tweet[0])):
                        tfidfdic[tweet[0]][t] = (math.log2(tweet[1]+1) * math.log2((len(self.Docment_info) +1)/ inv_ind[queryexsp[t][0]][0])) * queryexsp[t][1]
                    else:
                        tfidfdic[tweet[0]] = [0] * len(queryexsp)
                        tfidfdic[tweet[0]][t] = (math.log2(tweet[1]+1) * math.log2((len(self.Docment_info) +1)/ inv_ind[queryexsp[t][0]][0])) * queryexsp[t][1]
                t += 1

            """
            co-sim and sort the result of the best ranked tweet 
            """
            dq = [1] * len(queryexsp)
            #dq = np.array(dq)
            #dq = np.transpose(dq)
            A = np.array(list(tfidfdic.values()))
            tweets = list(tfidfdic.keys())
            cos_sim = np.dot(A, dq) / (np.linalg.norm(A) * np.linalg.norm(dq))
            #cos_sim= self.cosine_similarity(A.T,dq)
            # cos_sim=1 - spatial.distance.cosine(A, dq)
            tweetcos = zip(tweets, cos_sim)
            tweetcos = set(tweetcos)
            relevant_docs = sorted(tweetcos, reverse=True, key=lambda item: item[1])
            ranked_doc_ids = Ranker.rank_relevant_docs(relevant_docs,500)
            n_relevant = len(ranked_doc_ids)
            return n_relevant, ranked_doc_ids
        elif engine==2:
            """
            spell correction engine
            """
            #add the correct spell of a wrong spell word to the query
            query_as_list=self._parser.spellcorect(query_as_list)
            relevant_docs = {}
            tfidfdic = {}
            i, t = 0, 0
            #tf-idf improve model
            while t < len(query_as_list):
                inv_ind=self._indexer.inverted_idx
                if (not inv_ind.get(query_as_list[t])):
                    t += 1
                    continue
                tempdic = inv_ind[query_as_list[t]][2]
                for tweet in tempdic.items():
                    if (tfidfdic.get(tweet[0])):
                        tfidfdic[tweet[0]][t] = (math.log2(tweet[1] + 1) * math.log2(
                            (len(self.Docment_info) + 1) / inv_ind[query_as_list[t]][0]))
                    else:
                        tfidfdic[tweet[0]] = [0] * len(query_as_list)
                        tfidfdic[tweet[0]][t] = (math.log2(tweet[1] + 1) * math.log2(
                            (len(self.Docment_info) + 1) / inv_ind[query_as_list[t]][0]))
                t += 1
            """
            co-sim and sort the result of the best ranked tweet 
            """
            dq = [1] * len(query_as_list)
            # dq = np.array(dq)
            # dq = np.transpose(dq)
            A = np.array(list(tfidfdic.values()))
            tweets = list(tfidfdic.keys())
            cos_sim = np.dot(A, dq) / (np.linalg.norm(A) * np.linalg.norm(dq))
            # cos_sim= self.cosine_similarity(A.T,dq)
            # cos_sim=1 - spatial.distance.cosine(A, dq)
            tweetcos = zip(tweets, cos_sim)
            tweetcos = set(tweetcos)
            relevant_docs = sorted(tweetcos, reverse=True, key=lambda item: item[1])
            ranked_doc_ids = Ranker.rank_relevant_docs(relevant_docs, 500)
            n_relevant = len(ranked_doc_ids)
            return n_relevant, ranked_doc_ids

        elif (engine == 3):
            """
            thesaurus engine
            """
            # query expansion
            queryexsp = []
            for word in query_as_list:

                queryexsp += [(word, 1)]
                for sim in thes.scored_synonyms(word):
                    for w in  dict(sim[1]).items():
                        if w[1] >0.175:#the score we dicide that bring better results
                            queryexsp+=[(w[0].lower(),0.8)]#make the expansion word rank less
                        else:
                            break



                # queryexsp = [t for t in queryexsp if t[1] != None]

            queryexsp = set(queryexsp)
            queryexsp = sorted(queryexsp, reverse=True, key=lambda item: item[0])
            #print(queryexsp)
            # sort_query = sorted(query)
            """
            calculate tf-idf with the score of the word with the improve tf-idf
            """
            relevant_docs = {}
            tfidfdic = {}
            i, t = 0, 0
            while t < len(queryexsp):
                inv_ind=self._indexer.inverted_idx
                if (not inv_ind.get(queryexsp[t][0])):
                    t += 1
                    continue
                tempdic = inv_ind[queryexsp[t][0]][2]
                for tweet in tempdic.items():
                    if (tfidfdic.get(tweet[0])):
                        tfidfdic[tweet[0]][t] = (math.log2(tweet[1] + 1) * math.log2(
                            (len(self.Docment_info) + 1) / inv_ind[queryexsp[t][0]][0])) * queryexsp[t][1]
                    else:
                        tfidfdic[tweet[0]] = [0] * len(queryexsp)
                        tfidfdic[tweet[0]][t] = (math.log2(tweet[1] + 1) * math.log2(
                            (len(self.Docment_info) + 1) / inv_ind[queryexsp[t][0]][0])) * queryexsp[t][1]
                t += 1

            """
            co-sim and sort the result of the best ranked tweet 
            """
            dq = [1] * len(queryexsp)
            # dq = np.array(dq)
            # dq = np.transpose(dq)
            A = np.array(list(tfidfdic.values()))
            tweets = list(tfidfdic.keys())
            cos_sim = np.dot(A, dq) / (np.linalg.norm(A) * np.linalg.norm(dq))
            # cos_sim= self.cosine_similarity(A.T,dq)
            # cos_sim=1 - spatial.distance.cosine(A, dq)
            tweetcos = zip(tweets, cos_sim)
            tweetcos = set(tweetcos)
            relevant_docs = sorted(tweetcos, reverse=True, key=lambda item: item[1])
            ranked_doc_ids = Ranker.rank_relevant_docs(relevant_docs, 500)
            n_relevant = len(ranked_doc_ids)
            return n_relevant, ranked_doc_ids


        elif engine==4:
            """
            advanced parser
            """

            """
            calculate tf-idf with improve tf-idf
            """
            relevant_docs = {}
            tfidfdic = {}
            i, t = 0, 0
            while t < len(query_as_list):
                inv_ind=self._indexer.inverted_idx
                if (not inv_ind.get(query_as_list[t])):
                    t += 1
                    continue
                tempdic = inv_ind[query_as_list[t]][2]
                for tweet in tempdic.items():
                    if (tfidfdic.get(tweet[0])):
                        tfidfdic[tweet[0]][t] = (math.log2(tweet[1] + 1) * math.log2(
                            (len(self.Docment_info) + 1) / inv_ind[query_as_list[t]][0]))
                    else:
                        tfidfdic[tweet[0]] = [0] * len(query_as_list)
                        tfidfdic[tweet[0]][t] = (math.log2(tweet[1] + 1) * math.log2(
                            (len(self.Docment_info) + 1) / inv_ind[query_as_list[t]][0]))
                t += 1
            """
            co-sim and sort the result of the best ranked tweet 
            """
            dq = [1] * len(query_as_list)
            # dq = np.array(dq)
            # dq = np.transpose(dq)
            A = np.array(list(tfidfdic.values()))
            tweets = list(tfidfdic.keys())
            cos_sim = np.dot(A, dq) / (np.linalg.norm(A) * np.linalg.norm(dq))
            # cos_sim= self.cosine_similarity(A.T,dq)
            # cos_sim=1 - spatial.distance.cosine(A, dq)
            tweetcos = zip(tweets, cos_sim)
            tweetcos = set(tweetcos)
            relevant_docs = sorted(tweetcos, reverse=True, key=lambda item: item[1])
            ranked_doc_ids = Ranker.rank_relevant_docs(relevant_docs, 500)
            n_relevant = len(ranked_doc_ids)
            return n_relevant, ranked_doc_ids

        elif (engine == "best"):
            """
            best engine wordnet + spelling correction + advanced parser
            """
            # add the correct spell of a wrong spell word to the query
            query_as_list = self._parser.spellcorect(query_as_list)
            # query expansion
            queryexsp = []
            for word in query_as_list:

                queryexsp += [(word, 1)]
                for syn in wordnet.synsets(word):
                    w = wordnet.synsets(word)[0]
                    for l in syn.lemmas():
                        lname = l.name()
                        if l.name() == word or l.name().lower() == word or word == lname:
                            queryexsp += [(lname, 1)]
                        else:
                            w2 = wordnet.synsets(l.name())[0]
                            score = w.wup_similarity(w2)
                            if (score != None and score > 0.60):#the score we dicide that bring better results
                                if (len(l.name().replace("_", " ").replace("-", " ").split()) == 1):
                                    queryexsp.append((lname.lower(), score * 0.8))#make the expansion word rank less

                # queryexsp = [t for t in queryexsp if t[1] != None]

            queryexsp = set(queryexsp)
            queryexsp = sorted(queryexsp, reverse=True, key=lambda item: item[0])
            #print(queryexsp)
            # sort_query = sorted(query)
            relevant_docs = {}
            tfidfdic = {}
            i, t = 0, 0
            # listForName = list(tempdict.keys())
            """
            calculate tf-idf with the score of the word with the improve tf-idf
            """
            while t < len(queryexsp):
                inv_ind=self._indexer.inverted_idx

                if (not inv_ind.get(queryexsp[t][0])):
                    t += 1
                    continue
                tempdic = inv_ind[queryexsp[t][0]][2]
                for tweet in tempdic.items():
                    if (tfidfdic.get(tweet[0])):
                        tfidfdic[tweet[0]][t] = (math.log2(tweet[1] + 1) * math.log2(
                            (len(self.Docment_info) + 1) / inv_ind[queryexsp[t][0]][0])) * queryexsp[t][1]
                    else:
                        tfidfdic[tweet[0]] = [0] * len(queryexsp)
                        tfidfdic[tweet[0]][t] = (math.log2(tweet[1] + 1) * math.log2(
                            (len(self.Docment_info) + 1) / inv_ind[queryexsp[t][0]][0])) * queryexsp[t][1]
                t += 1
            """
            co-sim and sort the result of the best ranked tweet 
            """
            dq = [1] * len(queryexsp)
            # dq = np.array(dq)
            # dq = np.transpose(dq)
            A = np.array(list(tfidfdic.values()))
            tweets = list(tfidfdic.keys())
            cos_sim = np.dot(A, dq) / (np.linalg.norm(A) * np.linalg.norm(dq))
            # cos_sim= self.cosine_similarity(A.T,dq)
            # cos_sim=1 - spatial.distance.cosine(A, dq)
            tweetcos = zip(tweets, cos_sim)
            tweetcos = set(tweetcos)
            relevant_docs = sorted(tweetcos, reverse=True, key=lambda item: item[1])
            ranked_doc_ids = Ranker.rank_relevant_docs(relevant_docs, 500)
            n_relevant = len(ranked_doc_ids)
            return n_relevant, ranked_doc_ids





    # feel free to change the signature and/or implementation of this function 
    # or drop altogether.
    def _relevant_docs_from_posting(self, query_as_list):
        """
        This function loads the posting list and count the amount of relevant documents per term.
        :param query_as_list: parsed query tokens
        :return: dictionary of relevant documents mapping doc_id to document frequency.
        """
        relevant_docs = {}
        for term in query_as_list:
            posting_list = self._indexer.get_term_posting_list(term)
            for doc_id, tf in posting_list:
                df = relevant_docs.get(doc_id, 0)
                relevant_docs[doc_id] = df + 1
        return relevant_docs
