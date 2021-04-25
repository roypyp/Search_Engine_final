# you can change whatever you want in this module, just make sure it doesn't 
# break the searcher module
import numpy as np
class Ranker:
    def __init__(self):
        pass

    @staticmethod
    def rank_relevant_docs(relevant_docs, k=None):
        """
        This function provides rank for each relevant document and sorts them by their scores.
        The current score considers solely the number of terms shared by the tweet (full_text) and query.
        :param k: number of most relevant docs to return, default to everything.
        :param relevant_docs: dictionary of documents that contains at least one term from the query.
        :return: sorted list of documents by score
        """

        ranked_results = sorted(relevant_docs, key=lambda item: item[1], reverse=True)
        ranked_results=np.array(ranked_results)
        ranked_results=list(ranked_results[:,0])
        if k==None:
            return  ranked_results
        return ranked_results[:k]


