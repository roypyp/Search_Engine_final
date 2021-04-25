import time

import pandas as pd
from gensim import similarities
from gensim.corpora import Dictionary
from gensim.models import LsiModel

from reader import ReadFile
from configuration import ConfigClass
from parser_module import Parse
from indexer import Indexer
from searcher import Searcher
import utils


# DO NOT CHANGE THE CLASS NAME
class SearchEngine:

    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation, but you must have a parser and an indexer.
    def __init__(self, config=None):
        self._config = config
        persondic = utils.load_person_dict(config._model_dir)
        self._parser = Parse(persondic)
        self._indexer = Indexer(config)
        self._model = None
        self._dict = None
        self.index = None

    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def build_index_from_parquet(self, fn):
        """
        Reads parquet file and passes it to the parser, then indexer.
        Input:
            fn - path to parquet file
        Output:
            No output, just modifies the internal _indexer object.
        """
        df = pd.read_parquet(fn, engine="pyarrow")
        documents_list = df.values.tolist()
        # Iterate over every document in the file
        self._parser.deletcorp()
        number_of_documents = 0
        for idx, document in enumerate(documents_list):
            # parse the document
            parsed_document = self._parser.parse_doc(document,lsi=False)
            number_of_documents += 1
            # index the document data
            self._indexer.add_new_doc(parsed_document)
        print('Finished parsing and indexing.')


        # save the numbers of documents in the corpus
        utils.save_obj(self._indexer.DocmentInfo, "Docment_info")
        #print(len(self._indexer.inverted_idx))
        #self._indexer.save_index()

    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def load_index(self, fn):
        """
        Loads a pre-computed index (or indices) so we can answer queries.
        Input:
            fn - file name of pickled index.
        """
        self._indexer.load_index(fn)

    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def load_precomputed_model(self, model_dir=None):
        """
        Loads a pre-computed model (or models) so we can answer queries.
        This is where you would load models like word2vec, LSI, LDA, etc. and
        assign to self._model, which is passed on to the searcher at query time.
        """
        pass


    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def search(self, query,k=None):
        """
        Executes a query over an existing index and returns the number of
        relevant docs and an ordered list of search results.
        Input:
            query - string.
        Output:
            A tuple containing the number of relevant search results, and
            a list of tweet_ids where the first element is the most relavant
            and the last is the least relevant result.
        """
        searcher = Searcher(self._parser, self._indexer)
        return searcher.search(query,k=k,engine=3)
def main(queries='queries.txt',num_docs_to_retrieve=5):
    config= ConfigClass()
    engine= SearchEngine(config)
    engine.build_index_from_parquet("data/benchmark_data_train.snappy.parquet")
    engine.load_precomputed_model()
    if isinstance(queries, list):
        query = queries
    else:
        query = filetolist(queries)
    k = num_docs_to_retrieve  # int(input("Please enter number of docs to retrieve: "))

    i = 0
    for q in query:
        start_time = time.time()
        for doc_tuple in engine.search(q)[1]:
            print('tweet id: {}'.format(doc_tuple))
        i += 1
        print(time.time() - start_time)


def filetolist(q):
    """
    get a file path and make the file to a list of queries
    :param q:
    :return:
    """
    with open(q, encoding="utf-8") as f:
        content = f.readlines()
    query = []
    for line in content:
        if (line != "\n" and line != "" and line != " "):
            query += [line.replace("\n", "")]
    return query