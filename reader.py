import os
import pandas as pd
import glob2



class ReadFile:
    def __init__(self, corpus_path):
        self.corpus_path = corpus_path
        self.allfiles = glob2.glob(self.corpus_path + "/*/*/*.parquet") + glob2.glob(self.corpus_path + "/*/*.parquet") + glob2.glob(self.corpus_path + "/*.parquet")
        self.int=0
        self.maxlen=len(self.allfiles)
    def readonefile(self,i):
        return [(pd.read_parquet(self.allfiles[i], engine="pyarrow")).values.tolist()]
    def read_file(self,name=False):
        """
        This function is reading a parquet file contains several tweets
        The file location is given as a string as an input to this function.
        :param file_name: string - indicates the path to the file we wish to read.
        :return: a dataframe contains tweets.
        """
        if(name!=False):
            return (pd.read_parquet(self.corpus_path+"/"+name, engine="pyarrow")).values.tolist()
        #full_path = os.path.join(self.corpus_path)

        df = self.readonefile(self.int)
        self.int+=1

        return df
