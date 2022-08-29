import streamlit as st
import numpy as np
import pandas as pd
import re
import math
from collections import Counter
from unicodedata import normalize
from nltk import ngrams
from pandas import DataFrame
import csv

st.title("Comprevisão")
st.text("Sistema de orçamento")

catalogo = ''

data1 = st.file_uploader("Arquivo da Empresa",type=["csv"])
if data1 is not None:
  words = pd.read_csv(data1,sep=';')

data2 = st.file_uploader("Arquivo do Cliente",type=["csv"])
if data2 is not None:
  catalogo = pd.read_csv(data2,sep=';')

#words = pd.read_csv('/content/Base.csv',sep=';')
#catalogo = pd.read_csv('/content/Lista_de_Compras.csv',sep=';')

#Regex para encontrar tokens
REGEX_WORD = re.compile(r'\w+')
#Numero de tokens em sequencia
N_GRAM_TOKEN = 3

#Faz a normalizacao do texto removendo espacos a mais e tirando acentos
def text_normalizer(src):
    return re.sub('\s+', ' ',
                normalize('NFKD', src)
                   .encode('ASCII','ignore')
                   .decode('ASCII')
           ).lower().strip()
    
#Faz o calculo de similaridade baseada no coseno
def cosine_similarity(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        coef = float(numerator) / denominator
        if coef > 1:
            coef = 1
        return coef
    
#Monta o vetor de frequencia da sentenca
def sentence_to_vector(text, use_text_bigram):
    words = REGEX_WORD.findall(text)
    accumulator = []
    for n in range(1,N_GRAM_TOKEN):
        gramas = ngrams(words, n)
        for grama in gramas:
            accumulator.append(str(grama))

    if use_text_bigram:
        pairs = get_text_bigrams(text)
        for pair in pairs:
            accumulator.append(pair)

    return Counter(accumulator)

#Obtem a similaridade entre duas sentencas
def get_sentence_similarity(sentence1, sentence2, use_text_bigram=False):
    vector1 = sentence_to_vector(text_normalizer(sentence1), use_text_bigram)
    vector2 = sentence_to_vector(text_normalizer(sentence2), use_text_bigram)
    return cosine_similarity(vector1, vector2)


#Metodo de gerar bigramas de uma string
def get_text_bigrams(src):
    s = src.lower()
    return [s[i:i+2] for i in range(len(s) - 1)]


def get_dataframe_similarity(comparer, finder, cutoff):
    print('cutoff= ' + str(cutoff))
    result = []
    comparer = np.array(comparer)
    for find in np.array(finder):
        max_coef = 0
        data = find
        for compare in comparer:
            similarity = get_sentence_similarity(find[0], compare[0])
            if similarity >= cutoff:
                if similarity > max_coef:
                    print('Trocando ' + data[1] + ' por ' + compare[1])
                    print(data[0] + ' ---- ' + compare[0] + ' - similaridade: ' + str(float( '%g' % ( similarity * 100 ) )) + '%')
                    data[1] = compare[1]
                    max_coef = similarity
        result.append(data)

    result = np.array(result)
    dataFrame = pd.DataFrame()
    dataFrame['texto'] = result[..., 0]
    dataFrame['marca'] = result[..., 1]
    return dataFrame


df = pd.DataFrame(columns=['Produto                       ','Unitario ','Descrição                              ','Qtde     ','Valor     '])

#pd.set_option("max_colwidth", 100)


def insert(df, row):
    insert_loc = df.index.max()

    if np.isnan(insert_loc):
        df.loc[0] = row
    else:
        df.loc[insert_loc + 1] = row

try:
  removidos = []
  cleanlist = []
  total = 0
  #if __name__ == "__main__":
  final = []
  i = 0
  for i in range(len(catalogo)):
    w1 = catalogo['Produto'][i]
    w11 = catalogo['Qtde'][i]
    result = []
    z = 0
    similarity_sentence_text_bigram_ant = 0
    if not w1 in removidos:
      v = 0
      n = 0
      for n in range(len(words)):
        w2  = words['Produto'][v]
        w22 = words['Preco'][v]
        v = v+1
        list_cutoff = [1,0.9,0.8,0.7,0.6,0.5]
        for cutoff in list_cutoff:
          similarity_sentence_text_bigram = get_sentence_similarity(w1, w2, use_text_bigram=True)
          if similarity_sentence_text_bigram >= similarity_sentence_text_bigram_ant and similarity_sentence_text_bigram  > 0 and similarity_sentence_text_bigram  >= cutoff:
            insert(df,['{:<100}'.format(str(w2)),'{:>10}'.format(str(w22)),'{:<100}'.format(str(w1)),'{:>20}'.format(str(w11)),'{:>20}'.format(str(float(w11)*float(w22)))])
#            result.append('{:<40}'.format(str(w2)) + ";" + '{:>9}'.format(str(w22)) + ";" + '{:<40}'.format(str(w1)) + ";" + '{:>13}'.format(str(w11)) + ";" + '{:>13}'.format(str(float(w11)*float(w22))))
            total = total + (float(w11)*float(w22))
            removidos.append(w1)
            z = 1 
            similarity_sentence_text_bigram_ant = similarity_sentence_text_bigram
            #result=str(result)
            #result= result.replace("['","")
            #result= result.replace("'","")
            #result= result.replace("]","")
                                
            #final.append(result)
            break

            if z == 1:
              break
                    
  # =============================================================================
  def remove_values_from_list(the_list, val):
    return [value for value in the_list if value != val]
             
    final = remove_values_from_list(final, [])
                     
  # =============================================================================
  # executar quando terminar            
  #cleanlist.append("Produto; PrUnit; Produtos; Qtde; Valor")                    
#  [cleanlist.append(x) for x in final if x not in cleanlist]

  st.dataframe(df)

  st.write('Valor total da compra = R$  ' + str(total))
  
  df = pd.DataFrame(columns=['Produto                                              '])

  st.text("Produtos não encontrados")

  i=1
  for i in range(len(catalogo)):
    w1 = catalogo['Produto'][i]
    if not w1 in removidos:
      insert(df,w1)

  st.dataframe(df)
  

except ValueError as Argument:
  st.write ( "ERRO" ,  Argument)
