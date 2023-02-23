from sklearn.feature_extraction.text import CountVectorizer
import streamlit as st
import pandas as pd
import numpy as np
from st_aggrid import AgGrid, GridUpdateMode
from st_aggrid.grid_options_builder import GridOptionsBuilder
import io 

def atualizar():
        gd = GridOptionsBuilder.from_dataframe(df)
        gd.configure_selection(selection_mode='multiple', use_checkbox=True)
        gridoptions = gd.build()

        grid_table = AgGrid(df, width=800, height=250, gridOptions=gridoptions,
                    update_mode=GridUpdateMode.SELECTION_CHANGED)


st.set_page_config(layout="wide")
st.title("Comprevisão")
st.text("Sistema de orçamento")

catalogo = ''
lista = ''

data1 = st.file_uploader("Arquivo da Empresa",type=["csv"])
if data1 is not None:
  catalogo = pd.read_csv(data1,sep=';')

data2 = st.file_uploader("Arquivo do Cliente",type=["csv"])
if data2 is not None:
  lista = pd.read_csv(data2,sep=';')

df = pd.DataFrame(columns=['Produto                                   ','Unitario ','Descrição                                     ','Qtde           ','Valor         '])



def medidor_de_similaridade(text1, text2):
  stop_words = ['de','da']
  to_vect = CountVectorizer(analyzer = 'char', ngram_range = (1, 2),  stop_words=stop_words, strip_accents = 'ascii')
  result = []
  for comentario1 in text1:
    for comentario2 in text2:
      x1, x2 = to_vect.fit_transform([comentario1,comentario2])
      t1, t2 = x1.toarray(), x2.toarray()

      min = np.amin([t1, t2], axis = 0)
      sum = np.sum(min)
      count = np.sum([t1, t2][0])
      if text1[0].replace(' ','')==text2[0].replace(' ',''):
        to_mean=1
      else:
        to_mean = (sum/count)-0.01
      result.append(to_mean)
  return to_mean

def insert(df, row):
    insert_loc = df.index.max()

    if np.isnan(insert_loc):
        df.loc[0] = row
    else:
        df.loc[insert_loc + 1] = row

def to_excel(df) -> bytes:
         output = io.BytesIO()
         writer = pd.ExcelWriter(output, engine="xlsxwriter")
         df.to_excel(writer, sheet_name="Sheet1")
         writer.save()
         processed_data = output.getvalue()
         return processed_data



try:
    total=0
    nao_encontrados = []

    for t1 in range(len(lista)):
        x1=[]
        x1.append(lista['Produto'][t1])
        x11 = lista['Qtde'][t1]

        val_ant = -1
        i = 0

        for i in range(len(catalogo)):
            w1=[]
            w1.append(catalogo['Produto'][i])
            w11 = catalogo['Preco'][i]
            val = medidor_de_similaridade(x1, w1)

            if val > val_ant:
                prod = catalogo['Produto'][i]
                preco = catalogo['Preco'][i]
                val_ant = val


        if val_ant >= 0.61:
            if (prod.find('com')>0 or prod.find('sem')>0) and val_ant >= 0.90:
                insert(df,['{:<100}'.format(str("['"+prod+"']")),'{:>10}'.format(str(preco)),'{:<100}'.format(str(x1)),'{:>15}'.format(str(x11)),'{:>15}'.format(str(float(x11)*float(preco)))])
                total = total + (float(preco)*float(x11))
            elif prod.find('com')==-1 and prod.find('sem')==-1:
                insert(df,['{:<100}'.format(str("['"+prod+"']")),'{:>10}'.format(str(preco)),'{:<100}'.format(str(x1)),'{:>15}'.format(str(x11)),'{:>15}'.format(str(float(x11)*float(preco)))])
                total = total + (float(preco)*float(x11))
            else:
                nao_encontrados.append(lista['Produto'][t1])
        else:
            nao_encontrados.append(lista['Produto'][t1])

    

    if st.button('Excluir os Selecionados !'):
        gd = GridOptionsBuilder.from_dataframe(df)
        gd.configure_selection(selection_mode='multiple', use_checkbox=True)
        gridoptions = gd.build()

        grid_table = AgGrid(df, width=800, height=250, gridOptions=gridoptions,
                    update_mode=GridUpdateMode.SELECTION_CHANGED)

        selected_row = grid_table["selected_rows"]
        df3 = pd.DataFrame (selected_row, columns = ['Descrição                                     '])

        df.drop(df[df['Descrição                                     '].isin(df3['Descrição                                     '])].index, inplace=True)
        df = df.reset_index(drop=True)

        total = 0
        i = 0
        for i in range(len(df['Valor         '])):
            total = total + float(df['Valor         '][i])

    atualizar()

    st.download_button("Download as excel",
        data=to_excel(df),
        file_name="output.xlsx",
        mime="application/vnd.ms-excel",)

 #   st.dataframe(df.fillna(0))

    st.write('Valor total da compra = R$  ' + str(total))
  
    df2 = pd.DataFrame(columns=['Produto                                              '])

    st.text("Produtos não encontrados")

    i=0
    for i in range(len(nao_encontrados)):
        insert(df2,nao_encontrados[i])

    st.dataframe(df2)
  

    #st.dataframe(df3)



except ValueError as Argument:
    st.write ( "ERRO" ,  Argument)