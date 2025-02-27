#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Importar las bibliotecas necesarias
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime
import numpy as np
from scipy.stats import norm
import pip
pip.main(["install","openpyxl"])

# In[ ]:


#lectura del archivo con los datos
data=pd.read_excel('Ejercicio Inversiones Finaccess.xlsx',sheet_name='Portafolios')
data_2=pd.read_excel('Ejercicio Programacion Finaccess.xlsx')

# In[ ]:

st.image('Logo.png')
st.sidebar.subheader('Ramos Palacios Juan Alexis')
ejercicio=st.sidebar.selectbox('Seleccione la opción a visualizar',['','Ejercicio Inversiones Finaccess','Ejercicio Programación Finaccess'])

if ejercicio == 'Ejercicio Inversiones Finaccess':
  st.sidebar.title("Ejercicio Inversiones Finaccess")
  option = st.sidebar.radio("Seleccione el ejercicio a visualizar", 
                            ['','Introducción','Portafolios','Rendimiento','Valuación','Gráficas','Comentario'])
  
  
  if option=='Introducción':
      st.title('Introducción')
      st.write('''En esta entrega se busca mostrar la solución a los ejercicios con una implememtación 
      y aplicación de Streamlit para una mejor vizualización de los resultados.''')
  
  elif option == 'Portafolios':
      st.title('Portafolios')
      st.markdown('''1. Construya una gráfica con el rendimiento acumulado de cada portafolio.
              ¿En que portafolio invertiría? Explique su respuesta.
              ''')
      st.markdown('# Respuesta')
      st.subheader('Rendimientos Acumulados de Portafolios vs Benchmark')
      st.write('''Este análisis muestra los rendimientos acumulados de varios portafolios en comparación con un 
                  Benchmark durante un periodo con inicio en Enero 2021 y finalización en Diciembre 2022.
                  Se puede interactuar con la infromación de los datos originales y con los rendimientos acumulados''')
  
  
      #se extrae el nombre de las columnas y se les da formato
      columns_names=data.iloc[7][1:]
      columns_names=pd.DataFrame(columns_names)
      columns_names.reset_index(drop=True,inplace=True)
      columns_names.columns=['Fechas']
      columns_names=columns_names.map(lambda x:x.strftime("%b-%Y"))
      
      #formato a la tabla de los datos
      data_=pd.DataFrame(data[8:])
      data_.set_index('Unnamed: 0',inplace=True)
      data_.index.names=['Portafolios']
      data_.columns=columns_names['Fechas'].tolist()
      #se convierte en una traspuesta para poder aplicar los rendimientos acumulados
      data_=pd.DataFrame(data_).T
  
      if st.checkbox('Mostrar datos originales'):
          st.dataframe(data_)
          
      #se crean los rendimientos acumulados de cada portafolio
      rend_acumu=(1 + data_).cumprod()-1
  
      st.markdown('''
                  Para poder generar los rendimientos acumulados nos apoyamos de la siguiente fórmula:
                  
                  $$Rendimientos Acumulados = (1+r_1)(1+r_2)\cdots(1+r_n) - 1$$
  
                  Donde:
                  
                  - r es el rendimiento obtenido en cada periodo
  
                  Adicionalmente se realizó el cálculo de la volatilidad, la cual es la desviación estándar:
  
                  $$\sigma = \sqrt{ \\frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^2}$$
                  
                  Donde:
                  - $\sigma$ es la desviación estándar
                  - N es el tamaño de la población.
                  - $x_i$ son los valores individuales.
                  - $\mu$ es la media poblacional.
  
                  ''')
  
      if st.checkbox('Mostrar rendimientos acumulados'):
          st.dataframe(rend_acumu)
      
      st.write('''
      Se muestra un gráfico de los rendimientos acumulados y además el Benchmark. Gracias a la 
      implementación de streamlit podemos interactuar con los gráficos y escoger aquellos que 
      nos favorezcan.
      ''')
      
      opciones = list(rend_acumu.columns)
      seleccion = st.multiselect('Selecciona los portafolios a comparar:', opciones, default=opciones)
      
      # Graficar los rendimientos acumulados
      fig, ax = plt.subplots(figsize=(12, 6))
      for column in seleccion:
          ax.plot(rend_acumu.index, rend_acumu[column], label=column)
      
      # Personalizar el gráfico
      ax.set_title('Rendimientos Acumulados de Portafolios vs Benchmark (2021)')
      ax.set_xlabel('Mes')
      ax.set_ylabel('Rendimiento Acumulado')
      ax.legend(title='Portafolio')
      ax.grid(True)
      plt.xticks(rotation=45)
      ax.yaxis.grid()
      ax.xaxis.grid()
      
      # Mostrar el gráfico en Streamlit
      st.pyplot(fig)
  
      order=pd.DataFrame(rend_acumu.agg(['sum','std']).T)
      order.columns=['Suma Total','Volatilidad']
      st.dataframe(order.sort_values(ascending=False,by='Suma Total'))
      
      st.markdown('''
      Adicionalmente se muestra una tabla en la cual se nos indica la 
      suma de los rendimientos totales por cada portafolio y el Benchmark.
  
      Una buena estrategia para elegir un portafolio es aquella que tuvo rendimientos mayores al Benchmark,
      podemos observar que los portafolios A, B y D están por encima. Ahora una buena práctica es ver
      el comportamiento gráfico del portafolio y observar que tan volatil es, para este caso y de acuerdo con
      la tabla el Portafolio A es el más volátil por los cual se descarta.
  
      Realizando la misma interpretación para los portafolios B y D se observa que ambos son buena opción,
      pero de acuerdo con la volatilidad nos quedamos con el **Portafolio B**.
      ''')
      
  
  elif option == 'Rendimiento':
      st.title('Rendimiento')
      st.markdown('''
          2. Juan tiene un portafolio de inversión con los siguientes AUM (Assets Under Management).
          \\
          El 31/12/2022  - \$10 millones.
          \\
          El 31/01/2023 - \$9.5 millones.
          \\
          El 28/02/2023 - \$10.5 millones.
          \\
          El 31/03/2023 - \$10 millones.
          \\
          El 30/04/2023 - \$15 millones.
          \\
          El 31/05/2023 - \$8 millones.
          \\
          El 30/06/2023 - \$12 millones.
          
                  
          Durante el año realizó las siguientes operaciones:
          \\
          15/03/2023 - retiró \$1.5 millones
          \\
          25/04/2023 - depositó \$7 millones
          \\
          5/05/2023 - retiró \$8 millones
          \\
          10/06/2023 - depositó \$5 millones
          \\
          25/06/2023 - retiró \$2 millones
          
                  
          Calcule el rendimiento del mes de junio y YTD al cierre de junio.
          # Respuesta    
          ''')
      st.subheader('Fórmula de Dietz Modificado')
      st.markdown('''
          La Fórmula de Dietz Modificado es un método para calcular el rendimiento de una 
          cartera de inversiones teniendo en cuenta los flujos de efectivo (aportaciones y retiros)
          que ocurren durante el período de evaluación, pondera los flujos de efectivo según el tiempo que permanecen 
          invertidos en la cartera.
  
          Fórmula:
                  
          $$R = \\frac{V_f - V_i - \sum C_i}{V_i + \sum C_i \left( \\frac{w_i}{T} \\right)}$$
          
          Donde:
          - R es el rendimiento de la cartera.
          - $V_f$ es el valor final de la cartera.
          - $V_i$ es el valor inicial de la cartera.
          - $C_i$ son los flujos de efectivo (aportaciones o retiros) ocurridos durante el período.
          - $w_i$ son los días restantes en el período después de que ocurrió el flujo de efectivo.
          - $T$ es el total de días en el período de evaluación.
  
                  ''')
  
  
      V_f=12000000
      V_i=8000000
      flujos_efec=[5000000,-2000000]
      fechas_flujos=['10/06/2023','25/06/2023']
      #se obtiene el vector de los dias aplicado el ponderado
      pct_dias_flujos=[(datetime.strptime('30/06/2023', '%d/%m/%Y')-datetime.strptime(fecha, '%d/%m/%Y')).days/30 for fecha in fechas_flujos]
      
      #Dietz modificado
      D_m=np.round((V_f-V_i-sum(flujos_efec))/(V_i+sum(np.multiply(flujos_efec,pct_dias_flujos).tolist())),4)*100
  
      st.markdown('''
                  Como se pide el rendimiento del mes de junio, solo se consideran las aportaciones
                  y retiros (flujos de efectivo) correspondientes al mes de junio, en este caso se tiene que:
                  10/06/2023 - depositó \$5 millones y 25/06/2023 - retiró \$2 millones.
  
                  Además, para este primer inciso debemos considerar los AUM del mes correspondiente 
                  por lo que tenemos que:
                  - $V_i=8,000,000$
                  - $V_f=12,000,000$
                  ''')
      #rendimeinto del mes de junio
      st.write(f'Por lo tanto, el rendimiento del mes de Junio: {D_m}%')
      
      YTD_V_f=12000000
      YTD_V_i=10000000
      YTD_flujos_efec=[-1500000,7000000,-8000000,5000000,-2000000]
      total_dias_period=(datetime.strptime('30/06/2023', '%d/%m/%Y')-datetime.strptime('31/12/2022', '%d/%m/%Y')).days
      YTD_fechas_flujos=['15/03/2023','25/04/2023','5/05/2023','10/06/2023','25/06/2023']
      YTD_pct_dias_flujos=[((datetime.strptime('30/06/2023', '%d/%m/%Y')-datetime.strptime(fecha, '%d/%m/%Y')).days)/total_dias_period for fecha in YTD_fechas_flujos]
      
      #Dietz modificado
      YTD_D_m=np.round((YTD_V_f-YTD_V_i-sum(YTD_flujos_efec))/(YTD_V_i+sum(np.multiply(YTD_flujos_efec,YTD_pct_dias_flujos).tolist())),4)*100
      #rendimeinto del mes de junio
  
      st.markdown('''
                  Ahora se solicita el YTD, por lo que se consideran todas las aportaciones
                  y retiros (flujos de efectivo) correspondientes al periodo completo.
  
                  Además, para este segundo inciso debemos considerar los AUM del YTD correspondientes son:
                  
                  - $V_i=10,000,000$
                  - $V_f=12,000,000$
                  ''')
      st.write(f'Por lo tanto, el YTD del mes de junio: {round(YTD_D_m,4)}%')
  
  elif option=='Valuación':
      st.title('Valuación')
      st.markdown('''
      3. Calcule el valor de la compañía.
      \\
      Notas:
      
      *los primeros tres años de flujos crecerán 10%
      
      *En los años 4 y 5 el crecimiento será de 5%
      
      *Después del quinto año los flujos crecerán 3% indefinidamente.
      
      - WACC=15%
      - Rendimiento IPC = 13%
      - Cetes28=11.5%
      - cetes360=11%
      - Bono10Y=8.6%
      - Beta=1.2
      - inflación=5.5%
      
      Free cashflow (Hoy)=\$915,648
      ''')
  
      st.markdown('''
                  # Respuesta 
  
                  Para la soución de este ejercicio, es importante considerar los conceptos de
                  valor futuro y valor presente, ya que nos ayudarán a poder interpretar los datos
                  y además poder obtener la respuesta.
  
                  - Valor Presente: El **Valor Presente (VP)** es el valor actual de una suma de 
                  dinero que se recibirá en el futuro, descontada a una tasa de interés 
                  determinada. Se calcula con la fórmula:
  
                  $$VP = \\frac{VF}{(1+i)^n}$$
  
                  Donde:
                  - $VF$ = Valor Futuro  
                  - $i$ = Tasa de interés  
                  - $n$ = Número de periodos  
                  
                  Valor Futuro: El **Valor Futuro (VF)** es el valor que tendrá una suma de dinero 
                  invertida hoy después de acumular intereses durante un periodo determinado. Se 
                  calcula con la fórmula:
                  
                  $$VF = VP \cdot (1+i)^n$$
                  
                  Donde:
                  - $VP$ = Valor Presente  
                  - $i$ = Tasa de interés  
                  - $n$ = Número de períodos
  
                  **Nota:** Una variante de la fórmula al considerar diferentes tasas es:
                  $$VF = VP \cdot (1+i)^{n-t} \cdot(1+r)^{t}$$
              
                  Teniendo en cuenta lo anterior ya podemos obtener el valor futuro de los primeros 
                  5 años del Free Cashh Flow (FCF), teniendo que su calculo sería:
      
                  $$VF\_FCF_{5}=FCF\cdot (1+10\%)^{3} \cdot (1+5\%)^{2}$$
  
                  Lo anterior considera el valor futuro del FCF en donde los primeros 3 años
                  se usa una tasa del 10% y los próximos 2 años una tasa del 5%.
                  ''')
  
      WACC=0.15
      Rend_IPC=0.13
      Cetes28=0.115
      Cetes360=0.11
      FCF=915648
      
      #se considera el valor futuro de los primeros 5 años
      VF_5=round(FCF*(1+.10)**(3)*(1+0.05)**(2),2)
      #VF_5
  
      Bono10Y=0.086
      Beta=1.2
      CAPM=Bono10Y+Beta*(Rend_IPC-Bono10Y)
      st.markdown(''' 
              Para poder hacer el cálculo de la perpetuidad tomaremos dos opciones, usando la tasa
              de WACC y adicionalmente la de CAPM.
  
              Para poder hacer el cálculo de la perpetuidad es importante saber más sobre su fórmula.
              Además se debe de considerar que a partir del año 6 se aplica, por lo que nuestra fórmula
              de perpetuidad queda modificada y es de la siguiente forma:
  
              **Perpetuidad**
  
              $$P=\\frac{VF\_FCF_{5}\cdot(1+0.03)}{WACC-0.03}$$
  
              Donde:
              - P es el valor futuro de la perpetuidad
              - $VF\_FCF_{5}$ es el flujo de los primeros 5 años
              - WACC es la tasa de descuento o también sería el CAPM
              - 0.03 es la tasa que se usa para crecer indefinidamente
  
              **Nota:** Para el cálculo del CAPM se ve de la siguiente forma:
  
              $$CAPM=R_f\\beta(R_m-R_f)$$
              Donde:
              - $R_f$ es la rentabilidad de un activo sin riesgo, en este caso usaremos el Bono10Y
              - $R_m$ es la rentabilidad del mercado, usaremos el rendimiento del IPC
              - $\\beta$ es el coeficiente de variabilidad del rendimiento de los recursos propios de la empresa
              
              ''')
  
  
      def select_tasa(opcion):
          if opcion == 'WACC':
              st.markdown('''
                          Considerando lo anterior respecto a la perpetuidad y teniendo en cuenta que
                          el WACC es una tasa de escuento, por lo que:
                          ''')
              VF_perpetuidad=VF_5*(1+0.03)/(WACC-0.03)
              
              st.write(f'El valor futuro de la perpetuidad es de: ${VF_perpetuidad:,.2f}')
              #se obtiene el valor presente de la perpetuidad con el wacc
              VP_perpetuidad=VF_perpetuidad*(1+WACC)**(-5)
              st.markdown('''
                          Considerando el valor futuro de la perpetuidad, obtendremos ahor su valor
                          presente pero usando al WACC como nuestra tasa, por lo que entonces:
                          ''')
              
              st.write(f'El valor presente de la perpetuidad considerando la tasa de WACC es de: ${VP_perpetuidad:,.2f}')
              vc_VF_5=[]
              for i in range(1,6):
                  if i>=4:
                      valor=valor*(1+0.05)
                      vc_VF_5.append(round(valor,2))
                  else:
                      valor=FCF*(1+.10)**(i)
                      vc_VF_5.append(round(valor,2))
  
              st.markdown('''
                          Una vez obtenido el VP de la perpetuidad, haría falta los VP de los flujos
                          de los primeros 5 años, pero de igual forma considerando al WACC como la tasa.
                          ''')
              vec_VP_5=[vc_VF_5[i-1]*(1+WACC)**(-i) for i in range(1,len(vc_VF_5)+1)]
              #se suman todos los valores presentes
  
              st.markdown('''
                          Una vez teniendo los VP, la suma total de éstos VP nos dará el valor final,
                          es decir:
                          
                          $$Valor_{final}= VP_1+VP_2+VP_3+VP_4+VP_5+VP_{perpetuidad}$$
                          
                          por lo tanto:
                          ''')
              total_final=sum(vec_VP_5)+VP_perpetuidad
              st.write(f'El valor final de la empresa considerando WACC: ${total_final:,.2f}')
              
          elif opcion == 'CAPM':
              st.markdown('Tomando en cuenta el cálculo del CAPM')
              st.write(f'El resultado del CAPM es de: {CAPM}')
              st.markdown('Adicionalmente y análogo al inciso del WACC, tenemos que:')
              
              VF_perpetuidad=VF_5*(1+0.03)/(CAPM-0.03)
              st.write(f'El valor futuro de la perpetuidad es de: ${VF_perpetuidad:,.2f}')
              VP_perpetuidad=VF_perpetuidad*(1+CAPM)**(-5)
              st.markdown('Y de mismo modo obtenemos que: ')
              st.write(f'El valor presente de la perpetuidad considerando la tasa de CAPM es de: ${VP_perpetuidad:,.2f}')
              vc_VF_5=[]
              for i in range(1,6):
                  if i>=4:
                      valor=valor*(1+0.05)
                      vc_VF_5.append(round(valor,2))
                  else:
                      valor=FCF*(1+.10)**(i)
                      vc_VF_5.append(round(valor,2))
                      
              vec_VP_5=[vc_VF_5[i-1]*(1+CAPM)**(-i) for i in range(1,len(vc_VF_5)+1)]
              total_final=sum(vec_VP_5)+VP_perpetuidad
  
              st.markdown('Finalmente siguiendo la misma lógica que en el inciso del WACC, obtenemos que:')
              st.write(f'El valor final de la empresa considerando CAPM: ${total_final:,.2f}')
  
          elif opcion=='Conclusión':
              total_final=st.markdown('''
              En conclusión la valuación mediante el uso del CAPM nos da un monto mayor, esto
              debido a que la tasa de descuento es menor, lo cual ayuda a aumentar el valor presente
              de los flujos de efectivo.
              ''')
          
          return total_final
  
      select_value = st.selectbox('Selecciona la opción:', ['WACC','CAPM','Conclusión'])
      
      select_tasa(select_value)
  
  elif option == 'Gráficas':
      st.title('Gráficas')
      st.write('Use las gráficas y formato que desee para presentar la siguiente información a un cliente.')
      st.title('Respuesta')
      st.markdown('''
                  Para la solucón de este inciso se extrajo la información directamente del archivo,
                  además con ayuda de la implementación de streamlit podemos tener una interacción 
                  con los gráficos.
                  ''')
      st.subheader('Gráfico 1')
      st.write('''Para el primer gráfico se puede interactuar con los meses, de modo que si se deseara 
              algún mes en específico, se puede escoger.''')
      
      data_g=pd.read_excel('Ejercicio Inversiones Finaccess.xlsx',sheet_name='Gráficas')
      
      data_g1=data_g[['Unnamed: 1','Unnamed: 2','Unnamed: 3']][7:]
      data_g1.columns='Mes Rendimiento AUM(millions)'.split()
      #data_g1
  
      meses=data_g1['Mes'].tolist()
      select_meses=st.multiselect('Seleccione los meses a visualizar',meses,default=meses)
      data_g1_filt=data_g1[data_g1['Mes'].isin(select_meses)]
      st.dataframe(data_g1_filt)
      
      fig_1, ax_1 = plt.subplots(figsize=(12, 6))
      
      ax_1.bar(data_g1_filt['Mes'], data_g1_filt['AUM(millions)'], label='AUM',color='#49d2fa')
      ax_1.set_title('Rendimientos Acumulados y AUM(millions)')
      ax_1.set_xlabel('Mes')
      ax_1.set_ylabel('AUM(millions)',color='#49d2fa')
      ax_1.xaxis.grid(True)
      ax_1.grid()
      
      ax2 = ax_1.twinx()
      ax2.plot(data_g1_filt['Mes'], data_g1_filt["Rendimiento"], color='#fa5c49', marker='o', label='Rendimiento (%)')
      ax2.set_ylabel('Rendimiento (%)', color='#fa5c49')
      ax2.legend(title='Rendimiento')
      plt.xticks(rotation=45)
      
      # Mostrar el gráfico en Streamlit
      st.pyplot(fig_1)
  
      st.subheader('Gráfico 2')
      st.markdown('''
                  Para el segundo gráfico se observa que se puede interactuar con el tipo de activo,
                  de modo que si de desea ver algún conjunto en especial se puede escoger y observar
                  el porcentaje de ponderado que tiene cada activo.
                  ''')
      data_g2=data_g[['Unnamed: 6','Unnamed: 7','Unnamed: 8']][7:14]
      data_g2.columns='Activo Weight TipoActivo'.split()
      #data_g2
      
      opciones = data_g2['TipoActivo'].unique()
      seleccion = st.multiselect('Selecciona los tipos de activos:', opciones, default=opciones)
      
      df_filtrado = data_g2[data_g2['TipoActivo'].isin(seleccion)]
      
      # Mostrar el DataFrame filtrado
      st.dataframe(df_filtrado)
      
      fig, ax = plt.subplots(figsize=(10, 6))
      ax.bar(df_filtrado['Activo'], df_filtrado['Weight'], color='skyblue')
      ax.set_title('Distribución de Weights por Activo')
      ax.set_xlabel('Activo')
      ax.set_ylabel('Weight')
      plt.xticks(rotation=45)
      plt.grid(True, linestyle='--', alpha=0.5)
      ax.yaxis.grid()
      ax.xaxis.grid()
      
      # Mostrar el gráfico en Streamlit
      st.pyplot(fig)
  elif option == 'Comentario':
      st.title('Comentario')
      st.write('5. Redacte (en máximo tres párrrafos) un comentario económico financiero sobre el mercado mexicano actual.')
      st.subheader('Respuesta')
  
      st.markdown('''
      Actualmente la economía mexican se enfrenta a un escenario desafiante debido a factores internos y
      externos. Analistas esperan un crecimiento del PIB de 1.3% para este 2025 con la estimación de
      de una inflación anual del 4% y además con un pronóstico sobre el tipo de cambio en un aproximado 
      de $20.5.
  
      Sin embargo, la economía mexican aún se sigue viendo afecatada por la dependencia de otros países
      como Estados Unidos. Las recientes amenazas sobre el incremento de los arancales hasta un 25%,
      incluyendo materias como el acero, aluminio y vehículos, podrían tener un impacto negativo sobre
      el comercio e inversión.
      
      Ante la incertidumbre de este escenario, es de suma importancia que México diversifique sus 
      mercados de exportación y fortalezca su economía interna para mitigar los 
      riesgos asociados por factores internacionales y empezar a ver alternativas internas, como el 
      apoyo a las Pymes ya que estas generan hasta un 70% aproximado del empleo formal en el país y 
      además tienen un gran impacto en el Producto Interno Bruto (PIB) de hasta un 50% en donde 
      destacan los sectores del comercio, servios y manufactura.
  
                  ''')


elif ejercicio =='Ejercicio Programación Finaccess':
# # Ejercico de Programación

# In[ ]:

##ejercicio 2
  st.sidebar.radio("Ejercicio Programación Finaccess")
  
  inflation=data_2[['Instrucciones:','Unnamed: 1']][7:302]
  inflation.reset_index(inplace=True,drop=True)
  inflation.columns=['Date','US Inflation(%)']
  inflation['Date']=pd.to_datetime(inflation['Date'])
  inflation.set_index('Date', inplace=True)
  
  gdp=data_2[['Unnamed: 3','Unnamed: 4']][7:106]
  gdp.reset_index(inplace=True,drop=True)
  gdp.columns=['Date','US GDP(%)']
  gdp['Date']=pd.to_datetime(gdp['Date'])
  gdp.set_index('Date', inplace=True)
  
  col_names=data_2[data_2.columns[6:]].iloc[6:7].values
  indices=data_2[data_2.columns[6:]][7:len(data_2)]
  indices.reset_index(inplace=True,drop=True)
  indices.columns=col_names[0]
  indices['Date']=pd.to_datetime(indices['Date'])
  #indices['Date']=indices['Date'].map(lambda x:x.strftime("%d-b-%Y"))
  indices.set_index('Date', inplace=True)
  
  
  # In[ ]:
  
  
  #se ajustan los dataframes para que la info se tenga trimestral usando la media
  inflation=inflation.resample('QE').mean()
  indices=indices.resample('QE').mean()
  #para el caso del gdp no sera necesario ya que tenemos la info trimestral
  
  #se hace un merge parea tener un solo dataframe
  data_f=inflation.join([gdp,indices],how='inner')
  
  #se cambia el tipo de dato de los indices
  data_f=data_f.apply(pd.to_numeric, errors='coerce')
  
  #verificamos que no existan vacios
  #data_f.isna().sum()
  
  
  # In[ ]:
  
  
  #creamos una funcion la cual nos ayude a clasificar a que escenario pertenece
  def escenario(gdp, inflation):
      if gdp > 0 and inflation < 0:
          return 'Goldilocks'
      elif gdp > 0 and inflation > 0:
          return 'Reflation'
      elif gdp < 0 and inflation > 0:
          return 'Stagflation'
      elif gdp < 0 and inflation < 0:
          return 'Deflation'
      else:
          return 'Indefinido'
  
  data_f['Escenario']=data_f.apply(lambda x:escenario(x['US GDP(%)'],x['US Inflation(%)']),axis=1)
  
  
  # In[ ]:
  
  
  def metricas(escenario):
      if 'seed' not in st.session_state:
          st.session_state.seed = 2025
      np.random.seed(st.session_state.seed)
      
      aux=data_f[data_f['Escenario']==escenario]
      
      if st.checkbox(f'Mostrar DataFrame filtrado por: {escenario}'):
          st.dataframe(aux)
          
      aux_retornos=np.log(aux[aux.columns[2:-1].tolist()]).diff().dropna()
      medias=aux_retornos[aux.columns[2:-1].tolist()].mean().to_frame(name='Media')
      desv_est=aux_retornos[aux.columns[2:-1].tolist()].std().to_frame(name='Std')
      df=medias.join([desv_est],how='inner')
      
      #tasa anual de 1 año
      Rf=0.0931
      #tasa trimestral
      Rf_tri=round((1+Rf)**(1/4)-1,4)
      
      SR=pd.DataFrame()
      SR[f'SharpeRatio_{escenario}']=df.apply(lambda x:round((x['Media']-Rf_tri)/x['Std'],4),axis=1)
  
      metodo=st.selectbox('Escoge el método',['MonteCarlo','Paramétrico','Histórico'])
      alpha=st.radio('Escoge la alpha',[99,97.5,95],horizontal=True)
  
      if metodo=='MonteCarlo':
          st.markdown('''- **Monte Carlo:** Simula posibles escenarios futuros generando múltiples trayectorias de precios 
          con distribuciones estadísticas, evaluando riesgos y rendimientos.''')
          st.markdown('''
              - **VaR (Value at Risk):** Máxima pérdida esperada en un horizonte de tiempo dado con un nivel de confianza específico.  
              - **CVaR (Conditional Value at Risk):** Promedio de pérdidas que exceden el VaR, evaluando escenarios más extremos.  
              - **Sharpe Ratio:** Mide el rendimiento ajustado por riesgo, comparando el exceso de retorno sobre la tasa libre de riesgo respecto a la volatilidad.  
                      ''')
          VaR_MC=pd.DataFrame()
          VaR_MC['VaR_MonteCarlo']=df.apply(lambda x:np.percentile(np.random.normal(np.mean(x['Media']),x['Std'],1000),100-alpha),axis=1)
          VaR_MC['CVaR_MonteCarlo']=aux_retornos[aux_retornos<=VaR_MC['VaR_MonteCarlo']].mean()
          VaR_MC=VaR_MC.join([SR],how='inner')
          #___
          opciones = VaR_MC.columns.tolist()
          seleccion = st.multiselect('Selecciona los activos:', opciones, default=opciones)
          df_filtrado = VaR_MC[seleccion]
          
          
          # Mostrar el DataFrame filtrado
          st.dataframe(df_filtrado)
          
          fig, ax = plt.subplots(figsize=(10, 6))
          ax.plot(df_filtrado.index,df_filtrado[seleccion],label=seleccion)
          ax.set_title('Distribución de Métricas por Activo')
          ax.set_xlabel('Activo')
          ax.set_ylabel('Valor métricas')
          ax.legend(title='Métricas')
          plt.xticks(rotation=45)
          plt.grid(True, linestyle='--', alpha=0.5)
          ax.yaxis.grid()
          ax.xaxis.grid()
          #___
          return st.pyplot(fig)
          
      elif metodo=='Paramétrico':
          st.markdown('- **Paramétrico:** Calcula el riesgo asumiendo que los rendimientos siguen una distribución normal, usando media y desviación estándar.')
          st.markdown('''
              - **VaR (Value at Risk):** Máxima pérdida esperada en un horizonte de tiempo dado con un nivel de confianza específico.  
              - **CVaR (Conditional Value at Risk):** Promedio de pérdidas que exceden el VaR, evaluando escenarios más extremos.  
              - **Sharpe Ratio:** Mide el rendimiento ajustado por riesgo, comparando el exceso de retorno sobre la tasa libre de riesgo respecto a la volatilidad.  
                      ''')
          Prm=pd.DataFrame()
          Prm['VaR_Paramétrico']=df.apply(lambda x:round(norm.ppf((100-alpha)/100,x['Media'],x['Std']),4)*100,axis=1)
          Prm['CVaR_Paramétrico']=aux_retornos[aux_retornos<=Prm['VaR_Paramétrico']].mean()
          Prm=Prm.join([SR],how='inner')
          #___
          opciones = Prm.columns.tolist()
          seleccion = st.multiselect('Selecciona los activos:', opciones, default=opciones)
          df_filtrado = Prm[seleccion]
          
          
          # Mostrar el DataFrame filtrado
          st.dataframe(df_filtrado)
          
          fig, ax = plt.subplots(figsize=(10, 6))
          ax.plot(df_filtrado.index,df_filtrado[seleccion],label=seleccion)
          ax.set_title('Distribución de Métricas por Activo')
          ax.set_xlabel('Activo')
          ax.set_ylabel('Valor métricas')
          ax.legend(title='Métricas')
          plt.xticks(rotation=45)
          plt.grid(True, linestyle='--', alpha=0.5)
          ax.yaxis.grid()
          ax.xaxis.grid()
          #___
          return st.pyplot(fig)
          
      elif metodo=='Histórico':
          st.markdown('- **Histórico:** Evalúa el riesgo analizando rendimientos pasados, sin suponer una distribución específica.')
          st.markdown('''
              - **VaR (Value at Risk):** Máxima pérdida esperada en un horizonte de tiempo dado con un nivel de confianza específico.  
              - **CVaR (Conditional Value at Risk):** Promedio de pérdidas que exceden el VaR, evaluando escenarios más extremos.  
              - **Sharpe Ratio:** Mide el rendimiento ajustado por riesgo, comparando el exceso de retorno sobre la tasa libre de riesgo respecto a la volatilidad.  
                      ''')
          Hist=pd.DataFrame()
          Hist['VaR_Histórico']=aux_retornos.quantile((100-alpha)/100)
          Hist['CVaR_Histórico']=aux_retornos[aux_retornos<=Hist['VaR_Histórico']].mean()
          Hist=Hist.join([SR],how='inner')
          #___
          opciones = Hist.columns.tolist()
          seleccion = st.multiselect('Selecciona los activos:', opciones, default=opciones)
          df_filtrado = Hist[seleccion]
          
          
          # Mostrar el DataFrame filtrado
          st.dataframe(df_filtrado)
          
          fig, ax = plt.subplots(figsize=(10, 6))
          ax.plot(df_filtrado.index,df_filtrado[seleccion],label=seleccion)
          ax.set_title('Distribución de Métricas por Activo')
          ax.set_xlabel('Activo')
          ax.set_ylabel('Valor métricas')
          ax.legend(title='Métricas')
          plt.xticks(rotation=45)
          plt.grid(True, linestyle='--', alpha=0.5)
          ax.yaxis.grid()
          ax.xaxis.grid()
          #___
          return st.pyplot(fig)
          
  
  
  # In[ ]:
  
  
  option_2 = st.sidebar.selectbox("Seleccione el escenario a visualizar", 
                            ['','Introducción','Goldilocks','Reflation','Stagflation','Deflation'])
  
  if option_2 == 'Introducción':
      st.title('Introducción')
      st.write('''En esta entrega se busca mostrar la solución a los ejercicios con una implememtación 
      y aplicación de Streamlit para una mejor vizualización de los resultados.''')
      st.markdown('''
              La siguiente base de datos contiene el dato de inflación mensual y el GDP trimestral 
              de USA, así como los precios semanales de diversos índices accionarios.
              
              En la economía podemos definir 4 escenarios basados en la inflación y el GDP:
              
              - Goldilocks: crecimiento económico con inflación desacelerando.
              - Reflation: crecimiento económico con inflación acelerando.
              - Stagflation: decrecimiento económico con inflación acelerando.
              - Deflation: decrecimiento económico con inflación desacelerando.
      
              Utilizando la herramienta de programación de tu preferencia, analiza el comportamiento 
              de los diferentes índices en los 4 escenarios económicos. Puedes usar las métricas que 
              consideres relevante (rendimiento, volatilidad, Sharpe, distribución, etc.)
      
              Envía un notebook o un pdf con los resultados de tu análisis antes de la fecha límite.
      
              En la entrevista presencial presentarás y explicarás el proceso de tu trabajo para llegar a tus conclusiones.
              ''')
  elif option_2 == 'Goldilocks':
      st.title('Goldilocks')
      st.subheader('Crecimiento económico con inflación desacelerando')
      st.markdown('''
              - Características: La economía crece a un ritmo saludable sin generar presiones inflacionarias 
               significativas. Esto permite a los bancos centrales mantener políticas monetarias acomodaticias, 
               fomentando la inversión y el consumo.
               
              - Implicaciones: Es un entorno positivo para los mercados accionarios, ya que el crecimiento de 
              utilidades corporativas ocurre sin el aumento de costos por inflación o alzas agresivas de tasas 
              de interés.
                  ''')
      
      metricas(option_2)
      st.write('''
      Se puede observar que debido a la falta de información la mayoría de los datos no cumplen con las características
      para los métodos de MonteCarlo y Paramétrico
      ''')
  elif option_2 == 'Reflation':
      st.title('Reflation')
      st.subheader('Crecimiento económico con inflación acelerando')
      st.markdown('''
              - Características: La economía se expande rápidamente, pero con una inflación en aumento. 
              Esto puede ocurrir después de una recesión o una desaceleración prolongada, impulsada por 
              políticas fiscales o monetarias expansivas.
              
              - Implicaciones: Al principio, es positivo para los activos de riesgo como las acciones, 
              pero puede llevar a ajustes en la política monetaria (alzas de tasas), afectando negativamente 
              a bonos y sectores sensibles a tasas de interés.
                  ''')
      metricas(option_2)
      st.markdown('''Podemos observar que la mayoría de los datos se encuentran en este escenario, por lo tanto 
      nos da más resultados respecrto a las métricas y los métodos.''')
  elif option_2 == 'Stagflation':
      st.title('Stagflation')
      st.subheader('Decrecimiento económico con inflación acelerando')
      st.markdown('''
              - Características: La economía se contrae o crece muy lentamente mientras los precios continúan 
              aumentando. Esto suele ser causado por shocks de oferta (ej. aumento de precios de materias 
              primas).
              
              - Implicaciones: Es el peor escenario para los mercados, ya que la combinación de inflación 
              alta con crecimiento bajo limita la capacidad de los bancos centrales para estimular la 
              economía sin exacerbar la inflación.
                  ''')
      metricas(option_2)
      st.write('A pesar la poca información sus datos nos brindan infromación respecto a las diferentes métricas.')
  elif option_2 == 'Deflation':
      st.title('Deflation')
      st.subheader('Decrecimiento económico con inflación desacelerando o negativa')
      st.markdown('''
              - Características: La actividad económica se desacelera y los precios disminuyen. Esto 
              puede llevar a una espiral deflacionaria si las expectativas de precios bajos posponen 
              el consumo y la inversión.
              
              - Implicaciones: Es negativo para la mayoría de los activos de riesgo, ya que las 
              utilidades corporativas caen. Sin embargo, los bonos de alta calidad suelen beneficiarse 
              de un entorno deflacionario.
                  ''')
      st.write('''Este escenario es similar al de Stagflation, ya que son pocos los datos que pertenecen,
      sin embargo nos muestran más resultados que el escenario Goldilocks.''')
      metricas(option_2)
  
