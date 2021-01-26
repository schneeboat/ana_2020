# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from datetime import datetime
import nltk
import networkx as nx
import base64
import requests
import xlrd
from matplotlib.dates import DateFormatter
import openpyxl

st.title('Bibliographic analysis on covid-19 related publications in 2020 (BETA)')
st.write('For full analysis, refer to my repo https://github.com/schneeboat/ana_2020')
st.write('Since python library wordcloud has some build error for macOS which hasn\'t been resolved for so long, pictures of generated wordcloud from colab will be pasted.')
st.write('Given the long running time for building networks and calculating centrality measures, only calculated results will be pasted from colab.')
st.subheader('Introduction')
st.write('The main purpose of this analysis is to identify the research trend, collaboration pattern, most influential elements, etc. from publications related to covid-19 in 2020. All the data used for analysis were retrieved from Web of Science(WoS), a database that provides comprehensive citation data for many different academic disciplines. From WoS core collection, topic keywords: covid OR coronavirus OR covid-19 OR covid19 OR 2019-nCoV OR SARS-CoV-2 were selected, with time span ranging from 2020 to 2020. 80,050 records were downloaded. Data retrieving date: 26 Jan, 2021.')
st.write('Since the outbreak of covid-19, there has been many researchers studying this new type of virus that claims many lives in the world. As a result, many scientific publications have been published. Here, we only focus on the scientific publications that were published in 2020. We want to answer the following questions:')
st.write('1. What are the most popular research areas? What does the quantitative output look like?')
st.write('2. What are the most cited papers? And what are the corresponding research areas?')
st.write('3. Which author has the most significant impact?')
st.write('4. In the co-authorship network, what are the author pairs that collaborate most frequently?')
st.write('5. How does the ratio of multidisciplinary paper change over time? Is there any relation to the collaboration pattern?')
st.write('6. What are the keywords that describe the topics for each period? Is there any switch of research focus?')
st.write('...')

st.write('At the beginning, a few descriptive visualizations will be shown, then there will be wordcloud visualizations, lastly, analysis of 4 major networks from the data will be shown.')
st.subheader('Descriptive data analysis')

url= 'https://raw.githubusercontent.com/schneeboat/ana_2020/main/data.xlsx'
wb_obj = openpyxl.load_workbook(url)
sheet_obj = wb_obj.active

data = pd.DataFrame(sheet_obj)
mon = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
data.PD = data.PD.str[:3].str.capitalize()
data_w_date = data[data.PD.isin(mon)].copy()
data_w_date.PD = data_w_date.PD.apply(lambda x: datetime.strptime(x,'%b').strftime('2020-%m'))

#doc type
data_doc = data.DT.dropna().str.split(';').apply(lambda x: x[0]).value_counts().rename_axis('DocType').reset_index(name='Count')

with plt.style.context({'axes.prop_cycle' : plt.cycler('color', plt.cm.Set3.colors)}):
    fig1, ax1 = plt.subplots()
    labels = data_doc.iloc[:,0]
    size = data_doc.iloc[:,1]/data_doc.iloc[:,1].sum()*100
    ax1.pie(data_doc.iloc[:,1], startangle=90, shadow=True)
    ax1.legend(labels =['%s, %1.1f%%' % (l,s) for l,s in zip(labels,size)])
    ax1.set_title('Document Type Overview', fontsize=10, loc='left')
    plt.tight_layout()
    st.pyplot(fig1)
st.write('')

#language
data_lang = data.LA.dropna().value_counts().rename_axis('Language').reset_index(name='Count')[0:10]

fig2, ax2 = plt.subplots(figsize=(12,6))
ax2.bar(data_lang.Language, data_lang.Count, color='thistle')
for p in ax2.patches:
         ax2.annotate("%1.0f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
         ha='center', va='center', fontsize=11, color='slategrey',xytext=(0, 5), textcoords='offset points')
plt.xlabel('Language',fontsize=15)
plt.ylabel('Number of Publications',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('Number of Publications by Language', fontsize=19)
plt.tight_layout()
st.pyplot(fig2)
st.write('')

#source
data_source = data.JI.dropna().value_counts().rename_axis('Title').reset_index(name='Count')[0:10]
data_source['Title'] = data_source.Title.astype('category')
fig3, ax3 = plt.subplots(figsize=(12,7.5))
ax3.bar(data_source.Title, data_source.Count, color='thistle')
plt.xlabel('Source',fontsize=15)
plt.ylabel('Number of Publications',fontsize=15)
plt.xticks(fontsize=15, rotation = -45, ha='left', rotation_mode='anchor')
plt.yticks(fontsize=15)
for p in ax3.patches:
             ax3.annotate("%1.0f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
             ha='center', va='center', fontsize=11, color='slategrey',xytext=(0, 5), textcoords='offset points')
plt.title('Number of Publications by Source', fontsize=19)
plt.tight_layout()
st.pyplot(fig3)
st.write('')

#pagecount
data_page = data.PG.dropna().copy()

fig4, ax4 = plt.subplots(figsize=(12,5.7))
ax4.hist(data_page, color='thistle', bins=120)
plt.xlabel('Number of Pages',fontsize=15)
plt.ylabel('Frequency',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('Number of Pages per Publication', fontsize=19)
plt.tight_layout()
st.pyplot(fig4)
st.write('')

#institution
data_inst = data[['C1']].dropna().copy()
data_inst.C1= data_inst.C1.apply(lambda x: re.findall(r"\] (.*?)\,", x))
data_inst_all = data_inst.explode('C1').C1.value_counts().rename_axis('Institutions').reset_index(name='Count')[0:10]

fig5, ax5 = plt.subplots(figsize=(16,9))
ax5.bar(data_inst_all.Institutions, data_inst_all.Count, color='thistle')
plt.xlabel('Institutions',fontsize=15)
plt.ylabel('Number of Publications',fontsize=15)
plt.xticks(fontsize=15, rotation = -45, ha='left', rotation_mode='anchor')
plt.yticks(fontsize=15)
for p in ax5.patches:
             ax5.annotate("%1.0f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
             ha='center', va='center', fontsize=11, color='thistle',xytext=(0, 5), textcoords='offset points')
plt.title('Number of Publications by Institution', fontsize=19)
plt.tight_layout()
st.pyplot(fig5)
st.write('')

#country
data_country = data[['C1']].dropna().copy()
data_country.C1 = data_country.C1.apply(lambda x: re.sub(r"\[(.*?)\] ", "", x).split('; ')).to_list()
data_country['country']=[list(set(i)) for i in [[j.split(', ')[-1] for j in i] for i in data_country.C1]] 

def replace(string):
  if 'USA' in string:
    return 'USA'
  elif 'North Ireland' in string:
    return 'UK'
  elif 'Wales' in string:
    return 'UK'
  elif 'Scotland' in string:
    return 'UK'
  elif 'England' in string:
    return 'UK'
  elif 'P. R. China' in string:
    return 'China'
  elif 'Peoples R China' in string:
    return 'China'

  else:
    return string
data_country['replace_1'] = [[replace(i) for i in j] for j in data_country['country']]
data_country['replace'] = [list(set(i)) for i in data_country['replace_1']]

data_country_all = data_country.explode('replace')['replace'].value_counts().rename_axis('Countries').reset_index(name='Count')
data_country_10 = data_country_all[0:10]

fig6, ax6 = plt.subplots(figsize=(15,8))
ax6.bar(data_country_10.Countries, data_country_10.Count, color='thistle')
plt.xlabel('Countries',fontsize=15)
plt.ylabel('Number of Publications',fontsize=15)
plt.xticks(fontsize=15, rotation = -45)
plt.yticks(fontsize=15)
for p in ax6.patches:
             ax6.annotate("%1.0f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
             ha='center', va='center', fontsize=11, color='slategrey',xytext=(0, 5), textcoords='offset points')
plt.title('Number of Publications by Countries', fontsize=19)
plt.tight_layout()
st.pyplot(fig6)
st.write('')

#ra
data_research = data[['SC']].dropna().copy()
data_research.SC = data_research.SC.str.split('; ')
data_research_all = data_research.explode('SC').SC.value_counts().rename_axis('ResArea').reset_index(name='Count')[:10]

fig7, ax7 = plt.subplots(figsize=(17,10))
ax7.bar(data_research_all.ResArea, data_research_all.Count, color='thistle')
plt.xlabel('Research Area',fontsize=15)
plt.ylabel('Number of Publications',fontsize=15)
plt.xticks(fontsize=15, rotation = -45, ha='left', rotation_mode='anchor')
plt.yticks(fontsize=15)
for p in ax7.patches:
             ax7.annotate("%1.0f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
             ha='center', va='center', fontsize=11, color='slategrey',xytext=(0, 5), textcoords='offset points')

plt.title('Number of Publications by Research Area', fontsize=19)
plt.tight_layout()
st.pyplot(fig7)
st.write('')

#kw
data_keywords=data[['DE']].dropna().copy()
data_keywords.DE = data_keywords.DE.str.split('; ')
data_keywords_full = data_keywords.explode('DE').DE.str.lower().value_counts().rename_axis('Author Keywords').reset_index(name='Count')
st.dataframe(data_keywords_full.head(10))
st.write('')

#author tb
data_author = data[['AF']].dropna().copy()
data_author = data_author.loc[data_author.AF != '[Anonymous]'].copy()
data_author.AF = data_author.AF.str.split('; ')
data_author_all = data_author.explode('AF').AF.value_counts().rename_axis('AF').reset_index(name='Number of papers')
author_cited = data[['AF','Z9','SC']].dropna()
author_cited = author_cited.loc[author_cited.AF != '[Anonymous]'].copy()
author_cited.AF = author_cited.AF.str.split(';')
author_cited_all = author_cited.explode('AF').sort_values('Z9', ascending=False).drop(columns = 'SC').reset_index().drop(columns = 'index')
author_cited_all = author_cited_all[['AF', 'Z9']]
au_ci = author_cited_all.groupby('AF').sum().sort_values('Z9', ascending=False)
au_ci_1 = author_cited_all.groupby('AF').count().reset_index().rename(columns={'Z9':'Count'})
au_ci_per = pd.merge(au_ci, au_ci_1, on='AF', how='inner')
au_ci_per['Number cited per paper'] = au_ci_per['Z9']/au_ci_per['Count']
au_ci_per = au_ci_per.drop(columns=['Z9', 'Count']).sort_values('Number cited per paper', ascending=False)
author_tb = pd.merge(data_author_all, au_ci, on='AF', how='inner').dropna()
author_tb['Cited per paper'] = author_tb.Z9/author_tb['Number of papers']
author_tb = author_tb[['AF', 'Z9', 'Cited per paper', 'Number of papers']].rename(columns={'AF':'Author'})
atb = author_tb.sort_values(by=['Z9', 'Cited per paper', 'Number of papers'], ascending=False).reset_index(drop=True).rename(columns={"Z9": "Total number cited"})
st.dataframe(atb.head(10))
st.write('')


#most cited ppr
mcp = data[['TI', 'SC', 'Z9']].dropna().copy().sort_values('Z9', ascending=False).reset_index(drop=True)
mcp = mcp.rename(columns={'TI':'Document title','SC': 'Research area', 'Z9': 'Total number cited'})
st.dataframe(mcp.head(10))
st.write('')

#num author/ppr
data_author_wo_anony = data_author.loc[data_author.AF != '[Anonymous]'].copy()
author_per_ppr = data_author_wo_anony.AF.apply(lambda x: len(x))
author_per_ppr_df = author_per_ppr.value_counts().rename_axis('Number of Authors').reset_index(name='Count').sort_values('Number of Authors')

fig8, ax8 = plt.subplots(figsize=(12,6))
ax8.bar(author_per_ppr_df['Number of Authors'].astype('category'),author_per_ppr_df['Count'], color='thistle')
plt.xlabel('Number of Authors',fontsize=15)
plt.ylabel('Frequency',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('Number of Authors per Publication', fontsize=19)
plt.tight_layout()
st.pyplot(fig8)
st.write('')


from matplotlib import dates
#int colab+mulridis
int_colab = data_w_date[['PD','C1']].dropna().copy()
int_colab['C1'] = int_colab['C1'].apply(lambda x: re.sub(r"\[(.*?)\]", "", x).split('; ')).to_list()
int_colab['country_whole']=[list(set(i)) for i in [[j.split(', ')[-1] for j in i] for i in int_colab['C1']]]
int_colab['replace'] = [[replace(i) for i in j] for j in int_colab['country_whole']]
int_colab['replace'] = [list(set(i)) for i in int_colab['replace']]
int_colab['num_country'] = int_colab['replace'].apply(lambda x: len(x))
int_colab.loc[int_colab['num_country']>1, 'num_country'] = 2
int_colab_counts = int_colab.groupby('PD')['num_country'].value_counts().rename_axis(['date','collab']).reset_index(name='Count').pivot(index='date', columns='collab', values='Count').fillna(0).reset_index()
int_colab_counts['percentage_inter_colab']=int_colab_counts[2]/(int_colab_counts[1]+int_colab_counts[2])

ra_sm = data_w_date[['PD','SC']].dropna().copy()
ra_sm['SC'] = ra_sm['SC'].str.split(';')
ra_sm['cnt'] = ra_sm['SC'].apply(lambda x: len(x))

ra_sm.loc[ra_sm['cnt']>1, 'cnt'] = 2
ra_sm_counts = ra_sm.groupby('PD')['cnt'].value_counts().rename_axis(['date','ra']).reset_index(name='Count').pivot(index='date', columns='ra', values='Count').fillna(0).reset_index()
ra_sm_counts['percentage_multi_disp']=ra_sm_counts[2]/(ra_sm_counts[1]+ra_sm_counts[2])
with plt.style.context({'axes.prop_cycle' : plt.cycler('color', plt.cm.Set3.colors)}):
    fig13, ax13 = plt.subplots(figsize=(12,6))
    ax13.plot_date(dates.date2num(int_colab_counts['date']), int_colab_counts['percentage_inter_colab'], label='International collab', color='springgreen',ls='-')
    ax13.plot_date(dates.date2num(ra_sm_counts['date']), ra_sm_counts['percentage_multi_disp'], label='Multidisciplinary', color='salmon',ls='-')
    plt.xlabel('Date',fontsize=15)
    plt.ylabel('Ratio',fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend()
    plt.tight_layout()
    st.pyplot(fig13)
st.write('')

#by month
date_count = data_w_date.sort_values(by = 'PD').groupby('PD').size().rename_axis('Date').reset_index(name='Count')

dates9 = dates.date2num(date_count.Date)
fig9, ax9 = plt.subplots(figsize=(12,6))
ax9.plot_date(dates9, date_count.Count,color='darksalmon', ls='-')
plt.xlabel('Date',fontsize=15)
plt.ylabel('Number of Publications',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('Number of Publications over Months', fontsize=19)
plt.tight_layout()
st.pyplot(fig9)

#country/month
country_month = data_w_date[['C1', 'PD']].dropna().copy()
country_month.C1 = country_month.C1.apply(lambda x: re.sub(r"\[(.*?)\]", "", x).split('; ')).to_list()
country_month['country']=[list(set(i)) for i in [[j.split(', ')[-1] for j in i] for i in country_month.C1]]     
country_month['replace'] = [[replace(i) for i in j] for j in country_month['country']]
country_month['replace'] = [list(set(i)) for i in country_month['replace']]                                                 
country_month_all = country_month.explode('replace').drop(columns=['C1','country'])
country_month_all['num']=1
country_month_10 = country_month_all[country_month_all['replace'].isin(data_country_10['Countries'].to_list())].groupby(['PD','replace']).count().reset_index()
country_month_10['replace'] = country_month_10['replace'].astype('category')

fig10 = plt.figure(figsize=(11,5))
sns.lineplot(x=country_month_10['PD'], y=country_month_10['num'], hue=country_month_10['replace'],
            hue_order=data_country_10.Countries.to_list(),
            palette='Set2')
plt.xlabel('Date',fontsize=12)
plt.ylabel('Number of Publications',fontsize=12)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.title('Top 10 Countries over Months', fontsize=15)
plt.tight_layout()
st.pyplot(fig10)
st.write('')

#source/mon
pub_month = data_w_date[['JI', 'PD']].dropna().copy()
pub_month['Number']=1
pub_month_count_10 = pub_month[pub_month['JI'].isin(data_source['Title'].to_list())].groupby(['PD','JI']).count().reset_index()
pub_month_count_10['JI'] = pub_month_count_10['JI'].astype('category')

fig11=plt.figure(figsize=(11,5))
sns.lineplot(x=pub_month_count_10['PD'], y=pub_month_count_10['Number'], hue=pub_month_count_10['JI'],
            hue_order=data_source['Title'].to_list(),
            palette='Set2')
plt.xlabel('Date',fontsize=12)
plt.ylabel('Number of Publications',fontsize=12)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.title('Top 10 Publishers over Months', fontsize=15)
plt.tight_layout()
st.pyplot(fig11)
st.write('')

#ra/mon
research_month = data_w_date[['SC','PD']].dropna().copy()
research_month['SC'] = research_month['SC'].str.split('; ')
research_month_all = research_month.explode('SC')
research_month_all['num']=1
research_month_10 = research_month_all[research_month_all.SC.isin(data_research_all['ResArea'].to_list())].groupby(['PD','SC']).count().reset_index()
research_month_10['SC'] = research_month_10['SC'].astype('category')
fig12=plt.figure(figsize=(11,5))
sns.lineplot(x=research_month_10['PD'], y=research_month_10['num'], 
             hue=research_month_10['SC'],
            hue_order=data_research_all['ResArea'].to_list(),
            palette='Set2')
plt.xlabel('Date',fontsize=12)
plt.ylabel('Number of Publications',fontsize=12)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.title('Top 10 Research Areas over Months', fontsize=15)
plt.tight_layout()
st.pyplot(fig12)
st.write('')

st.subheader('Correlation matrix')

#corr
corr_ana = data[['AF', 'NR', 'Z9', 'PG', 'SC', 'C1']].dropna().copy()
corr_ana['Number of Authors']=corr_ana['AF'].str.split(';').apply(lambda x: len(x))
corr_ana['Number of Research Areas']=corr_ana['SC'].str.split(';').apply(lambda x: len(x))
corr_ana['addresses'] = corr_ana['C1'].apply(lambda x: re.sub(r"\[(.*?)\]", "", x).split('; ')).to_list()
country_replace_corr = [[i.replace('North Ireland', 'UK').replace('Wales','UK').replace('Scotland', 'UK').replace('England', 'UK').replace('Czech Republic','Czech') 
for i in x] for x in corr_ana['addresses']]
corr_ana['country_whole']=[list(set(i)) for i in [[j.split()[-1] for j in i] for i in country_replace_corr]]
corr_ana['Number of Countries'] = corr_ana['country_whole'].apply(lambda x: len(x))

corr= corr_ana[['Number of Authors', 'NR', 'Z9', 'PG', 'Number of Research Areas', 'Number of Countries']].rename(
    columns={"NR": "Cited Reference Count", "Z9": "Total Times Cited Count", "PG": "Page Count"}).corr()

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask, k=1)] = True
sns.set(style="white")

f, ax = plt.subplots(figsize=(5, 5))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

g = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1,
            square=True, annot=True, fmt = '.2f',
            linewidths=.5, cbar_kws={"shrink": .8}, ax=ax )

locs, labels = plt.yticks();
g.set_yticklabels(labels, rotation=0, size='medium')
locs, labels = plt.xticks();
g.set_xticklabels(labels, rotation=90, size='medium')
st.pyplot(f)
st.write('')

st.write('h-index and g-index are author-level metrics to measure both the productivity and citation impact of publications of an author. The higher the number, the greater the impact.')

#h g
def h_index(li):
    li_sorted=sorted(li, reverse=True)
    for i in li_sorted:
        if i < li_sorted.index(i)+1:
            break
    return li_sorted.index(i)

def g_index(li):
    li_sorted=sorted(li, reverse=True)
    for i in li_sorted:
        if sum(li_sorted[:li_sorted.index(i)+1])<(li_sorted.index(i)+1)**2:
            break
    return li_sorted.index(i)+1
data_author_index = data[['AF', 'Z9']].dropna().copy()
data_author_index['Authors'] = data_author_index['AF'].str.split(';')
data_author_index_sep = data_author_index.explode('Authors').reset_index().drop(columns=['index','AF'])
data_author_index_sep=data_author_index_sep[['Authors','Z9' ]]
data_h=data_author_index_sep.groupby('Authors').agg(lambda x: list(x)).reset_index()
data_h['h_index']= data_h['Z9'].apply(h_index)
h_index_top10 = data_h.sort_values('h_index',ascending=False).drop(columns='Z9').reset_index(drop=True).head(20)
data_h['g_index']= data_h['Z9'].apply(g_index)
g_index_top10 = data_h.sort_values(['h_index','g_index'],ascending=False).drop(columns='Z9').reset_index(drop=True).head(20)
st.dataframe(g_index_top10.head(10))
st.write('')

st.subheader('Wordcloud visualization')
st.write('Whole year:')
st.image('./header.png')

st.write('Jan-Apr')
st.image('./header.png')
st.write('May-Aug')
st.image('./header.png')
st.write('Sep-Dec')
st.image('./header.png')




st.header('Network analysis')

st.subheader('Co-authorship network analysis')
st.write('A co-authorship network is a complex network where each node represents an author, two nodes are connected by a link if there is one collaboration between them.')

st.subheader('Institution-level collaboration network analysis')
st.write('A institution-level collaboration network is a network where each node represents an institution. There is a link between two nodes if there is one publication collaborated by two institutions.')

st.subheader('Country-level collaboration network analyis')
st.write('A country-level collaboration network is a complex network where each node represents a country, two nodes are connected by a link if there is a collaboration between two institutions that belong to those two countries.')

st.subheader('Research area co-occurrence network analysis')
st.write('A research area co-occurrence network is a network where each node represents a research area, and two nodes are connected by a link if two research areas appear together in one paper. ')



st.subheader('Scale-free & small-world property of co-authorship network')
st.write('A scale-free network is a network whose degree distribution follows a power law, at least asymptotically.')
st.write('A small-world network is defined to be a network where the typical distance L (shortest length path) between two randomly chosen nodes (the number of steps required) grows proportionally to the logarithm of the number of nodes N in the network.')
st.write('To verify the scale-free property of the co-authorship network, powerlaw library is imported to measure the distribution of degree sequence and the frequency of appearing. And it asymptotically follows power law distribution and it is indeed a scale-free network.')
st.write('To further verify the small-world property, it\'s enough to consider the largest component of the network. Two measures should be calculated: average shortest path length and natural logrithm of the number of nodes. And they are: , . Since they are of the same order of magnitude, it suggests that the network has small-world property.')









