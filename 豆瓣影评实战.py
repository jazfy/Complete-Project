
# coding: utf-8

# # 豆瓣影评分析实战
        技术层面实现中文文本分析,不仅可深刻掌握人力无法企及的本质内容,而且在呈现方式上常与图表等方式结合,我们始终相信,技术是具有普世价值的,不应局限于被某个特定群体垄断.
        今天,从Python开始,以豆瓣影评数据为对象,进行真正意义上的文本挖掘,go🐶🐶🐶!
# ## 分词
        中文文本在处理上,和英文最大的不同之处就在于需要将文章切分为一组组具有特定意义的词汇和短语,分词并非只是简单把文章切开即可,实际上还存在过滤停用词等方面,而采用何种分词算法是我们不必关注的,因为jieba足够好用!
# ### 加载数据和导入依赖库

# In[1]:


import jieba #结巴分词
import pandas as pd #pandas库,处理数据岂能不用,简直完美取代Excel

basic_information = pd.read_csv('/Users/meininghang/Downloads/douban_movie/basic_information.csv') #读取基本信息文件
basic_information.head(10) #显示前十项


# In[2]:


movie_basic_information = pd.read_csv('/Users/meininghang/Downloads/douban_movie/movie_basic_information.csv') #另一份基本信息文件
movie_basic_information.head(10) #显示前十项


# In[3]:


comment_information = pd.read_csv('/Users/meininghang/Downloads/douban_movie/comments.csv') #加载影评信息
comment_information.head(10) #显示前十项


# ### 清洗数据
        注意到邮件上认为抓取的是变3一部电影的评论,其实并不是一部,而是很多部,但在实践中,我们在分析时,也是要分门别类进行分析,数据分析可以减轻一些人工检索和探索特征的工作,但其也有局限性,筛选有意义的数据,祛除无意义的数据占用了大量时间,因此,为贴合实际,我们筛选出关于变3的各项信息.
# In[4]:


basic_information.影片名.unique() #查看影片数量


# In[5]:


basic_information.影片名.value_counts() #查看各影片具体数量


# In[6]:


T_3 = basic_information.iloc[:40] #变3基本信息共有40行,往上面👆,最后一行,所以截取全部信息,注意0是第一个起始数字
#由于此时数据量很少,人工查看相关信息即可,我们实际上需要的信息只有演职人员/姓名两列,单独拿出来即可,其余信息随用随取


# In[7]:


#重构信息
T_3_information = T_3[['姓名','演职人员']] #选取信息
T_3_information = T_3_information.drop_duplicates(['姓名']) #去重
T_3_information.index = [1,2,3,4,5,6,7,8,9,10,11,12] #重命名列
T_3_information #显示信息,此数据用于分析各演员在评论中的表现


# In[8]:


movie_basic_information.电影名称.unique() #对另一份信息数据进行查验,并找到变3的信息


# In[9]:


T_3_Star = movie_basic_information[movie_basic_information.电影名称 == '变形金刚3 Transformers: Dark of the Moon'] #设定筛选标准
T_3_Star = T_3_Star.T#评分信息比较特殊,每部电影只有一个信息

T_3_Star


# In[10]:


#汇总基本信息
T_3_INF = pd.read_csv('/Users/meininghang/Downloads/douban_movie/T_3_information.csv')
T_3_INF


# In[11]:


T_3_comment = comment_information[comment_information.电影名称 == '变形金刚3的影评 (1590)'] #筛选影评信息
T_3_comment.head() #查看前五项


# In[12]:


T_3_comment.info() #查看基本信息


# #### 开始分词

# #####  过滤停用词
        停用词是指缺乏实在意义的词和标点符号等内容,对这些信息进行筛选,可以有效减少'噪音'
# In[13]:


#加载停用词典,在此选用哈工大停用词典
with open('/Users/meininghang/Downloads/stopwords-master/哈工大停用词表.txt') as f:
    stop_words = f.read()


# In[14]:


stop_words.splitlines() #祛除换行符


# In[15]:


all_title_seg = [] #所有题目分词
for i in T_3_comment.题目:
    single_title_seg = [] #存放单个题目分词结果
    segs = jieba.cut(i) #对每个题目进行切分
    for seg in segs:
        if seg not in stop_words: #判断是否在停用词典内,
            single_title_seg.append(seg) #单个评论题目加入[]
    all_title_seg.append(single_title_seg) #全体加入
    


# In[16]:


print('/'.join (all_title_seg[9])) #查验


# In[17]:


#主角,所有评论,皆与上类似
all_comment_seg = [] 
for i in T_3_comment.评论内容:
    single_comment_seg = []
    segs = jieba.cut(i)
    for seg in segs:
        if seg not in stop_words:
            single_comment_seg.append(seg)
    all_comment_seg.append(single_comment_seg)


# In[18]:


print('/'.join(all_comment_seg[388]))


# ## 情感分析
        情感分析实际上是二元分类的推演,通过概率来判断给定样本的概率大小,随后设定阈值,超过此阈值则给出判断,在本案例中,情感分析实际上有某种预示,五星好评和一星差评的情感倾向一定是不同的,这也是进行情感判断的出发点.
# In[21]:


from snownlp import SnowNLP #加载snownlp


# In[22]:


#test
s = SnowNLP(T_3_comment.题目[1])


# In[23]:


s.words #词


# In[26]:


s.tags #词性标注


# In[27]:


for i in s.tags:
    print(i)


# In[31]:


s.sentiments #情绪


# In[32]:


s.keywords(3) #关键词


# In[37]:


s.summary(3)  #摘要,似乎没有效果,可能因为太短


# In[39]:


s.sentences #切句子


# In[42]:


s.tf #tf


# In[43]:


s.idf #idf


# In[46]:


s.sim(['作']) #相似度


# In[47]:


s = SnowNLP([[T_3_comment.题目[1],T_3_comment.题目[0],T_3_comment.题目[7]]])


# In[66]:


from snownlp import SnowNLP
s = SnowNLP(T_3_comment.评论内容[1])
print(s.summary(3),) #用段落比较好

对snownlp的认识结束,开始实践
# ### 情感分析

# In[74]:


from snownlp.sentiment import Sentiment
import matplotlib.pyplot as plt #画图表示下
import numpy as np


# In[75]:


def s_nlp_sentiment(self):
    sentiment_fruit = []
    for i in self:
        s_t = SnowNLP(i)
        sentiment_fruit.append(s_t.sentiments)
    plt.hist(sentiment_fruit,bins=np.arange(0,1,0.01))
    plt.show()


# In[81]:


sent = []
for i in T_3_comment.题目:
   #print(i)
   s = SnowNLP(i)
   senti = s.sentiments
   sent.append(senti)
   #plt.hist(sent,bins=np.arange(0,1,0.01))
   #plt.show()


# In[89]:


len(sent)#检查


# In[91]:


T_3_comment['情绪'] = sent


# In[95]:


T_3_comment.to_csv('/Users/meininghang/Desktop/T_3_comment.csv') #导出


# In[98]:


#加载星级和情绪得分之间的关系

        从其中可得出两种结论:
        1.评分人整体上对电影持认同态度,情绪平均值高于0.5;
        2.星级和情绪持正相关,即评分越高,印象也较为正面.
# ## 主题和文本相似度

# In[1]:


import pandas as pd
import jieba
import gensim #构建依赖库
from gensim import corpora


# In[2]:


#加载数据
T_3_comment = pd.read_csv('/Users/meininghang/douban-movie-sets-anyalase/T_3_comment.csv')
T_3_comment['评论内容'][:5] #查看内容


# In[6]:


with open("/Users/meininghang/Downloads/stopwords-master/四川大学机器智能实验室停用词库.txt") as f: #停用词
    stop_words = f.read().splitlines()


# In[8]:


comment = []
for i in T_3_comment.评论内容:
    comment_seg = [] #加载单个评论分词结果
    segs = jieba.cut(i) #分词
    for seg in segs:
        if seg not in stop_words:# 过滤
            comment_seg.append(seg)
    comment.append(comment_seg)


# In[11]:


comment[0] #查验


# In[12]:


#TF-IDF
dictionary = corpora.Dictionary(comment)


# In[13]:


cipin = [dictionary.doc2bow(comm) for comm in comment]
cipin


# In[18]:


from gensim import models
from gensim.corpora import corpus
tf_idf_model = models.TfidfModel(cipin)
tfidf = tf_idf_model[cipin]
#tf_idf_mat = #corpus2dense(tfidf,len(dictionary))


# In[19]:


model = gensim.models.Word2Vec(comment,size=1000,window=5,min_count=5)


# In[20]:


model['评论']


# In[36]:


model.similarity('剧','电影')


# In[ ]:


#jeibatf-idf示例
'''基于 TF-IDF 算法的关键词抽取
import jieba.analyse

jieba.analyse.extract_tags(sentence, topK=20, withWeight=False, allowPOS=())
sentence 为待提取的文本
topK 为返回几个 TF/IDF 权重最大的关键词，默认值为 20
withWeight 为是否一并返回关键词权重值，默认值为 False
allowPOS 仅包括指定词性的词，默认值为空，即不筛选
jieba.analyse.TFIDF(idf_path=None) 新建 TFIDF 实例，idf_path 为 IDF 频率文件'''


# In[23]:


import jieba
import jieba.analyse


# In[88]:


k = jieba.analyse.extract_tags(T_3_comment.评论内容[1],topK=2,withWeight=True)


# In[89]:


k


# In[41]:


k[1][0] #type==str


# In[42]:


model.similarity(k[0][0],k[1][0])


# In[48]:


for i in T_3_comment.评论内容:
    k = jieba.analyse.extract_tags(i,topK=2,)


# In[52]:


k


# In[60]:


from gensim.similarities import MatrixSimilarity


# In[62]:


sim_index = MatrixSimilarity(comment)


# In[63]:


sim_index[comment[0]]


# In[64]:


sort_sims = sorted(enumerate(sim_index[comment[0]]), key=lambda item: item[1],reverse=True)  # 利用 sorted 函数按照相似度从大到小排序
print(sort_sims[0:10])  # 输出最为相似的前 10 个文档编号及相似度


# In[68]:


for j in [i[0] for i in sort_sims[0:10]]:
    print(j,"\n",T_3_comment.评论内容[j])   #输出原始文档内容，查看检索效果

        首先回溯第一篇影评的直观感受是"讽刺',其情绪得分较低,而此时返回的相近文本具有相同的感受,即"不满'
# In[70]:


T_3_comment.情绪[0]


# ## 可视化

# In[97]:


get_ipython().system('pip3 install pyldavis')


# In[98]:


import pyLDAvis
import pyLDAvis.gensim
from gensim import corpora,models
from gensim.models.ldamodel import LdaModel


# In[106]:


with open('/Users/meininghang/Downloads/stopwords-master/四川大学机器智能实验室停用词库.txt') as f:
    stop_words = f.read().splitlines()
    comment = []
    for i in T_3_comment.评论内容:
        segment = []
        segs = jieba.cut(i)
        for seg in segs:
            if seg not in stop_words:
                segment.append(seg)
        comment.append(segment)


# In[107]:


dictionary = corpora.Dictionary(comment)
corpus = [dictionary.doc2bow(commen) for commen in comment]


# In[108]:


tfidf_model = models.TfidfModel(corpus)
corpus_tfidf = tfidf_model[corpus]


# In[109]:


lda = LdaModel(corpus=corpus_tfidf,id2word=dictionary,num_topics=3)


# In[118]:


lda.show_topics(9)


# In[120]:


vis_data = pyLDAvis.gensim.prepare(lda,corpus,dictionary)
vis_data


# In[123]:


pyLDAvis.display(vis_data)


# In[140]:


k = T_3_comment.评论时间
k = str(k)
print(k,'\n')


# In[146]:


year = []
month = []
day = []

for i in T_3_comment.评论时间:
    year.append(i.split('-')[0])
    month.append(i.split('-')[1])
    day.append(i.split('-')[2])

T_3_comment['year'] = year
T_3_comment['month'] = month
T_3_comment['day'] = day

T_3_comment['count'] = len(T_3_comment) * [1]
T_3_comment.head()


# In[147]:


dataset_ym = pd.pivot_table(T_3_comment,values='count',
                            index = 'year',columns='month',aggfunc='sum')
dataset_ym


# In[148]:


get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sea
sea.heatmap(dataset_ym,linewidths=.5)

        表示不同年份各月份累计评论数热图
# In[149]:


sea.heatmap(dataset_ym,annot=True,fmt="n",linewidths=.5)


# In[151]:


title[2]


# In[155]:


T_3_comment['words'] = comment


# In[156]:


T_3_comment['word_count'] = [i.count('评论') for i in T_3_comment['words']]


# In[157]:


dataset_md_words=pd.pivot_table(T_3_comment, values = 'word_count', index = 'month', columns = 'day',aggfunc='sum')


# In[158]:


dataset_md_words


# In[159]:


sea.heatmap(dataset_md_words,linewidths=.5)


# In[160]:


from datetime import datetime


# In[181]:


T_3_comment.评论时间[1]


# In[183]:


time = datetime.strptime(T_3_comment['评论时间'][0], '%Y-%m-%d %H:%M:%S')


# In[185]:


print(time)


# In[192]:


T_3_comment['具体时间'] = [datetime.strptime(i,'%Y-%m-%d %H:%M:%S') for i in T_3_comment['评论时间']]


# In[188]:


T_3_comment.具体时间.head()


# In[193]:


T_3_comment.具体时间.value_counts()


# In[194]:


T_3_comment.评论时间.value_counts().plot(kind='line', rot=0, figsize=(14, 8))


# In[197]:


group_word = T_3_comment.groupby(['评论时间'])['word_count'].sum()


# In[198]:


group_word.plot(kind='line',figsize=(14, 8))

