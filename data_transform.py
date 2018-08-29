
# coding: utf-8

# In[13]:


import pandas as pd
import re
from collections import defaultdict
import jieba


#     read material and anlysis

# In[14]:


text_path = './data/workexp.csv'
text = pd.read_csv(text_path,encoding='gb18030')
print(text.shape)


# In[15]:


t ='{}\n{}'.format(text.at[1,'title'],text.at[1,'workexp'])
s = t.split('\n')
print(s)


# In[18]:


dataset = defaultdict(lambda:0)
index_list = text.index.tolist()
txtpath = './data/jd.txt'
zhPattern = re.compile(u'[\u4e00-\u9fa5]+')

with open(txtpath,'w',encoding='gb18030') as f: 
    for i in index_list:
        sentence = '{}'.format(text.at[i,'workexp'])
        match = zhPattern.search(sentence)
        if not match:
            continue
        
        title = '{}\n'.format(text.at[i,'title'])
        f.write(title)
        
        sentence.replace("</br>","\n")
        sentence_list = sentence.split('\n')
        for sentence in sentence_list:
            result = jieba.cut(sentence)
            result_blank = ' '.join(result)
            f.write(result_blank)
            f.write("\n")
            
        f.write("\n\n")
        
    


# In[ ]:




