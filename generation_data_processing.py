# coding: utf-8
#data processing

import helper
import numpy as np
from collections import Counter


data_dir = './data/jd.txt'
text = helper.load_data(data_dir)

# data view 
view_sentence_range = (0, 10)
print('Dataset Stats')
print('Roughly the number of unique words: {}'.format(len({word: None for word in text.split()})))
scenes = text.split('\n\n')
print('Number of scenes: {}'.format(len(scenes)))
sentence_count_scene = [scene.count('\n') for scene in scenes]
print('Average number of sentences in each scene: {}'.format(np.average(sentence_count_scene)))

sentences = [sentence for scene in scenes for sentence in scene.split('\n')]
print('Number of lines: {}'.format(len(sentences)))
word_count_sentence = [len(sentence.split()) for sentence in sentences]
print('Average number of words in each line: {}'.format(np.average(word_count_sentence)))

print()
print('The sentences {} to {}:'.format(*view_sentence_range))
print('\n'.join(text.split('\n')[view_sentence_range[0]:view_sentence_range[1]]))


def create_lookup_tables(text):
    """
    准备word2id, id2word键值对
    """
    
    int_to_vocab={}
    vocab_to_int = {}
    _counter = Counter(text)
    _sorted_counter = sorted(_counter,key = _counter.get,reverse=True)
    
    for word,i in enumerate(_sorted_counter):
        int_to_vocab[i],vocab_to_int[word] = word,i
              
    return int_to_vocab, vocab_to_int



def token_lookup():
    """
   准备标点符号的键值对
    """
    
    return {
            '。':'||句号||',
            '，':'||逗号||',
            '：':'||冒号||',
            '！':'||感叹号||',
            '？':'||问号||',
            '(':'||左括号||',
            ')':'||右括号||',
            '--':'||引号||',        
            '\n':'||回车||',
            '；':'||分号||',
           }

		   
if name = '__main__':
	# data view 
	view_sentence_range = (0, 10)
	print('Dataset Stats')
	print('重复词数量 : {}'.format(len({word: None for word in text.split()})))
	scenes = text.split('\n\n')
	print('JD数量: {}'.format(len(scenes)))
	sentence_count_scene = [scene.count('\n') for scene in scenes]
	print('平均每个JD包含{}个句子'.format(np.average(sentence_count_scene)))

	sentences = [sentence for scene in scenes for sentence in scene.split('\n')]
	print('总句子行数: {}'.format(len(sentences)))
	word_count_sentence = [len(sentence.split()) for sentence in sentences]
	print('平均每行句子包含{}个词语'.format(np.average(word_count_sentence)))

	print('举个栗子')
	print('行 {} to {}:'.format(*view_sentence_range))
	print('\n'.join(text.split('\n')[view_sentence_range[0]:view_sentence_range[1]]))




	#保存为preprocess.p
	helper.preprocess_and_save_data(data_dir, token_lookup, create_lookup_tables)






