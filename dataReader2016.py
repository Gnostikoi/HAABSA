import os
import json
import xml.etree.ElementTree as ET
from collections import Counter
import string
import en_core_web_sm
en_nlp = en_core_web_sm.load()
import nltk
import re
import json
import numpy as np
nltk.download('averaged_perceptron_tagger')
pattern = 'NP: {<DT>?<JJ>*<NN>}'
parser = nltk.RegexpParser(pattern)
GENERAL_ASPECT_TAG = 'general_aspect'

def general_asp_insert_pos(sptoks, NPs):
    if len(NPs) == 0:
        return len(sptoks)//2
    mid_np = NPs[len(NPs)//2]
    mid_np_tok = nltk.word_tokenize(mid_np)
    len_mid_np = len(mid_np_tok)
    pos = None
    i = 0
    while not pos and i < len(sptoks):
        if sptoks[i] == mid_np_tok[0]:
            pos = i
            for j in range(len_mid_np):
                # print(i,j, mid_np_tok, sptoks)
                if sptoks[i+j] != mid_np_tok[j]:
                    pos = None
                    break
        i += 1
    assert pos is not None, print(sptoks, NPs, pos)
    assert pos >= 0 and pos <= len(sptoks), print(sptoks, NPs, pos)
    return pos

def chunk(sent):
    sent = nltk.word_tokenize(sent)
    sent = nltk.pos_tag(sent)
    tree = parser.parse(sent)

    NPs = []

    for sub in tree.subtrees():
        if sub.label() == 'NP':
            NPs.append(' '.join([tup[0] for tup in sub]))

    return NPs

def window(iterable, size): # stack overflow solution for sliding window
    i = iter(iterable)
    win = []
    for e in range(0, size):
        win.append(next(i))
    yield win
    for e in i:
        win = win[1:] + [e]
        yield win

def _get_data_tuple(sptoks, asp_termIn, label):
    # Find the ids of aspect term
    aspect_is = []
    asp_term = ' '.join(sp for sp in asp_termIn).lower()
    for _i, group in enumerate(window(sptoks,len(asp_termIn))):
        if asp_term == ' '.join([g.lower() for g in group]):
            aspect_is = list(range(_i,_i+len(asp_termIn)))
            break
        elif asp_term in ' '.join([g.lower() for g in group]):
            aspect_is = list(range(_i,_i+len(asp_termIn)))
            break


    # print(aspect_is)
    assert len(aspect_is) != 0, print(sptoks, asp_termIn)
    pos_info = []
    for _i, sptok in enumerate(sptoks):
        pos_info.append(min([abs(_i - i) for i in aspect_is]))

    # lab = None
    # if label == 'negative':
    #     lab = -1
    # elif label == 'neutral':
    #     lab = 0
    # elif label == "positive":
    #     lab = 1
    # else:
    #     raise ValueError("Unknown label: %s" % lab)
    label_map = {
        'RESTAURANT#GENERAL': 0,
        'SERVICE#GENERAL': 1, 
        'FOOD#QUALITY': 2, 
        'FOOD#STYLE_OPTIONS': 3, 
        'DRINKS#STYLE_OPTIONS': 4, 
        'DRINKS#PRICES': 5, 
        'RESTAURANT#PRICES': 6, 
        'RESTAURANT#MISCELLANEOUS': 7, 
        'AMBIENCE#GENERAL': 8, 
        'FOOD#PRICES': 9, 
        'LOCATION#GENERAL': 10, 
        'DRINKS#QUALITY': 11,
        'NONE': 13
    } 
    lab = label_map[label]

    return pos_info, lab


"""
This function reads data from the xml file

Iput arguments:
@fname: file location
@source_count: list that contains list [<pad>, 0] at the first position [empty input]
and all the unique words with number of occurences as tuples [empty input]
@source_word2idx: dictionary with unique words and unique index [empty input]
.. same for target

Return:
@source_data: list with lists which contain the sentences corresponding to the aspects saved by word indices 
@target_data: list which contains the indices of the target phrases: THIS DOES NOT CORRESPOND TO THE INDICES OF source_data 
@source_loc_data: list with lists which contains the distance from the aspect for every word in the sentence corresponding to the aspect
@target_label: contains the polarity of the aspect (0=negative, 1=neutral, 2=positive)
@max_sen_len: maximum sentence length
@max_target_len: maximum target length

"""
def read_data_2016(fname, source_count, source_word2idx, target_count, target_phrase2idx, file_name):
    if os.path.isfile(fname) == False:
        raise ("[!] Data %s not found" % fname)

    # parse xml file to tree
    tree = ET.parse(fname)
    root = tree.getroot()

    outF= open(file_name, "w")

    # save all words in source_words (includes duplicates)
    # save all aspects in target_words (includes duplicates)
    # finds max sentence length and max targets length
    source_words, target_words, max_sent_len, max_target_len = [], [], 0, 0
    target_phrases = []

    countConfl = 0
    for sentence in root.iter('sentence'):
        sent = sentence.find('text').text
        sentenceNew = re.sub(' +', ' ', sent)
        sptoks = nltk.word_tokenize(sentenceNew)
        NPs = chunk(sentenceNew)
        for sp in sptoks:
            source_words.extend([''.join(sp).lower()])
        if len(sptoks) > max_sent_len:
            max_sent_len = len(sptoks)

        aspects = []
        for opinions in sentence.iter('Opinions'):
            for opinion in opinions.findall('Opinion'):
                # if opinion.get("polarity") == "conflict":
                #     countConfl += 1
                #     continue
                asp = opinion.get('target')
                if asp == 'NULL':
                    asp = GENERAL_ASPECT_TAG      
                for np in NPs[:]:
                    if asp in np or np in asp and len(np) > 3:
                        NPs.remove(np)

                aspNew = re.sub(' +', ' ', asp)
                aspects.append(aspNew)
        aspects.extend(NPs)
        # print(aspects)
        for aspNew in aspects:
            t_sptoks = nltk.word_tokenize(aspNew)
            for sp in t_sptoks:
                target_words.extend([''.join(sp).lower()])
            target_phrases.append(' '.join(sp for sp in t_sptoks).lower())
            if len(t_sptoks) > max_target_len:
                max_target_len = len(t_sptoks)

    source_words.append(GENERAL_ASPECT_TAG)
    if len(source_count) == 0:
        source_count.append(['<pad>', 0])
    source_count.extend(Counter(source_words + target_words).most_common())
    target_count.extend(Counter(target_phrases).most_common())

    for word, _ in source_count:
        if word not in source_word2idx:
            source_word2idx[word] = len(source_word2idx)

    for phrase, _ in target_count:
        if phrase not in target_phrase2idx:
            target_phrase2idx[phrase] = len(target_phrase2idx)

    source_data, source_loc_data, target_data, target_label = list(), list(), list(), list()

    # collect output data (match with source_word2idx) and write to .txt file
    asp_cnt = 0
    sentence_n_aspects = dict()
    for sentence in root.iter('sentence'):
        sentence_id = sentence.get("id")
        sentence_n_aspects[sentence_id] = []
        sent = sentence.find('text').text
        sentenceNew = re.sub(' +', ' ', sent)
        sptoks = nltk.word_tokenize(sentenceNew)
        if len(sptoks) != 0:
            idx = []
            for sptok in sptoks:
                idx.append(source_word2idx[''.join(sptok).lower()])
            NPs = chunk(sentenceNew)
            aspects = []
            labels = []
            for opinions in sentence.iter('Opinions'):
                for opinion in opinions.findall('Opinion'):
                    # if opinion.get("polarity") == "conflict": continue
                    asp = opinion.get('target')
                    if asp != 'NULL': #removes implicit targets
                        for np in NPs[:]:
                            if asp in np or np in asp and len(np) > 3:
                                NPs.remove(np)
                    else:
                        asp = GENERAL_ASPECT_TAG
                    aspects.append(asp)
                    labels.append(opinion.get('category'))
            for np in NPs:
                aspects.append(np)
                labels.append('NONE')
            for asp, category in zip(aspects, labels):
                sentence_n_aspects[sentence_id].append(asp_cnt)
                aspNew = re.sub(' +', ' ', asp)
                t_sptoks = nltk.word_tokenize(aspNew)
                if asp == GENERAL_ASPECT_TAG:
                    insert_point = general_asp_insert_pos(sptoks, NPs)
                    sptoks.insert(insert_point, asp)
                source_data.append(idx)
                outputtext = ' '.join(sp for sp in sptoks).lower()
                outputtarget = ' '.join(sp for sp in t_sptoks).lower()
                outputtext = outputtext.replace(outputtarget, '$T$')
                outF.write(outputtext)
                outF.write("\n")
                outF.write(outputtarget)
                outF.write("\n")
                pos_info, lab = _get_data_tuple(sptoks, t_sptoks, category)
                # pos_info, lab = _get_data_tuple(sptoks, t_sptoks, opinion.get('polarity'))
                pos_info = [(1-(i / len(idx))) for i in pos_info]
                source_loc_data.append(pos_info)
                targetdata = ' '.join(sp for sp in t_sptoks).lower()
                # print(targetdata, t_sptoks, sptoks)
                target_data.append(target_phrase2idx[targetdata])
                target_label.append(lab)
                outF.write(str(lab))
                outF.write("\n")
                asp_cnt += 1

    # store sentence_n_aspects
    # outF.write(' '.join(sentence_n_aspects))
    outF.write(json.dumps(sentence_n_aspects))
    outF.close()
    print("Read %s aspects from %s" % (len(source_data), fname))
    # assert len(source_data) == len(sentence_n_aspects)
    # print(countConfl)
    return len(source_data), source_loc_data, target_data, target_label, max_sent_len, max_target_len, sentence_n_aspects


