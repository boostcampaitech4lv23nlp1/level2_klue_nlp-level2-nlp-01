from tqdm import tqdm
import time
import random
import selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import pandas as pd
import re

def chrome_setting():
  chrome_options = webdriver.ChromeOptions()
  chrome_options.add_argument('--headless')
  chrome_options.add_argument('--no-sandbox')
  chrome_options.add_argument('--disable-dev-shm-usage')
  driver = webdriver.Chrome('chromedriver', options=chrome_options)
  return driver


# Crawling
def back_trans(text_data, lang):
    try:
        target_present = EC.presence_of_element_located((By.XPATH, '//*[@id="txtTarget"]'))
        driver.get('https://papago.naver.com/?sk=ko&tk='+lang+'&st='+text_data)
        time.sleep(random.uniform(0.2, 1))
        element=WebDriverWait(driver, 10).until(target_present)
        time.sleep(random.uniform(0.2, 1))
        backtrans = element.text
    except:
        print(text_data)
        return ""
    
    try:
        target_present = EC.presence_of_element_located((By.XPATH, '//*[@id="txtTarget"]'))
        driver.get('https://papago.naver.com/?sk='+lang+'&tk=ko&st='+backtrans)
        time.sleep(random.uniform(0.2, 1))
        element=WebDriverWait(driver, 10).until(target_present)
        time.sleep(random.uniform(0.2, 1))
        new_backtrans = element.text
        return new_backtrans
    except:
        print(text_data)
        return ""

def find_foreign(start: int, end: int, sentence: str):
    count=0
    a = int(start, 16) 
    b = int(end, 16)
    return_sentence = ''
    for i, w in enumerate(sentence):
        if a <= ord(w) and ord(w) <= b:
            return True
    return False

if __name__ == '__main__':

    driver=chrome_setting()
    print('finish driver setting')

    raw_df = pd.read_csv('back_translation_input.csv')  

    df=raw_df.loc[24001:32423] #현정
    print('finish loading data')

    trans_list=[]

    for i, sen, sub, obj, label, source in tqdm(zip(df['id'], df['sentence'], df['subject_entity'], df['object_entity'], df['label'], df['source'])):
        #no_relation이 아니면서 / 외국어가 전혀 없으면서 / 각 엔티티가 문장에 한 번만 존재하는 경우에
        if( label!="no_relation"):
            if(find_foreign('4e00', '9fff', sen)==False) and (find_foreign('0400', '04ff', sen)==False) and (find_foreign('0600', '06ff', sen)==False) and (find_foreign('0370', '03ff', sen)==False) and (find_foreign('3040', '309f', sen)==False) and (find_foreign('30a0', '30ff', sen)==False):
                sub = eval(sub)
                obj = eval(obj)
        
                sub_word=sub['word']
                obj_word=obj['word']
        
                sub_word_cnt=sen.count(sub_word)
                obj_word_cnt=sen.count(obj_word)
                
                if(sub_word_cnt==1 and obj_word_cnt==1): #역번역하자! #17658개에 대해서
                    new_sen=back_trans(sen,"zh-CN")
                    # new_sen=back_translation(sen,"en")
                    if sub_word in new_sen and obj_word in new_sen: #번역시에도 문장의 엔티티가 잘 살아있는 경우에만 추가
                        ss=new_sen.find(sub_word)
                        se=ss+len(sub_word)-1
                        os=new_sen.find(obj_word)
                        oe=os+len(obj_word)-1
                        
                        sub['start_idx']=ss
                        sub['end_idx']=se
                        obj['start_idx']=os
                        obj['end_idx']=oe
                        
                        sub=str(sub)
                        obj=str(obj)
                        
                        temp=[]
                        temp.append(i)
                        temp.append(new_sen)
                        temp.append(sub)
                        temp.append(obj)
                        temp.append(label)
                        temp.append(source)
                        trans_list.append(temp)

    new_df = pd.DataFrame(trans_list)

    new_df.to_csv('/opt/ml/project2/level2_klue_nlp-level2-nlp-01/dataset/back_translation/back_translation_output_cha_5.csv',index=False)
    print('finish back translation')