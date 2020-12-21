import sys
from PIL import Image
import cv2 as cv
import numpy as np
import pandas as pd
import pytesseract
import nltk
import re
from nltk.corpus import stopwords
import logging
import time
from datetime import datetime
from openpyxl.workbook import Workbook


def main(image_path):
    print("starting text extraction from image")
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
    img = cv.imread(image_path)

    # img = clean_image(img)
    custom_config = r'-l eng --psm 3'
    text = pytesseract.image_to_string(img, lang="eng",config=custom_config )
    text = text.replace(",", "")
    text = text.split("\n")
    cleaned_text = []
    cleaned_text = [item for item in text if item not in [" ", '']]

    return cleaned_text


def clean_image(img):
    img = cv.imread(img, cv.IMREAD_GRAYSCALE)
    img = cv.multiply(img, 1.2)
    kernel = np.ones((1, 1), np.uint8)
    img = cv.erode(img, kernel, iterations=1)
    cv.imwrite("cleaned_img.png", img)

    return img

# Preprocessing  and preparing dataset with tagging each word


def preprocess_input_data(x,sentence_id):
    new_x = []
    x = [item.lower() for item in x if item not in stopwords.words('english')]
    # x=[item.replace(".","") for item in x]
    x = [item.strip() for item in x]
    x = [re.sub("[^-\n$,a-zA-Z.0-9]+", " ", text) for text in x]
    for item in x:
        item = item.split(" ")
        for item1 in item :
            if item1 not in [' ','.','','-']:
                new_x.append(item1)
    df = pd.DataFrame()
    df['Data'] = new_x
    df['Sentence']=sentence_id


    return x,df




if __name__ == '__main__':
    begin_time = time.time()
    print("Starting with Main *****")
    final_df = pd.DataFrame(columns=['Data','Sentence'])
    import os

    sentence_id=0
    directory = r'C:\Users\Lovely\Downloads\Invoice_testing'
    for filename in os.listdir(directory):
        sentence_id+=1
        if filename.endswith(".tif"):
            image_path=os.path.join(directory, filename)
            cleaned_text = main(image_path)
            z,df= preprocess_input_data(cleaned_text,sentence_id)
            print("Text after extraction :--- \n")
            print(z)
            final_df=final_df.append(df,ignore_index=True)


    # writting final dataframe to Excel

    final_df.to_excel("required2_dataset.xlsx", index=False)
    end_time = time.time()

    # creating log file

    logging.basicConfig(filename='Building_Dataset.log', level=logging.DEBUG)
    logging.info("********** Starting LOG *************")
    current_time = str(datetime.now().strftime("%H:%M:%S"))
    logging.warning('At ' + current_time + '  this event was logged.')
    logging.info('Text extraction and cleaning from doc finished  in  ' + str(np.round((begin_time - end_time) / 60.0, 3)) + '  mins')
    logging.info('********* Done ***********\n')