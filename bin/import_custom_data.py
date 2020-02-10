#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
import pandas
import numpy as np
import speech_recognition as sr

# Make sure we can import stuff from util/
# This script needs to be run from the root of the DeepSpeech repository
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from util.downloader import download

import pandas
import requests
from bs4 import BeautifulSoup

def load_existed(data_dir):
    def get_csv(csv_file):
        if os.path.exists(csv_file):
            return pandas.read_csv(csv_file)
        return None
    train = get_csv(os.path.join(data_dir, "custom_data_train.csv"))
    dev = get_csv(os.path.join(data_dir, "custom_data_dev.csv"))
    test = get_csv(os.path.join(data_dir, "custom_data_test.csv"))
    return train, dev, test

def get_url_paths(url, ext='', params={}):
    response = requests.get(url, params=params)
    if response.ok:
        response_text = response.text
    else:
        return response.raise_for_status()
    soup = BeautifulSoup(response_text, 'html.parser')
    parent = [node.get('href') for node in soup.find_all('a') if node.get('href').endswith(ext)]
    return parent

def preprocess(transcript):
    transcript = transcript.lower()
    digit2string = [
        ('701', 'seven hundred one'),
        ('702', 'seven hundred two'),
        ('703', 'seven hundred three'),
        ('704', 'seven hundred four'),
        ('705', 'seven hundred five'),
        ('706', 'seven hundred six'),
        ('707', 'seven hundred seven'),
        ('708', 'seven hundred eight'),
        ('709', 'seven hundred nine'),
        ('801', 'eight zero one'),
        ('802', 'eight zero two'),
        ('803', 'eight zero three'),
        ('*'  , 'star'),
        ('1'  , 'one'),
        ('2'  , 'two'),
        ('3'  , 'three'),
        ('4'  , 'four'),
        ('5'  , 'five'),
        ('6'  , 'six'),
        ('7'  , 'seven'),
        ('8'  , 'eight'),
        ('9'  , 'nine'),
        ('.'  , '')
    ]
    for digit, string in digit2string:
        transcript = transcript.replace(digit, string)
    return transcript

def transcribe(AUDIO_FILE):
    # AUDIO_FILE = path.join(path.dirname(path.realpath(__file__)), "english.wav")

    # use the audio file as the audio source
    r = sr.Recognizer()
    with sr.AudioFile(AUDIO_FILE) as source:
        audio = r.record(source)  # read the entire audio file
    try:
        rv = r.recognize_google(audio)
        print("Google Speech thinks you said " + rv)
        return preprocess(rv)
    except Exception as e:
        print("Google Speech not understand audio")
        # sphinx_transcribe(AUDIO_FILE)
        return ''

def _download_and_preprocess_data(url, file, data_dir):
    # Conditionally download data
    if not file.endswith('.wav'):
        return
    local_file, is_new_file = download(file, data_dir, url + file)
    # trans_file = maybe_download(LDC93S1_BASE + ".txt", data_dir, LDC93S1_BASE_URL + LDC93S1_BASE + ".txt")
    # with open(trans_file, "r") as fin:
    #     transcript = ' '.join(fin.read().strip().lower().split(' ')[2:]).replace('.', '')
    transcript = ''
    if is_new_file:
        transcript = transcribe(local_file)
    if transcript == '':
        return None

    return (os.path.abspath(local_file).replace('raw_files', 'audio_files'), os.path.getsize(local_file), transcript)


if __name__ == "__main__":
    reload_data = sys.argv[1] if len(sys.argv) > 1 else True
    url = sys.argv[2] if len(sys.argv) > 2 else None
    url = 'http://xmn02-i01-hpc04.lab.nordigy.ru:10001/' if url == None else url
    data_dir = './data/custom_data'
    audio_dir = os.path.join(data_dir, 'raw_files')
    files = get_url_paths(url)
    features = []
    i = 0
    for file in files:
        i += 1
        feature = _download_and_preprocess_data(url, file, audio_dir)
        if feature:
            features.append(feature)
    df = pandas.DataFrame(data=features,
                          columns=["wav_filename", "wav_filesize", "transcript"])

    msk = np.random.rand(len(df)) < 0.95
    train_and_dev = df[msk]
    test = df[~msk]

    msk = np.random.rand(len(train_and_dev)) < 0.95
    train = train_and_dev[msk]
    dev = train_and_dev[~msk]

    if reload_data:
        old_train, old_dev, old_test = load_existed(data_dir)
        train = train.append(old_train) if not old_train.empty else train
        dev = dev.append(old_dev) if not old_dev.empty else dev
        test = test.append(old_test) if not old_test.empty else test
    train.to_csv(os.path.join(data_dir, "custom_data_train.csv"), index=False)
    dev.to_csv(os.path.join(data_dir, "custom_data_dev.csv"), index=False)
    test.to_csv(os.path.join(data_dir, "custom_data_test.csv"), index=False)
