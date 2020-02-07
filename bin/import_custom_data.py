#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
import pandas
import speech_recognition as sr

# Make sure we can import stuff from util/
# This script needs to be run from the root of the DeepSpeech repository
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from util.downloader import maybe_download

import requests
from bs4 import BeautifulSoup

def get_url_paths(url, ext='', params={}):
    response = requests.get(url, params=params)
    if response.ok:
        response_text = response.text
    else:
        return response.raise_for_status()
    soup = BeautifulSoup(response_text, 'html.parser')
    parent = [node.get('href') for node in soup.find_all('a') if node.get('href').endswith(ext)]
    return parent

def transcribe(AUDIO_FILE):
    # AUDIO_FILE = path.join(path.dirname(path.realpath(__file__)), "english.wav")

    # use the audio file as the audio source
    r = sr.Recognizer()
    with sr.AudioFile(AUDIO_FILE) as source:
        audio = r.record(source)  # read the entire audio file
    try:
        rv = r.recognize_google(audio)
        print("Google Speech thinks you said " + rv)
        return rv
    except Exception as e:
        print("Google Speech not understand audio")
        # sphinx_transcribe(AUDIO_FILE)
        return ''

def _download_and_preprocess_data(url, file, data_dir):
    # Conditionally download data
    if not file.endswith('.wav'):
        return
    local_file = maybe_download(file, data_dir, url + file)
    # trans_file = maybe_download(LDC93S1_BASE + ".txt", data_dir, LDC93S1_BASE_URL + LDC93S1_BASE + ".txt")
    # with open(trans_file, "r") as fin:
    #     transcript = ' '.join(fin.read().strip().lower().split(' ')[2:]).replace('.', '')
    transcript = transcribe(local_file)
    if transcript == '':
        return None

    return (os.path.abspath(local_file), os.path.getsize(local_file), transcript)


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else None
    url = 'http://xmn02-i01-hpc04.lab.nordigy.ru:10001/' if url == None else url
    data_dir = './data/custom_data'
    files = get_url_paths(url)
    features = []
    i = 0
    for file in files:
        i += 1
        feature = _download_and_preprocess_data(url, file, data_dir)
        if feature:
            features.append(feature)
        if i > 5:
            break
    df = pandas.DataFrame(data=features,
                          columns=["wav_filename", "wav_filesize", "transcript"])
    df.to_csv(os.path.join(data_dir, "custom_data.csv"), index=False)
