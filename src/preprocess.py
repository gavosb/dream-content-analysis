
from time import time
import os.path
import re
import contractions
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

def clean_text(raw_html):
    CLEANR = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    clean = re.sub(CLEANR, '', raw_html)
    clean = contractions.fix(clean)
    return clean

def create_data(journal):
    data = np.array([])

    dream_xml = os.path.join(os.path.dirname(__file__), '../../data/dreambank-public/dreambank-public.xml')

    tree = ET.parse(dream_xml, parser=ET.XMLParser(encoding='iso-8859-5'))
    root = tree.getroot()

    for report in root[journal].iter('report'):
        data = np.append(data, clean_text(report.text))
    return data
