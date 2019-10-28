import pdfplumber
import os
import collections
import json
import nltk

def ocr_pdf(infile, outfile):
    os.system('ocrmypdf -l eng --rotate-pages --deskew --title "a" --jobs 4 --output-type pdfa {} {}'.format(infile, outfile))

def get_lines(words, epsilon=2):
    lines = []
    previous_word = None
    buffer = []
    for word in words:
        if previous_word and word['bottom'] - previous_word['bottom'] > epsilon:
            lines.append(buffer)
            buffer = []
        previous_word = word
        buffer.append(word)
    if len(buffer) != 0: lines.append(buffer)
    return lines

def is_invoice_start(line, keywords=['date', 'description', 'price'], align=2):
    aligned_words = set()
    for word in line:
        text = word['text'].lower()
        if text in keywords: aligned_words.add(text)
    if len(aligned_words) >= align: return True
    return False

def is_invoice_end(line, keywords=['subtotal'], align=1):
    aligned_words = set()
    for word in line:
        text = word['text'].lower()
        if text in keywords: aligned_words.add(text)
    if len(aligned_words) >= align: return True
    return False

def extract_invoice_page(page):
    words = page.extract_words()
    words = sorted(words, key=lambda x: x['top'])
    lines = get_lines(words)
    invoice_lines = []
    is_invoice = False
    for line in lines:
        if is_invoice_start(line):
            is_invoice = True
        if is_invoice_end(line):
            is_invoice = False
        if is_invoice:
            invoice_lines.append(line)
    return invoice_lines

def extract_invoice(filename, ocr=False):
    if ocr: 
        ocr_pdf(filename, filename + '.out')
        filename = filename + '.out'
    pdf = pdfplumber.open(filename)
    ret = [extract_invoice_page(page) for page in pdf.pages]
    ret = [i for i in ret if len(i) != 0]
    return ret

def pretty_print_line(line):
    return ' '.join([word['text'] for word in line])
        
def pretty_print_lines(lines):
    return [pretty_print_line(line) for line in lines]

def prepare_data(infilenames, labels, outfilename):
    data = []
    for filename, label in zip(infilenames, labels):
        lines = pretty_print_lines(extract_invoice(filename, ocr=True)[0])
        text = '\n'.join(lines).replace('\t', ' ').replace('\n', ' _eol_ ')
        data.append([text, label])
    with open(outfilename, 'w') as fout:
        for item in data:
            text, label = item
            fout.write(text + '\t' + label + '\n')

def prepare_char_encoder(infilename, outfilename, lower=True):
    data = []
    with open(infilename) as fin:
        for line in fin:
            splitline = line.strip().split('\t')
            if lower: text = splitline[0].strip().lower()
            else: text = splitline[0].strip()
            data += list(text)
    counter = collections.Counter(data) # TODO: handle _eol_ in one char
    encoder = {}
    for char, freq in counter.most_common():
        encoder[char] = len(encoder)
    json.dump(encoder, open(outfilename, 'w'), indent=2)

def prepare_word_encoder(infilename, outfilename, lower=True, pretokenized=True, threshold=0):
    data = []
    with open(infilename) as fin:
        for line in fin:
            splitline = line.strip().split('\t')
            if lower: text = splitline[0].strip().lower()
            else: text = splitline[0].strip()
            if pretokenized: tokens = text.split()
            else: tokens = nltk.word_tokenize(text)
            data += tokens
    counter = collections.Counter(data) 
    encoder = {}
    for char, freq in counter.most_common():
        encoder[char] = len(encoder)
        if freq == threshold: break
    json.dump(encoder, open(outfilename, 'w'), indent=2)


if __name__ == '__main__':
    # filenames = ['data/pdf/C-1013641.pdf', 'data/pdf/C-1016457.pdf', 'data/pdf/C-1131091.pdf', 'data/pdf/C-1193336.pdf']
    # labels = map(str, range(len(filenames)))
    # prepare_data(filenames, labels, 'data/cornerstone/cornerstone_train.tsv')
    prepare_char_encoder('data/cornerstone/cornerstone_train.tsv', 'data/encoder_char.json', lower=True)
    prepare_word_encoder('data/cornerstone/cornerstone_train.tsv', 'data/encoder_word.json', lower=True, pretokenized=False)
