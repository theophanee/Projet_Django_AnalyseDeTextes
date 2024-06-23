#text_processing.py
import requests
from bs4 import BeautifulSoup
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from collections import defaultdict
import string
from collections import Counter
import re
import nltk
from nltk import ngrams
from collections import Counter
from collections import defaultdict
import random

class TextProcessor:

    var_text = None
    var_ngrams = None

    def __init__(self):
        pass

    def get_html_content(self,source):
        # Vérifier si la source est une URL ou un chemin de fichier
        if source.startswith("http://") or source.startswith("https://"):
            # Si c'est une URL, récupérer le contenu de l'URL
            response = requests.get(source)
            # Vérifier si la requête a réussi
            if response.status_code == 200:
                return response.text
            else:
                print("Erreur lors de la récupération de la page :", response.status_code)
                return None
        else:
            # Si c'est un chemin de fichier, lire le contenu du fichier
            try:
                with open(source, "r", encoding="utf-8") as file:
                    return file.read()
            except FileNotFoundError:
                print("Fichier introuvable :", source)
                return None

    def extract_text(self,html_content):
        if html_content is None:
            print("HTML content is None.")
            return None
        
        # Use BeautifulSoup to parse the HTML
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Check if parsing was successful
        if soup is None:
            print("Failed to parse HTML content.")
            return None
        
        full_text = soup.get_text()
        # on verifie si url contient "gutenberg"
        if(html_content.find("gutenberg") != -1):
            # on commence apres ce qui est entre les etoiles
            patternDebut = "\*\*\* [^*]+ \*\*\*"
            regex = re.compile(patternDebut, re.DOTALL)

            match = regex.search(full_text)
            end_index = full_text.find("FIN")

            if match:
                # Retourne le text apres l'expression reguliere et avant le mot "fin"
                if end_index:
                    return full_text[match.end():end_index]
                # Retourne le text apres l'expression reguliere jusqu'a la fin
                else :
                    return full_text[match.end():]
            else:
                return "Marker not found."
        else :
            # Extraire le texte du contenu de la page
            return soup.get_text()
        
    def justify_text_greedy(self, text, line_width, align='left'):
        paragraphs = re.split(r'\n\s*\n', text)  # Split the text into paragraphs
        justified_paragraphs = []
        for paragraph in paragraphs:
            justified_paragraphs.append(self.justify_paragraph_greedy(paragraph, line_width, align))
        return '\n\n'.join(justified_paragraphs)

    def justify_paragraph_greedy(self, paragraph, line_width, align='left'):
        words = paragraph.split()
        lines = []
        current_line = []
        current_width = 0

        for word in words:
            word_width = len(word)
            
            # Check if the word can be added to the current line
            if current_width + len(current_line) + word_width > line_width:
                # Add the current line to the list of lines
                if align == 'right':
                    lines.append(' '.join(current_line).rjust(line_width)) 
                elif align == 'both':
                    num_spaces_needed = line_width - current_width
                    num_spaces_between_words = len(current_line) - 1
                    if num_spaces_between_words > 0:
                        spaces_per_word = num_spaces_needed // num_spaces_between_words
                        extra_spaces = num_spaces_needed % num_spaces_between_words
                        justified_line = ''
                        for idx, word_in_line in enumerate(current_line[:-1]):
                            justified_line += word_in_line + (' ' * spaces_per_word + (' ' if idx < extra_spaces else ''))
                        justified_line += current_line[-1]
                    else:
                        justified_line = current_line[0] 
                    lines.append(justified_line) 
                else:
                    lines.append(' '.join(current_line))
                # Reset the current line and current width
                current_line = [word]
                current_width = word_width
            else:
                current_line.append(word)
                current_width += word_width

        # Add the last line
        if align == 'right':
            lines.append(' '.join(current_line).rjust(line_width))
        elif align == 'both':
            justified_line = ' '.join(current_line)
            lines.append(justified_line + (' ' * (line_width - len(justified_line))))
        else:
            lines.append(' '.join(current_line))

        # Concatenate the lines into a single justified text
        justified_text = '\n'.join(lines)
        
        return justified_text

    def justify_text_dynamic(self, text, line_width, align='left'):
            paragraphs = re.split(r'\n\s*\n', text)  # Split the text into paragraphs
            justified_paragraphs = []
            for paragraph in paragraphs:
                justified_paragraphs.append(self.justify_paragraph_dynamic(paragraph, line_width, align))
            return '\n\n'.join(justified_paragraphs)

    def justify_paragraph_dynamic(self, text, line_width, align='left'):
        words = text.split()
        n = len(words)
        dp = [float('inf')] * (n + 1)
        split_index = [-1] * (n + 1)

        dp[0] = 0

        for i in range(1, n + 1):
            line_length = 0
            for j in range(i - 1, -1, -1):
                word_width = len(words[j])
                if j < i - 1:
                    line_length += 1  # Add space between words
                line_length += word_width

                if line_length <= line_width:
                    gap = line_width - line_length
                    badness = gap ** 3  # Calculate badness
                    if dp[i] > dp[j] + badness:
                        dp[i] = dp[j] + badness
                        split_index[i] = j

        lines = []
        idx = n
        while idx > 0:
            lines.append(' '.join(words[split_index[idx]:idx]))
            idx = split_index[idx]

        lines = reversed(lines)
        if align == 'right':
            lines = [line.rjust(line_width) for line in lines]
        elif align == 'both':
            lines = [self.justify_line(line, line_width) for line in lines]

        return '\n'.join(lines)


    def justify_line(self, line, line_width):
        words = line.split()
        num_words = len(words)
        num_spaces_needed = line_width - sum(len(word) for word in words)
        num_spaces_between_words = max(num_words - 1, 1)
        spaces_per_word = num_spaces_needed // num_spaces_between_words
        extra_spaces = num_spaces_needed % num_spaces_between_words
        justified_line = ''
        for idx, word in enumerate(words[:-1]):
            justified_line += word + ' ' * (spaces_per_word + (1 if idx < extra_spaces else 0))
        justified_line += words[-1]
        return justified_line


    def count_word_frequencies(slef,text):
        # on utilise une expression reguliere pour recuperer tous les mots
        # text = re.sub(r'\W+', ' ', text)
        # # on converti en minuscule
        # words = text.lower().split()

        words = re.findall(r'\b[\w\'-]+\b', text)
        # Convert all words to lowercase
        words = [word.lower() for word in words]
        
        # On utilise Counter pour compter les occurences de chaque mot
        word_counts = Counter(words)

        return sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

    def try_detect_language(self,text):
        # On definit quelques mots utilisés dans diffentes langues
        stop_words = {
            'Anglais': {'the', 'and', 'is', 'that', 'to', 'then', 'of', 'the', 'they'},
            'Français': {'et', 'la', 'les', 'elle', 'de', 'l', 'à', 'est'},
            'Allemand': {'und', 'die', 'der', 'den', 'ist', 'zu','das', 'ein', 'eine'},
            'Italien': {'e', 'è', 'che', 'per', 'gli', 'di', 'una', 'del', 'lo'}
        }
        
        # On met tout en miniscule
        text = text.lower()
        words = text.split()
        
        # On compte le nombre de mots trouvés pour chaque langue dans le text
        word_counts = Counter(words)

        # on fait la somme de ces mots
        language_scores = {
            lang: sum(word_counts[word] for word in words_set if word in word_counts)
            for lang, words_set in stop_words.items()
        }

        # On prend celui qui a le plus de match
        most_likely_language = max(language_scores, key=language_scores.get)
        
        return most_likely_language, language_scores

    def count_sentences(self,text):
        # Utilisation d'une expression régulière pour diviser le texte en phrases
        # On suppose que les phrases se terminent par un point, un point d'exclamation ou un point d'interrogation, suivis d'un espace ou de la fin du texte.
        sentences = re.split(r'(?<=[.!?])\s+(?!FIN|\s*$)', text)
        return len(sentences)

    def count_paragraphs(self,text):
        # Cette expression régulière suppose que les paragraphes sont séparés par une ou plusieurs lignes vides.
        paragraphs = re.split(r'\n\s*\n(?!FIN|\s*$)', text)
        return len(paragraphs)

    def count_letter_frequency(self,text):
        # Convertir le texte en minuscules pour une correspondance insensible à la casse
        text_lower = text.lower()
        
        # Initialiser un dictionnaire pour stocker les fréquences des lettres
        letter_frequency = {}
        
        # Itérer sur chaque lettre de l'alphabet
        for letter in string.ascii_lowercase:
            # Compter le nombre d'occurrences de la lettre dans le texte
            frequency = text_lower.count(letter)
            # Stocker la fréquence de la lettre dans le dictionnaire
            letter_frequency[letter] = frequency
        
        return letter_frequency

    def generate_ngram_counts(self, text, n):
        # Séparons le texte en mots en incluant la ponctuation
        # words = re.split(r'(\r\n|\n\n|\n| +)', text)

        words = re.findall(r'\S+\s*', text)

        # on prend en compte les retours a la ligne
        words = [token for token in words if token != '' and token != ' ']
        # Utilisons Counter pour simplifier la collecte des fréquences
        n_gram_counts = Counter(tuple(words[i:i+n]) for i in range(len(words) - n + 1))

        # print("ngrams",n_gram_counts)
        
        # Retourne le dictionnaire des n-grammes triés par fréquence décroissante
        # La conversion en dict n'est nécessaire que si vous avez besoin spécifiquement d'un dictionnaire
        return dict(n_gram_counts.most_common())

        
    def build_ngram_model(self, n_gram_counts):
        # print(n_gram_counts)
        model = defaultdict(Counter)
        # on organise les ngrams
        for n_gram, count in n_gram_counts.items():
            prefix = n_gram[:-1]
            next_word = n_gram[-1]
            model[prefix][next_word] += count
        # print(model)
        return model

    def predict_next_word(self, current_word, model, used_words, recent_words):
        # print(model)
        # on recupere les cles
        # print("current",current_word)
        possible_next_words = model[current_word]
        if possible_next_words:
            # on prend les mots qui apparaissent le plus de fois en enlevant ceux utilisés recemment
            max_freq = max(possible_next_words.values())
            max_freq_words = [word for word, freq in possible_next_words.items() if freq == max_freq]
            # print("max_freq_words 1",max_freq_words)
            max_freq_words = [word for word in max_freq_words if word not in used_words]
            # print("max_freq_words 2",max_freq_words)
            
            # si ils ont tous ete utilisé recemment, on prend les valeurs utilisées plus d'une fois 
            if not max_freq_words:
                max_freq_words = [word for word, freq in possible_next_words.items() if word not in recent_words and freq > 1]
                # sinon on prend les valeurs
                if not max_freq_words:
                    # si il n'y en a pas, on prend n'importe lequel d'entre eux
                    max_freq_words = [word for word, freq in possible_next_words.items() if word not in recent_words]
                    # sinon on prend la valeur même si elle est deja utilisée
                    if not max_freq_words:
                        max_freq_words = [word for word, freq in possible_next_words.items()]
                # print("freq",possible_next_words.items())
                # print("recent",recent_words)
                # print("deja utilise",max_freq_words)
            # on choisit un au hasard
            selected_word = random.choice(max_freq_words) if max_freq_words else None
            used_words.add(selected_word)
            # print("selected",selected_word)
            return selected_word
        else:
            print("Aucun mot possible")
            return " "
        
    def get_first_next_word_user(self, model, word):
        # on recupere le tuple correspondant (tuple ayant comme premier elt le mot predi)
        possible_starting_word = [key for key in model.keys() if word in key[0].split()]
        # On choisi au hasard
        if possible_starting_word != []:
            return random.choice(possible_starting_word)
        else:
            return None
        
    def get_first_next_word(self, model, word):
        # on recupere le tuple correspondant (tuple ayant comme premier elt le mot predi)
        possible_starting_word = [key for key in model.keys() if word in key[0]]
        # On choisi au hasard
        if possible_starting_word != []:
            return random.choice(possible_starting_word)
        else:
            return None
        
    def get_first_word_generate(self, model):
        # on cherche tous les tuples qui ont comme dernier elt une ponctuation de fin
        possible_previous_starting_word = [key for key in model.keys() if "." in key[-1] or "?" in key[-1] or "!" in key[-1]]
        # on choisi un tuple au hasard
        previous_starting_word = random.choice(possible_previous_starting_word)
        # on cherche le prochain mot
        predict = self.predict_next_word(previous_starting_word, model, set(), set())
        return self.get_first_next_word(model, predict)
    
    def get_len_sentence(self, sentence):
        return len([word for word in sentence if "\n" not in word])
        
    def generate_sentences(self, model, start_word, max_words):
        # TODO : si n mots entrés ? faire aleatoire ? prendre n-1 mots ?
        # TODO : essayer de terminer la phrase en n mots si OK : la finir, sinon, s'arreter a aux n mots
        sentences = []
        # pas de nombre de mots entrés, on fixe un nombre
        if(max_words == ""):
            max_words = 400
        # si l'utilisateur entre un mot
        if(start_word != ""):
            if isinstance(start_word, str):
                # Converti string en tuple
                initial_words = tuple(word for word in start_word.split())
            # qu'il est dans notre model
            # print(model)
            # print("init", initial_words)
            if initial_words in model:
                # print("in model")
                current_word = initial_words
            else : 
                # ou qu'il appartient au texte, on commence avec
                word_in = self.get_first_next_word_user(model, start_word)
                if(word_in):
                    # print("not in mais in text")
                    current_word = word_in
                else:
                    # print("not in alea")
                    # sinon on genere un mot au hasard
                    current_word = self.get_first_word_generate(model)
        # sinon on prend un aleatoirement
        else :
            # print("pas mot, alea")
            current_word = self.get_first_word_generate(model)
        sentence = list(current_word)
        # print("sentence",sentence)
        used_words = set(sentence)
        # on garde les mots utilisés recements
        recent_words = list(current_word)
        # pas de repetition dans les 5 mots
        buffer_size = 5 
        while self.get_len_sentence(sentence) < max_words:
            if current_word in model:
                # on genere le prochain mot
                next_word = self.predict_next_word(current_word, model, used_words, recent_words)
                if next_word:
                    # on l'ajoute a notre phrase et aux mots recemment utilisés
                    sentence.append(next_word)
                    recent_words.append(next_word)
                    # pas plus de buffer_size mots recemment utilises, donc on en enleve un a chaque fois
                    if len(recent_words) > buffer_size:
                        recent_words.pop(0)
                    previous_word = current_word
                    current_word = tuple(sentence[-(len(current_word)):])
                else:
                    print("Pas de next word ! ")
                    break
            else:
                print(current_word, " pas dans model")
                break

        sentences.append(' '.join(sentence))
        return ' '.join(sentences)
    
    def calculate_statistics(self,text, ngrame):
        word_frequencies = self.count_word_frequencies(text)
        letter_frequencies = self.count_letter_frequency(text)
        language, language_scores = self.try_detect_language(text)
        num_sentences = self.count_sentences(text)
        num_paragraphs = self.count_paragraphs(text)
        TextProcessor.var_ngrams = self.generate_ngram_counts(text, ngrame)
        TextProcessor.var_text = text
        ngrames = str(TextProcessor.var_ngrams)
        feelings = self.analyze_sentiment(text, self.load_sentiment_dictionary(language))

        # print(TextProcessor.var_ngrams)

        stats = {
            'word_frequencies': word_frequencies,  #frequence des mots
            'num_unique_words': len(word_frequencies),  # nombre de mots différents
            'letter_frequencies': letter_frequencies, #frequence des lettres
            'language': language,                      #langue
            'num_sentences': num_sentences,  # nombre de phrases
            'num_paragraphs': num_paragraphs,  # nombre de paragraphes
            'n-grames': ngrames,
            'feelings': feelings
        }

        return stats
    
    def generation_text(self, word, num_sentences):
        model = self.build_ngram_model(TextProcessor.var_ngrams)
        generate_text = self.generate_sentences(model, word, num_sentences)

        text_genere = {
            'generate_text': generate_text
        }

        return text_genere

    def load_sentiment_dictionary(self, language):
        sentiment_dict = {}

        if language == 'Français':
            url = 'https://raw.githubusercontent.com/leopold-fr/humeur/master/lexique/lexique_des_emotions.txt'
        elif language == 'Anglais':
            url = 'https://raw.githubusercontent.com/fnielsen/afinn/master/afinn/data/AFINN-111.txt'
        elif language == 'Allemand':
            url = 'https://raw.githubusercontent.com/WinfriedSchulze/SentiWS/master/SentiWS_v2.0_Negative.txt'
        elif language == 'Italien':
            url = 'https://raw.githubusercontent.com/r0zetta/sentix/master/it/sentix_it.txt'
        else:
            print("Langue non prise en charge.")
            return sentiment_dict

        response = requests.get(url)
        if response.status_code == 200:
            for line in response.text.split('\n'):
                if line and not line.startswith(';'):
                    word, score = line.strip().split('\t')
                    sentiment_dict[word] = int(score)
        else:
            print("Impossible de récupérer le dictionnaire de sentiment pour la langue spécifiée.")

        return sentiment_dict

    def analyze_sentiment(self,text, sentiment_dict):
        words = text.split()
        sentiment_score = 0
        for word in words:
            if word.lower() in sentiment_dict:
                sentiment_score += sentiment_dict[word.lower()]
        if sentiment_score < 0 : 
            return "Négatif"
        if sentiment_score >0 : 
            return "Positif"
        return "Neutre"
