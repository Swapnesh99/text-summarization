import math
import io
from nltk import sent_tokenize, word_tokenize, PorterStemmer,pos_tag
from nltk.corpus import stopwords


stopWords = set(stopwords.words("english"))

#Limiting length for the sentence in summary
#Beyond this length the 'Length score' will decrease
ideal=20

def _create_frequency_matrix(sentences):
    frequency_matrix = {}
    stopWords = set(stopwords.words("english"))
    ps = PorterStemmer()
    # here i denotes the index of sentence
    for i,sent in list(enumerate(sentences)):
        freq_table = {}
        words = word_tokenize(sent)
        for word in words:
            word = word.lower()
            word = ps.stem(word)
            if word in stopWords:
                continue

            if word in freq_table:
                freq_table[word] += 1
            else:
                freq_table[word] = 1

        frequency_matrix[sent[:15]+str(i)] = freq_table

    return frequency_matrix


def _create_tf_matrix(freq_matrix):
    tf_matrix = {}

    for sent, f_table in freq_matrix.items():
        tf_table = {}

        count_words_in_sentence = len(f_table)
        for word, count in f_table.items():
            tf_table[word] = count / count_words_in_sentence

        tf_matrix[sent] = tf_table

    return tf_matrix


def _create_documents_per_words(freq_matrix):
    word_per_doc_table = {}

    for sent, f_table in freq_matrix.items():
        for word, count in f_table.items():
            if word in word_per_doc_table:
                word_per_doc_table[word] += 1
            else:
                word_per_doc_table[word] = 1
    
    return word_per_doc_table


def _create_idf_matrix(freq_matrix, count_doc_per_words, total_documents):
    idf_matrix = {}

    for sent, f_table in freq_matrix.items():
        idf_table = {}

        for word in f_table.keys():
            idf_table[word] = math.log10(total_documents / float(count_doc_per_words[word])) #here total_document means the total number of sentences

        idf_matrix[sent] = idf_table

    return idf_matrix


def _create_tf_idf_matrix(tf_matrix, idf_matrix):
    tf_idf_matrix = {}

    for (sent1, f_table1), (sent2, f_table2) in zip(tf_matrix.items(), idf_matrix.items()):

        tf_idf_table = {}

        for (word1, value1), (word2, value2) in zip(f_table1.items(),
                                                    f_table2.items()):  # here, keys are the same in both the table
            tf_idf_table[word1] = float(value1 * value2)

        tf_idf_matrix[sent1] = tf_idf_table

    return tf_idf_matrix

################################### SCORE FOR TF_IDF ############################
def _score_sentences(tf_idf_matrix) -> dict:
    """
    score a sentence by its word's TF
    Basic algorithm: adding the TF frequency of every non-stop word in a sentence divided by total no of words in a sentence.
    :rtype: dict
    """

    sentenceValue = {}

    for sent, f_table in tf_idf_matrix.items():
        total_score_per_sentence = 0

        count_words_in_sentence = len(f_table)
        for word, score in f_table.items():
            total_score_per_sentence += score

        sentenceValue[sent] = total_score_per_sentence / count_words_in_sentence

    return sentenceValue


################################## SCORE FOR SIMILARITY WITH HEADLINE ############################
def headline_score(headline, sentences):
        """ Gives sentence a score between (0,1) based on percentage of words common to the headline. """
        head_score= dict()
        ps = PorterStemmer()
        title_stems = [ps.stem(w) for w in headline if w not in stopWords]
        # here i denotes the index of sentence
        for i,sent in list(enumerate(sentences)):
            sentence_stems = [ps.stem(w) for w in sent if w not in stopWords]
            count = 0.0
            for word in sentence_stems:
                if word in title_stems:
                    count += 1.0
            score = count / len(title_stems)
            head_score[sent[:15]+str(i)]=score
        return head_score


################################## SCORE FOR POSITION OF SENTENCE ############################
def position_score(sentences):
    score_pos= dict()
    # here i denotes the index of sentence
    for i,sent in list(enumerate(sentences)):
        
        if(i<3 or i>len(sentences)-4):
            score_pos[sent[:15]+str(i)]=1
        else:
            score_pos[sent[:15]+str(i)]=0

    return score_pos

################################## SCORE FOR LENGTH OF SENTENCE ############################

def length_score(sentences):
    score_len= dict()
    # here i denotes the index of sentence
    for i,sent in list(enumerate(sentences)):
        score_len[sent[:15]+str(i)]=1 - math.fabs((ideal - len(sent)) / ideal)
    return score_len

################################## SCORE FOR NOUNS SENTENCE ############################
def pronoun_score(sentences):
    
    score_pro =dict()
    # here i denotes the index of sentence
    for i,sent in list(enumerate(sentences)):
        total_word= len(sent)
        count=0
        for item in pos_tag(word_tokenize(sent)):
            
            if item[1]=='NNP':
                
                count=count+1
           
        score_pro[sent[:15]+str(i)]=float(count/total_word)
    
    return score_pro

################################## Finding total Score for sentences########################
def tot_score(sentenceValue,headline_score,score_pos,score_len,pronoun_score):
    final_score=dict()
    for (sent1, v1), (sent2, v2),(sent3, v3),(sent4, v4),(sent5, v5) in zip(sentenceValue.items(), headline_score.items(),score_pos.items(),score_len.items(),pronoun_score.items()):
        final_score[sent1]=(2.0*v1+1.5*v2+1.0*v3+1.0*v4+1.0*v5)/5
    return final_score


################################# Finding Threshold #################################
def _find_average_score(final_score) -> int:
    """
    Find the average score from the sentence value dictionary
    :rtype: int
    """
    sumValues = 0
    for entry in final_score:
        sumValues += final_score[entry]

    # Average value of a sentence from original summary_text
    average = (sumValues / len(final_score))

    return average

############################# generating Summary ########################
def _generate_summary(sentences, final_score, threshold):
    sentence_count = 0
    summary = ''

    score=dict(zip(sentences,final_score.values()))
    
    sorted_final_score=dict(sorted(score.items(),key= lambda kv:kv[1],reverse=True))
    #print(sorted_final_score)
    #print('\n')
    
    for sentence,score in sorted_final_score.items():
        if score>threshold:
            summary += " " + sentence
            sentence_count += 1
    #print(sentence_count)
    return summary


def run_summarization(headline1,body):
    '''
    Parameters: Headline and Body of the Text to be Summarized
       Returns: Summary of the Text
    
    '''
    headline=headline1
    ####################### PART1: Sentence Tokenization ################################

    sentences = sent_tokenize(body)
    total_documents = len(sentences)
    #print(total_documents)
    
    ####################### PART2:Calculating Total Score of Sentence ################################

    #Create the Frequency matrix of the words in each sentence.
    freq_matrix = _create_frequency_matrix(sentences)
    #print(freq_matrix)

    '''
    Term frequency (TF) is how often a word appears in a document, divided by how many words are there in a document.
    Here the Document is sentence of the text 
    '''
    #Calculate TermFrequency 
    tf_matrix = _create_tf_matrix(freq_matrix)
    #print(tf_matrix)

    #creating table for documents per words
    count_doc_per_words = _create_documents_per_words(freq_matrix)
    #print(count_doc_per_words)

   
    #Calculate IDF 
    idf_matrix = _create_idf_matrix(freq_matrix, count_doc_per_words, total_documents)
    #print(idf_matrix)

    #Calculate TF-IDF 
    tf_idf_matrix = _create_tf_idf_matrix(tf_matrix, idf_matrix)
    #print(tf_idf_matrix)

    # calculate score of the sentence based on TF_IDF 
    sentence_scores = _score_sentences(tf_idf_matrix)
    #print(sentence_scores)

    
    #calculate score of the sentence based on Similarity with Headline 
    head_score=headline_score(headline, sentences)
    #print(head_score)

    #calculate score of the sentence based on position of sentence 
    pos_score=position_score(sentences)
    #print(pos_score)

    #calculate score of the sentence based on Length of sentence 
    len_score=length_score(sentences)
    #print(len_score)

    #calculate score of the sentence based on number of pronouns in sentence 
    pro_score=pronoun_score(sentences)
    #print(pro_score)

    #Calculating Total Score of Sentence
    final_score=tot_score(sentence_scores,head_score,pos_score,len_score,pro_score)
    #print(final_score)
    #print('\n')
    
    #Find the threshold
    threshold = _find_average_score(final_score)
    #print(1.3*threshold)

    ####################### PART3:Generating the Summary by Ranking Sentences based on the Scores  ################################
    summary = _generate_summary(sentences, final_score,1.3*threshold)
    return summary


if __name__ == '__main__':
    #Open Text Document
    f=open(r'E:\Rushi Intern\Text Sumarizer\VW-nyt.txt',encoding='utf8')
    #Fetch the headline
    headline=f.readline()
    #Fetch the body 
    body=f.read()
    #Generate summary
    result = run_summarization(headline,body)
    print(result)
