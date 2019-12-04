from flask import Flask, jsonify, request
from flask_restful import reqparse, abort, Api, Resource
from flask import redirect, url_for
import re
import jieba
import numpy as np
from sklearn.decomposition import PCA
from gensim.models import Word2Vec
from flask_cors import CORS

def load_model(fname: str):
    return Word2Vec.load(fname)


class AutomaticTextSummarizer:
    def __init__(self, word_vec_model, a=1e-4, C_compute_model='cs', n_neighbors=5, summarize_sentences_size=0.3):
        """
        Parameters
        ----------
        word_vec_model: Word2Vec
            Word2Vec model
        a: float
            sentence embedding weight
        C_compute_model: str
            C_i calculation model
            'cs'
                Cosine similarity
            'cc'
                Correlation coefficient
        n_neighbors: int
            KNN smooth window size
        summarize_sentences_size: float
            summarize sentences size, should be between 0.0 and 1.0
        """
        self.word_vec_model = word_vec_model
        self.a = a
        self.C_compute_model = C_compute_model
        self.n_neighbors = n_neighbors
        self.summarize_sentences_size = summarize_sentences_size

    def summarize(self, title: str, content: str):
        """
        Parameters
        ----------
        content: str
            news content
        title: str
            news title
        Returns
        -------
        summarize: dict
            summarization: str
            C: dict
            Vs: dict
            Vt: list
            Vc: list
        """
        sentences = self.cut_sentences(content)

        all_sentence_embeddings = self.compute_sentence_embeddings([title, content] + sentences)
        Vt = all_sentence_embeddings[0]
        Vc = all_sentence_embeddings[1]
        Vs = all_sentence_embeddings[2:]

        C = self.compute_C(Vs, Vt, Vc)

        summarize_sentences_size = int(self.summarize_sentences_size * len(sentences))
        summarize_sentences = [(sentence, index) for sentence, index, _ in
                               sorted(zip(sentences, range(len(sentences)), C), key=lambda e: e[2], reverse=True)[
                               :summarize_sentences_size]]
        summarization = ''.join([sentence for sentence, _ in sorted(summarize_sentences, key=lambda e: e[1])])

        return {
            'summarization': summarization,
            'C': dict(zip(sentences, C)),
            'Vs': dict(zip(sentences, Vs)),
            'Vc': Vc,
            'Vt': Vt
        }

    def compute_sentence_embeddings(self, sentences):
        model = self.word_vec_model
        a = self.a
        sentence_embeddings = np.array(
            [self.compute_sentence_embedding(sentence, model, a) for sentence in sentences])
        u = self.compute_first_principal_component(sentence_embeddings)
        return (1 - np.dot(u, u.T)) * sentence_embeddings

    def compute_C(self, Vs, Vt, Vc):
        if self.C_compute_model == 'cs':
            C_compute_model = self.cosine_similarity
        elif self.C_compute_model == 'cc':
            C_compute_model = self.correlation_coefficient
        else:
            C_compute_model = self.cosine_similarity

        return self.KNN_smooth([C_compute_model(Vsi, Vt, Vc) for Vsi in Vs])

    def KNN_smooth(self, C):
        n_neighbors = self.n_neighbors
        smooth_C = []
        C_len = len(C)
        for i in range(C_len):
            begin_index = i - n_neighbors
            if begin_index < 0: begin_index = 0
            
            end_index = i + n_neighbors + 1
            if end_index > C_len: end_index = C_len
            
            smooth_C.append(sum(C[begin_index:end_index]) / (end_index - begin_index))
            
        return smooth_C

    def set_word_vec_model(self, word_vec_model):
        self.word_vec_model = word_vec_model

    def set_a(self, a):
        self.a = a

    def set_C_compute_model(self, C_compute_model):
        self.C_compute_model = C_compute_model

    def set_n_neighbors(self, n_neighbors):
        self.n_neighbors = n_neighbors

    def set_summarize_sentences_size(self, summarize_sentences_size):
        self.summarize_sentences_size = summarize_sentences_size

    @staticmethod
    def cosine_similarity(Vsi, Vt, Vc):
        """
        余弦相似度
        """
        si_t_cs = abs(np.sum(Vsi * Vt) / (np.sqrt(np.sum(Vsi ** 2)) * np.sqrt(np.sum(Vt ** 2))))

        si_c_cs = abs(np.sum(Vsi * Vc) / (np.sqrt(np.sum(Vsi ** 2)) * np.sqrt(np.sum(Vc ** 2))))

        return (si_t_cs + si_c_cs) / 2

    @staticmethod
    def correlation_coefficient(Vsi, Vt, Vc):
        """
        相关系数
        """
        pass

    @staticmethod
    def compute_sentence_embedding(sentence, model, a):
        """
        如果词向量不存在应该如何处理？（目前的处理是忽略该词向量）(out-of-word)
        """
        words = AutomaticTextSummarizer.cut_words(sentence)
        # 词向量加权求和
        word_embeddings = np.array(
            [a / (a + (model.wv.vocab[word].count / model.corpus_total_words)) * model.wv[word] for word in words if
             word in model.wv])
        result = np.sum(word_embeddings, axis=0) / word_embeddings.shape[0]
        if not isinstance(result, np.ndarray):
            result = np.zeros(400, dtype=float)
        return result

    @staticmethod
    def compute_first_principal_component(sentence_embeddings):
        pca = PCA(n_components=1)
        pca.fit(sentence_embeddings)
        return pca.components_

    @staticmethod
    def cut_words(content: str):
        return [word for word in list(jieba.cut(AutomaticTextSummarizer.clean_data(content))) if word != ' ']

    @staticmethod
    def cut_sentences(content: str):
        sentence_division = '[〇一-\u9fff㐀-\u4dbf豈-\ufaff𠀀-\U0002a6df𪜀-\U0002b73f𫝀-\U0002b81f丽-\U0002fa1f⼀-⿕⺀-⻳0-9a-zA-G ＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､\u3000、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔·]*[！？｡。][」﹂”』’》）］｝〕〗〙〛〉】]*'
        return re.findall(sentence_division, content)

    @staticmethod
    def clean_data(content: str):
        chinese_punctuation = '＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､\u3000、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔·！？｡。'
        english_punctuation = ',.?;:\'"`~!'
        special_char = r'<>/\\|\[\]{}@#\$%\^&\*\(\)-\+=_\n'
        return re.sub(
            '(?P<punctuation>[{}]|[{}])|(?P<special_char>[{}])'.format(chinese_punctuation, english_punctuation,
                                                                       special_char), ' ', content)


#path = '/home/student/project/project-01/no_name_group/model/word2vec_normal_full.model'
path = '/home/student/project/project-01/no_name_group/model/word2vec_selective_10000.model'

model = load_model(path)

automatic_text_summarizer = AutomaticTextSummarizer(model)
automatic_text_summarizer.set_n_neighbors(5)
automatic_text_summarizer.set_summarize_sentences_size(0.4)


app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})
api = Api(app)

parser = reqparse.RequestParser()
parser.add_argument('title', type=str,location='form')
parser.add_argument('content', type=str, location='form')

class HelloWorld(Resource):
    def post(self):
        args=parser.parse_args()
        title = args['title']
        content = args['content']
        print(title)
        result = automatic_text_summarizer.summarize(title, content)
        return jsonify(summarization= result['summarization'])

api.add_resource(HelloWorld, '/testing')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5444 , debug=True)

