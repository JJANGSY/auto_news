import os
import pandas as pd
from konlpy.tag import Mecab
from gensim import corpora, models
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer    


class Summary:
    
    def __init__(self, num_sentences=3):
        self.num_sentences = num_sentences
        self.korean_stop_words = set()
    
    def stopword(self, file_path='stop_word.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            self.korean_stop_words = set(line.strip() for line in f)

    def preprocess(self, text): 
        mecab = Mecab(dicpath=r'/tmp/mecab-ko-dic-2.1.1-20180720')
        tokens = [word for word in mecab.pos(text) if word[1][0] in ['N', 'V']]
        tokens = [word for word, pos in tokens if word not in self.korean_stop_words]
        return tokens

    def remove_sources(self, text, sources=['출처=IT동아', '사진=트위터 @DylanXitton','▶△▶️◀️▷ⓒ■◆●©️…※↑↓▲☞ⓒ⅔▼�','([(\[])(.*?)([)\]])|【(.*?)】','<(.*?)>','제보는 카톡 okjebo','  ']):
        for source in sources:
            text = text.replace(source, '')
        return text
    
    def sentence_similarity(self, sentence1, sentence2):
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([sentence1, sentence2])
        return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    def lda_summarize_v2(self, text, num_topics=3, num_words=10, similarity_threshold=0.4):
        text = self.remove_sources(text)
        sentences = sent_tokenize(text)
        tokenized_sentences = [self.preprocess(sentence) for sentence in sentences]
        dictionary = corpora.Dictionary(tokenized_sentences)
        corpus = [dictionary.doc2bow(tokenized_sentence) for tokenized_sentence in tokenized_sentences]
        lda_model = models.ldamodel.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=20)
        topic_keywords = lda_model.show_topics(num_topics=num_topics, num_words=num_words, formatted=False)
        topic_sentences = []

        for topic, keywords in topic_keywords:
            topic_words = [word[0] for word in keywords]
            for sentence in sentences:
                tokenized_sentence = self.preprocess(sentence)
                if any(word in tokenized_sentence for word in topic_words):
                    topic_sentences.append(sentence)

        ranked_sentences = sorted([(sentence, lda_model[corpus[sentences.index(sentence)]][0][1]) for sentence in topic_sentences], key=lambda x: x[1], reverse=True)

        summary = []
        for sentence, score in ranked_sentences:
            if len(summary) >= self.num_sentences:
                break
        
            for chosen_sentence, i in summary:
                if self.sentence_similarity(sentence, chosen_sentence) > similarity_threshold:
                    break
        
            else:
                summary.append((sentence, score))

        summary_sentences = [sentence[0] for sentence in summary]
        return summary_sentences


    def generate_html(self, df):
        html_content = '''<div class="mail_view_contents"><div class="mail_view_contents_inner" data-translate-body-17025=""><div><!--[if !mso]><!-- --><!--&lt;![endif]--><!--[if mso]>            <noscript>              </noscript>    <![endif]--><!--  --><div lang="en" style="-webkit-text-size-adjust: 100%;-ms-text-size-adjust: 100%;padding: 20px 0px;margin: 0 auto;"><div style="height:0px;max-height:0px;border-width:0px;border: 0px;border-color:initial;border-image:initial;visibility:hidden;line-height:0px;font-size:0px;overflow:hidden;display:none;"></div><table><tbody><tr><td></td></tr></tbody></table><table style="width:100%;" border="0"><tbody><tr><td align="center"><!--[if mso]>          <table align="center" style="width:630px;">          <tr><td>            <div>          <![endif]--><!--[if !mso]><!-- --><div style="width:100%;max-width:630px;margin: 0px auto;"><!--&lt;![endif]--><table style="width: 100%;border:0;" cellpadding="0" cellspacing="0"><tbody><tr><td style="text-align:center;font-size:0;box-sizing:border-box;"><img src="https://img.stibee.com/107_1626771509.png" alt="" style="display:inline;vertical-align:bottom;text-align:center;max-width:100% !important;height:auto;border:0;" width="630" loading="lazy"></td></tr></tbody></table><table style="width: 100%;border:0;" cellpadding="0" cellspacing="0"><tbody><tr><td style="word-break:break-all;text-align:left;margin:0px;;line-height:1.7;word-break:break-word;font-size:16px;font-family:noto sans kr, noto sans cjk kr, noto sans cjk, Malgun Gothic, apple sd gothic neo,     nanum gothic, malgun gothic, dotum, arial, helvetica, Meiryo, MS Gothic, sans-serif!important;;-ms-text-size-adjust: 100%;-webkit-text-size-adjust: 100%;color:#000000;padding:0px 15px 0px 15px;"><div style="text-align: center;"><span style="font-size: 60px;">ECONOMY LETTER</span></div></td></tr></tbody></table><table style="width: 100%;border:0;" cellpadding="0" cellspacing="0"><tbody><tr><td style="text-align:center;font-size:0;box-sizing:border-box;"><img src="https://img.stibee.com/107_1626771518.png" alt="" style="display:inline;vertical-align:bottom;text-align:center;max-width:100% !important;height:auto;border:0;" width="630" loading="lazy"></td></tr></tbody></table><table width="100%" cellpadding="0" cellspacing="0" style="border:0;background:none;"><tbody><tr><td style="height: 50px"></td></tr></tbody></table><table width="100%" cellpadding="0" cellspacing="0" style="border:0;background:none;"><tbody><tr><td style="height: 50px"></td></tr></tbody></table><table style="width: 100%;border:0;" cellpadding="0" cellspacing="0"><tbody><tr><td style="word-break:break-all;text-align:left;margin:0px;;line-height:1.7;word-break:break-word;font-size:16px;font-family:noto sans kr, noto sans cjk kr, noto sans cjk, Malgun Gothic, apple sd gothic neo,     nanum gothic, malgun gothic, dotum, arial, helvetica, Meiryo, MS Gothic, sans-serif!important;;-ms-text-size-adjust: 100%;-webkit-text-size-adjust: 100%;color:#000000;padding:15px 0px 0px 0px;"><div style="text-align: center;"><span style="font-weight: bold; color: #000000; font-size: 26px;">HELLO, I'M ECONOMY LETTER</span></div></td></tr></tbody></table><table style="width: 100%;border:0;" cellpadding="0" cellspacing="0"><tbody><tr><td style="word-break:break-all;text-align:left;margin:0px;;line-height:1.7;word-break:break-word;font-size:16px;font-family:noto sans kr, noto sans cjk kr, noto sans cjk, Malgun Gothic, apple sd gothic neo,     nanum gothic, malgun gothic, dotum, arial, helvetica, Meiryo, MS Gothic, sans-serif!important;;-ms-text-size-adjust: 100%;-webkit-text-size-adjust: 100%;color:#000000;padding:15px 15px 15px 15px;"><p style="text-align: left;">이코노미 레터는 1시간 마다 주요 경제 뉴스를 요약해서 전달해주는 서비스입니다. 1시간마다 경제 뉴스를 이메일로 편하게 확인해보세요!</p></td></tr></tbody></table><table width="100%" cellpadding="0" cellspacing="0" style="border:0;background:none;"><tbody><tr><td style="height: 50px"></td></tr></tbody></table><table style="width: 100%;border:0;" cellpadding="0" cellspacing="0"><tbody><tr><td style="word-break:break-all;text-align:left;margin:0px;;line-height:1.7;word-break:break-word;font-size:16px;font-family:noto sans kr, noto sans cjk kr, noto sans cjk, Malgun Gothic, apple sd gothic neo,     nanum gothic, malgun gothic, dotum, arial, helvetica, Meiryo, MS Gothic, sans-serif!important;;-ms-text-size-adjust: 100%;-webkit-text-size-adjust: 100%;color:#000000;padding:15px 0px 15px 0px;"><div style="text-align: center;"><span style="font-weight: bold;"><span style="font-size: 26px;">NEWS SUMMARY&nbsp;</span>&nbsp;</span></div></td></tr></tbody></table><table style="width: 100%;border:0;" cellpadding="0" cellspacing="0"><tbody><tr><td style="word-break:break-all;text-align:left;margin:0px;;line-height:1.7;word-break:break-word;font-size:16px;font-family:noto sans kr, noto sans cjk kr, noto sans cjk, Malgun Gothic, apple sd gothic neo,     nanum gothic, malgun gothic, dotum, arial, helvetica, Meiryo, MS Gothic, sans-serif!important;;-ms-text-size-adjust: 100%;-webkit-text-size-adjust: 100%;color:#000000;padding:15px 0px 15px 0px;">'''
        
        for index, row in df.iterrows():
            content = row['content']
            summary = self.lda_summarize_v2(content, num_topics=3, num_words=10, similarity_threshold=0.4)
            url = row['url']

            html_content += f'<div><span style="text-decoration: underline;">{index+1}. {row["title"]}</span></div><div>- '
            for i in range(min(self.num_sentences, len(summary))):
                html_content += f"{summary[i]} "
            html_content += f'<div>-<span>&nbsp;</span><span style="color: #ff9400;"><a href=""{url}" style="color: #ff9400; text-decoration: none;" target="_blank" rel="noreferrer noopener">{url}</a></span>'
            html_content += "</div></div><div><br></div>"
        
        html_content += '''</td></tr></tbody></table><table width="100%" cellpadding="0" cellspacing="0" style="border:0;background:none;"><tbody><tr><td style="height: 40px"></td></tr></tbody></table><table width="100%" cellpadding="0" cellspacing="0" style="border:0;background:none;"><tbody><tr><td style="height: 40px"></td></tr></tbody></table><table width="100%" cellpadding="0" cellspacing="0" style="border:0;"><tbody><tr><td style="text-align:center;margin:0px;;line-height:1.7;word-break:break-word;font-size:12px;font-family:noto sans kr, noto sans cjk kr, noto sans cjk, Malgun Gothic, apple sd gothic neo,     nanum gothic, malgun gothic, dotum, arial, helvetica, Meiryo, MS Gothic, sans-serif!important;-ms-text-size-adjust: 100%;-webkit-text-size-adjust: 100%;color:#747579;border:0;"><table border="0" cellpadding="0" cellspacing="0" style="width: 100%"><tbody><tr><td style="padding:15px 0px 15px 0px;text-align:center;"><div style="text-align: left;"><span style="font-weight: bold;">경제레터</span><span>&nbsp;by newsfeeding_team6</span><br><a href="https://github.com/kdt-service/AutoNewsFeeding_6" target="_blank" style="color: #747579;" rel="noreferrer noopener">https://github.com/kdt-service/AutoNewsFeeding_6</a></div></td></tr></tbody></table></td></tr></tbody></table></div><!--[if mso]>          </div></td></tr>          </table>          <![endif]--></td></tr></tbody></table><table align="center" border="0" cellpadding="0" cellspacing="0" width="100%"><tbody><tr></tr></tbody></table><div style="padding: 10pt 0cm 10pt 0cm;"><p align="center" style="text-align: center;"><span lang="EN-US"></span></p></div></div>'''

        count = len(os.listdir('./news'))   # news폴더 내의 파일목록 갯수

        file_path = f'news/output_test_{count}.html'
        with open(file_path, "w", encoding='utf-8') as file:
            file.write(html_content)