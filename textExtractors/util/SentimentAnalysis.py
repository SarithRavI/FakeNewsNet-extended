from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class SentimentAnalyzer:
    def __init__(self):
        self.vaderAnalyzer = SentimentIntensityAnalyzer()

    def getVaderSentimentScores(self, content):
        return self.vaderAnalyzer.polarity_scores(content)

    @staticmethod
    def getTextBlobSubjectivity(content):
        return TextBlob(content).sentiment.subjectivity
