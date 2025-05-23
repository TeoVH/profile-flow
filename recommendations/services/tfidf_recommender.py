# recommendations/services/tfidf_recommender.py

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from jobs.models import Job
from interactions.models import Interaction
from .base_recommender import BaseRecommender

class TFIDFRecommender(BaseRecommender):
    def obtener_corpus(self, user):
        todas_las_vacantes = Job.objects.all()
        textos_vacantes = [f"{job.title} {job.description} {job.keywords}" for job in todas_las_vacantes]

        vacantes_interes_usuario = Job.objects.filter(interactions__user=user).distinct()
        texto_usuario = " ".join([
            f"{job.title} {job.description} {job.keywords}"
            for job in vacantes_interes_usuario
        ])

        return textos_vacantes, texto_usuario, todas_las_vacantes

    def vectorizar(self, textos):
        vectorizer = TfidfVectorizer(stop_words="english")
        return vectorizer.fit_transform(textos)

    def calcular_similitud(self, matriz):
        return cosine_similarity(matriz[-1], matriz[:-1]).flatten()

    def seleccionar(self, similitudes, objetos, top_n):
        top_indices = similitudes.argsort()[-top_n:][::-1]
        return [list(objetos)[i] for i in top_indices]

# Función de ayuda externa
def recomendar_vacantes(user, top_n=5):
    recomendador = TFIDFRecommender()
    return recomendador.recomendar(user, top_n)
