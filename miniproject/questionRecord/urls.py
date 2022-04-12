from django.conf.urls import url
from django.contrib import admin
from django.urls import path
from django.urls import path, include
from . import views,audioRecognize


urlpatterns = [
    # 微信小程序登录
    # 微信登录页面userinfo
    path('userinfo', views.userinfo),
    path('lectureUpdate', views.LectureUpdate),
    url(r"^$", views.upload),
    url(r"^refreshDatabase/$", audioRecognize.refreshDatabase),
    url(r"^addNewQuestion/$", audioRecognize.addNewQuestion),
    url(r"^getUserInformation/$",views.getUserInformation),
    url(r"^getRankWithLevel/$", views.getRankWithLevel),
    url(r"^getRankWithoutLevel/$", views.getRankWithoutLevel),
    url(r"^getNewQuestion/$", views.getNewQuestion),
    url(r"^getOneQuesiton/$", views.getOneQuesiton),
    url(r"^getWrongQuestion/$", views.getWrongQuestion),
    url(r"^getNotesCollection/$", views.getNotesCollection),
    url(r"^getHistoryNum/$", views.getHistoryNum),
    url(r"^toCollect/$", views.toCollect),
    url(r"^toCancelCollect/$", views.toCancelCollect),
    url(r"^judgeAnswer/$", views.judgeAnswer),
    url(r"^getUserRank/$", views.getUserRank),
    url(r"^recordAnswer/$", views.recordAnswer),
    url(r"^getWrongNum/$", views.getWrongNum),
    url(r"^correctAnswer/$", views.correctAnswer),
    url(r"^signAddScore/$", views.signAddScore),
    url(r"^textToSpeechEN_CN/$", views.textToSpeechEN_CN),
    url(r"^textToSpeechEN/$", views.textToSpeechEN),
    url(r"^GetLectures/$", views.GetLectures),
    url(r"^recognize$", audioRecognize.recognize, name="recognize"),
]
