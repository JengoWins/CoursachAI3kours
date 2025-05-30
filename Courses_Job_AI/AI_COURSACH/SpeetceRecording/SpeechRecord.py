#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import io
import os
import sys
import pyttsx3
import speech_recognition as sr
from pydub import AudioSegment


def talk(words):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # скорость речи
    engine.setProperty('volume', 0.8)  # громкость (0-1)

    mp3_path = "E:/Courses_Job_AI/AI_COURSACH/SpeetceRecording/temp_voice.mp3"
    engine.save_to_file(words, mp3_path)
    engine.runAndWait()
    return mp3_path


def command(wav_path):
    # Распознаем речь
    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_path) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data, language="ru-RU")
        except sr.UnknownValueError:
            text = "Не удалось распознать речь"
        except sr.RequestError:
            text = "Ошибка сервиса распознавания речи"
    return text


#Голосовые Команды для их выполнения
def ParseZadanie(zadanie: str) -> io.BytesIO:
    """Разбор голосового задания/команды """
    mp3_path = talk(zadanie)
    #AudioSegment.from_mp3(mp3_path)



# узнаем какие голоса есть в системе
engine = pyttsx3.init()
voices = engine.getProperty('voices')
for voice in voices:  # голоса и параметры каждого
    print('------')
    print(f'Имя: {voice.name}')
    print(f'ID: {voice.id}')
    print(f'Язык(и): {voice.languages}')
    print(f'Пол: {voice.gender}')
    print(f'Возраст: {voice.age}')
# Задать голос по умолчанию
# Попробовать установить предпочтительный голос


for voice in voices:
    if 'Vsevolod' in voice.name:
        engine.setProperty('voice', voice.id)
