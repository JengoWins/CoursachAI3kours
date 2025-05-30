#!/usr/bin/env python
# coding: utf-8
import asyncio
import io
import os
from telebot.async_telebot import AsyncTeleBot
#from AI_COURSACH.TestGPT import botDialog, stats
#from AI_Generate_Questien import botDialog, stats
#from TestExamples import botDialog, stats
from pathlib import Path
from pydub import AudioSegment
from AI_COURSACH.SpeetceRecording.SpeechRecord import command, ParseZadanie
from Setting_Intents import botDialog, stats, context
bot = AsyncTeleBot('7908515516:AAGPJuZuu7ViRId3uVFt8eYGx_-iHD69U00')


def send_run():
    print("Запуск бота ...")


# Обработчик голосовых сообщений
@bot.message_handler(content_types=['voice'])
async def handle_voice(message):
    try:
        # Скачиваем голосовое сообщение
        file_info = await bot.get_file(message.voice.file_id)
        downloaded_file = await bot.download_file(file_info.file_path)

        # Создаем file-like объект из bytes
        audio_bytes = io.BytesIO(downloaded_file)

        # Конвертируем OGG в WAV через pydub
        audio = AudioSegment.from_file(audio_bytes, format="ogg")
        wav_data = io.BytesIO()
        audio.export(wav_data, format="wav")
        wav_data.seek(0)  # Важно: переводим указатель в начало

        text = command(wav_data)
        # Определяем намерение
        answer = botDialog(text)

        response = (
            f"Я распознал голосовое сообщение как:\n{text}\n"
            f"Определенное намерение: {answer}"
        )

        print(response)
        ParseZadanie(answer)
        audio = open(r'E:/Courses_Job_AI/AI_COURSACH/SpeetceRecording/temp_voice.mp3', 'rb')
        print("Что это?")
        await bot.send_voice(message.chat.id, audio)
        audio.close()

    except Exception as e:
        await bot.reply_to(message, f"Произошла ошибка при обработке голосового сообщения: {str(e)}")


# Handle '/start' and '/help'
@bot.message_handler(commands=['test_mess'])
async def send_welcome(message):
    text = 'Ха-ха. Видишь ли. Я больше, чем просто сотрудник Junes'
    await bot.reply_to(message, text)


# Handle all other messages with content_type 'text' (content_types defaults to ['text'])
@bot.message_handler(func=lambda message: True)
async def echo_message(message):
    replica = message.text
    answer = botDialog(replica)
    await bot.reply_to(message, answer)

    print("Статистика применения датасетов: ", stats)
    print("Реплика пользователя: ", replica)
    print("Последний контекст из истории: ", context)
    print("Ответ бота: ", answer)
    print()

send_run()
asyncio.run(bot.polling())
