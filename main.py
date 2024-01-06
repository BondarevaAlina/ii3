import asyncio
from aiogram import Bot, Dispatcher,types
import config
import logging
from aiogram import F
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from aiogram.types import FSInputFile


logging.basicConfig(level=logging.INFO)
bot = Bot(token=config.token)
dp = Dispatcher()


async def main():
    await dp.start_polling(bot)



@dp.message(F.text)
async def answer(message: types.Message):
    if len(message.text) == 4 and 2004<=int(message.text)<=2018:
        text = message.text
        model_active(message.text)
        image_path = FSInputFile('graph.png')
        await message.answer_photo(image_path)

def prepare_data_for_year(data, year,scaler):
    data_year = data[data['ds'].dt.year == year]['y'].values
    scaled_data_year = scaler.transform(data_year.reshape(-1, 1))
    sequences = []
    for i in range(len(scaled_data_year) - 10):
        sequence = scaled_data_year[i:i + 10]
        sequences.append(sequence)
    return np.array(sequences)

def model_active(text:str):

    from tensorflow import keras
    model = keras.models.load_model('model_checkpoint.h5')
    data = pd.read_csv('AEP.csv')
    data.columns = ['ds', 'y']
    data['ds'] = pd.to_datetime(data['ds'])

    # Масштабирование данных
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(data['y'].values.reshape(-1, 1))

    input_year = int(text)
    in_base = True
    if input_year > 2018:
        in_base = False
        input_year = 2017

        
    # Подготовка данных для прогноза на указанный год
    year_seq = prepare_data_for_year(data, input_year, scaler)
    
    # Получение прогнозов для указанного года
    year_predictions = model.predict(year_seq)
    year_predictions = scaler.inverse_transform(year_predictions)

    # Получение фактических данных для указанного года из датасета
    actual_data = data[data['ds'].dt.year == input_year]['y'].values[10:]

    # Визуализация прогнозов и фактических данных для указанного года
    plt.figure(figsize=(10, 6))
    if in_base:
        plt.plot(data[data['ds'].dt.year == input_year]['ds'].values[10:], actual_data, label='Actual Data')
    plt.plot(data[data['ds'].dt.year == input_year]['ds'].values[10:], year_predictions, label='Predictions')
    plt.xlabel('Время')
    plt.ylabel('Средняя потребляемая мощность энергии, кВт')
    if in_base:
        plt.title(f'Предсказание vs Фактические данные за {input_year} год')
    else: 
        plt.title(f'Прогноз для введенного года ')
    plt.legend()
    plt.savefig('graph.png')


if __name__ == '__main__':
    asyncio.run(main())
    