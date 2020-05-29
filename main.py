import telebot
import config
import keras
import cv2
import tensorflow as tf
import numpy as np
keras.backend.set_learning_phase(0)


animals = ['белка', 'кот', 'слон', 'конь', 'курица', 'паук', 'бабочка', 'собака', 'корова', 'овца']
model = tf.keras.models.load_model('ResNet50V2.h5')

bot = telebot.TeleBot(config.TOKEN)


def centering_image(img):
    size = [256, 256]
    img_size = img.shape[:2]
    row = (size[1] - img_size[0]) // 2
    col = (size[0] - img_size[1]) // 2
    resized = np.zeros(list(size) + [img.shape[2]], dtype=np.uint8)
    resized[row:(row + img.shape[0]), col:(col + img.shape[1])] = img
    return resized


def classify(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if img.shape[0] > img.shape[1]:
        tile_size = (int(img.shape[1] * 256 / img.shape[0]), 256)
    else:
        tile_size = (256, int(img.shape[0] * 256 / img.shape[1]))

    img = centering_image(cv2.resize(img, dsize=tile_size))
    img = img[16:240, 16:240]
    img = img[None, :, :, :]
    img = img.astype('float32')
    img /= 255
    return model.predict(img)


@bot.message_handler(commands=['start'])
def start(message):
    photo = open('AnimatedSticker.tgs', 'rb')
    bot.send_sticker(message.chat.id, photo)

    bot.send_message(message.chat.id,
                     "Добро пожаловать, {0.first_name}!\nЯ - <b>{1.first_name}</b>, "
                     "бот созданный угадывать животное по фотографии!".format(
                         message.from_user, bot.get_me()
                     ), parse_mode='html')


@bot.message_handler(content_types=['photo'])
def main_stream(message):
    fileID = message.photo[-1].file_id
    file_info = bot.get_file(fileID)
    downloaded_file = bot.download_file(file_info.file_path)

    with open("image.jpg", 'wb') as new_file:
        new_file.write(downloaded_file)

    predicted = classify('image.jpg')

    bot.send_message(message.chat.id, 'Это <b>{}</b> с вероятностю <b>{}%</b>'.format(
        animals[predicted.argmax()], int(predicted[0, predicted.argmax()] * 100)), parse_mode='html')


bot.polling(none_stop=True)
