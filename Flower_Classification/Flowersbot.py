import telebot
from telebot import types
import tensorflow as tf
import cv2
import numpy as np


bot = telebot.TeleBot("7981577956:AAFL2o0Iz389HAtznBlfpQHHSn2s4hHCfow", parse_mode=None)
model = tf.keras.models.load_model("weights\Flowers.h5")
flowers_name = ['bluebell', 'buttercup', 'coltsfoot', 'cowslip', 'crocus', 'daffodil','daisy', 'dandelion', 'fritillary', 'iris', 'lilyvalley', 'pansy', 'snowdrop', 'sunflower', 'tigerlily', 'tulip', 'windflower']
@bot.message_handler(commands=['start'])
def send_welcome(message): 
    msg = bot.send_message(message.chat.id,"سلام "+str(message.chat.first_name)+"✋ به بات تلگرام تشخیص گل خوش آمدی"+" \n"+
                            "/Photo- حدست گل ")
    
@bot.message_handler(commands=['Photo'])
def send_photo(message): 
    msg = bot.send_message(message.chat.id,"🫡 لطفا عکس یه گل بفرست تا اسمش رو حدس بزنم 👀")
    bot.register_next_step_handler(msg,photo)
	
@bot.message_handler(content_types = ["Photo"])
def photo(message):
    fileID = message.photo[-1].file_id
    file_info = bot.get_file(fileID)
    downloaded_file = bot.download_file(file_info.file_path)

    with open("Flowers.jpg", 'wb') as new_file:
        new_file.write(downloaded_file)

    img = cv2.imread("Flowers.jpg")
    image = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.resize(image ,(224,224))
    img = img / 255
    img = img.reshape(1,224,224,3)        
    result = np.argmax(model.predict(img))
    print(result)
    print(flowers_name[result])

    bot.send_message(message.chat.id,f' فکر کنم 🤔 اسم گلت {flowers_name[result]} باشه >>> 🪴 <<<  اینم گلدونش از طرف من ')
bot.infinity_polling()
