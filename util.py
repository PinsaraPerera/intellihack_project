import tensorflow as tf
# # Hide GPU from visible devices
tf.config.set_visible_devices([], 'GPU')
import tensorflow_hub as hub
import tensorflow_text as text
from lyricsgenius import Genius
token = "# genius api key"

genius = Genius(token)
bert_model = tf.keras.models.load_model("# location of fine tuned bert model")


