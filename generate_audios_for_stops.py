import scipy.io.wavfile as wavfile
from transformers import VitsModel, AutoTokenizer
import torch
import pandas as pd
import re

model = VitsModel.from_pretrained("facebook/mms-tts-fra")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-fra")


def transform_text(text):
  """ 
  - Station name like 'Lausanne, Vennes' is separated into 'Lausanne' and 'Vennes'.
  - Abreviations are also replaced by original texts and special characters are removed.  
  """
  def replace_st_with_saint(input_str):
    # Use re.sub to replace groups of characters 'st' with 'saint'
    str1 = re.sub(r'\bst\b', 'saint', input_str, flags=re.IGNORECASE)
    str2 = re.sub(r'\bste\b', 'sainte', str1, flags=re.IGNORECASE)
    return str2

  output = text.lower()
  replace_rules = [
      ('(', ''),
      (')', ''),
      ('pl.', 'place'),
      ('rte ', 'route '),
  ]
  for x, y in replace_rules:
    output = output.replace(x, y)
  output = replace_st_with_saint(output)
  return output.strip()


def prepare_text_inputs(stop_file):
  """
  Prepare the list of texts as input for the audio generation.
  - Input: a csv file containing station names in `stop_name` column
  - Return a dictionary e.g. 
      {"route_de_genève": "route de genève"}
  """
  df = pd.read_csv(stop_file, delimiter=',')
  texts_dict = {}
  for stop in list(df["stop_name"]):
    splitted_texts = [transform_text(x) for x in stop.split(',')]
    for text in splitted_texts:
      key = "_".join(text.split())
      if not texts_dict.get(key):
        texts_dict[key] = text

  return texts_dict


def generate_audio_for_text(text, output_file):
  with torch.no_grad():
    inputs = tokenizer(text, return_tensors="pt")

    output = model(**inputs).waveform
    wavfile.write(
        output_file,
        rate=model.config.sampling_rate, data=output.T.numpy()
    )


def generate_audios(stop_file, templates):
  """
  @param templates: E.g.  
    {
      "tpl1": "J'habite à {}. Mais je viens de Provence, une région ensolleillé de la France."
    }
  """
  from pathlib import Path

  texts_dict = prepare_text_inputs(stop_file)

  for (cat, tpl) in templates.items():
    # create output directory if not existing
    Path(f"data/audios/{cat}").mkdir(parents=True, exist_ok=True)

    with open(f"data/audios/{cat}/index.csv", 'w') as index_file:
      index_file.write("stop_name;text\n")       # write header

      for key, text in texts_dict.items():
        augmented_text = tpl.format(text)
        output_file = f"data/audios/{cat}/{key}.wav"
        try:
          generate_audio_for_text(augmented_text, output_file)
          index_file.write(f"{key};{text}\n")
        except Exception as err:
          print("Failed to generate audio for", cat, text)
          print(err)


if __name__ == "__main__":

  templates = {
      "tpl1": "J'habite à {}. Mais je viens de Provence, une région ensolleillé de la France."
  }
  generate_audios(stop_file="data/lausanne_stops_small.csv",
                  templates=templates)
