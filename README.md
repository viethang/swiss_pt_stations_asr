The goal of this project is to train an automatic speech recognition (ASR) model to recognize the name of all Swiss public transport stations. This model then can be used to enhance public transport planner applications with voice input support.
To obtain this, we will fine-tune an open Speech-to-text model with audio files pronouncing the name of public transport stations.

1. PT stations are extracted from CFF's GTFS https://opendata.swiss/de/dataset/timetable-2024-gtfs2020
2. The base ASR model used is OpenAI's pretrained model whisper-large-v3 https://huggingface.co/openai/whisper-large-v3
3. The main difficulty is to have audio files for the fine-tuning. Our strategy is to generate them automatically using a Text-To-Speech (TTS) model. The TTS that we use is Meta's Massively Multilingual Speech TTS fined-tune for each of 3 Swiss official language: French, German and Italian.


In the fist phase we will try with stations in the French speaking region first.
1. Extract the stations from CFF's GTFS for Lausanne
2. Generate the audio files pronouncing the name of these stations
  - Before generating the audio file, we need to transform a bit the station names so that the generated speeches correspond to how the places are pronounced in Switzerland. E.g.: many places finish with "ens", but should be pronounced as "an". Also, we need also to handle abreviation like St-, rte,...
3. Tune ASR model with these files
