import numpy as np
from pathlib import Path
from SpeechFeatureExtraction.util.loader import SPHLoader
from SpeechFeatureExtraction.speech_feature_extraction import Extractor
from tqdm import tqdm

source_dir = Path("source")

extractor = Extractor(SPHLoader)
extractor.load_phoneme_list(source_dir.glob("**/*.PHN"), sp={0: "h#"})
extractor.load_word_list(source_dir.glob("**/*.WRD"))

print("Phoneme N: {}".format(len(extractor.phoneme_list)))
print("Word N: {}".format(len(extractor.word_list)))

static = {}
delta = {}
delta_delta = {}
phoneme_label = {}
word_label = {}
speaker_id = {}
i = 0
for speaker_dir in tqdm(list(source_dir.glob("*"))):
    speaker_name = speaker_dir.name
    data_names = [f.stem for f in speaker_dir.glob("*.WAV")]
    for dname in data_names:
        wavf = speaker_dir / (dname+".WAV")
        phnf = speaker_dir / (dname+".PHN")
        wrdf = speaker_dir / (dname+".WRD")
        (s, d, dd), p, w = extractor.load(wavf, phnf, wrdf)
        dict_key = speaker_name + "_" + dname
        static[dict_key] = s
        delta[dict_key] = d
        delta_delta[dict_key] = dd
        phoneme_label[dict_key] = p
        word_label[dict_key] = p
        speaker_id[dict_key] = i
    i += 1

print("Skeaper N: {}".format(i))
print("Sentence N: {}".format(len(speaker_id.keys())))

np.savez("feature/static.npz", **static)
np.savez("feature/delta.npz", **delta)
np.savez("feature/delta_delta.npz", **delta_delta)
np.savez("feature/phoneme_label.npz", **phoneme_label)
np.savez("feature/word_label.npz", **word_label)
np.savez("feature/speaker_id.npz", **speaker_id)
