
def separate_speaker(speaker_npz_obj):
    all_speaker = sorted(list(set(map(str, speaker_npz_obj.values()))))
    all_keys = sorted(list(speaker_npz_obj.keys()))
    speaker_individual_keys = [
        [
            key
        for key in all_keys if speaker_npz_obj[key] == speaker
        ]
    for speaker in all_speaker
    ]
    return all_speaker, speaker_individual_keys

def get_separated_values(npz_obj, speaker_individual_keys):
    speaker_individual_datas = [
        [
            npz_obj[key]
        for key in keylist
        ]
    for keylist in speaker_individual_keys
    ]
    return speaker_individual_datas
