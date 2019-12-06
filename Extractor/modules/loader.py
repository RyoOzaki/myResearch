def load_wav(f):
    import scipy.io.wavfile as wav
    (rate,data) = wav.read(f)
    return (rate, data)

def load_sph(f):
    from sphfile import SPHFile
    sph = SPHFile(f)
    return (sph.format['sample_rate'], sph.content)
