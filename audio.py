def main():
    import argparse

    import numpy
    try:
        import cupy
        available = True
    except ImportError:
        available = False
    import librosa

    from PCA import RPCA

    # use CPU or GPU
    parser = argparse.ArgumentParser()
    parser.add_argument('file', metavar='file',
                        help='file name')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU1 ID (negative value indicates CPU)')
    args = parser.parse_args()

    if available and args.gpu >= 0:
        cupy.cuda.Device(args.gpu).use()
        xp = cupy
        available = True
    else:
        xp = numpy
        available = False

    # load image
    (accmp, vocal), _ = librosa.load(args.file, 16000, mono=False)
    mixed = 0.5 * (vocal + accmp)
    M = librosa.stft(mixed, n_fft=1024, hop_length=512)
    phase = numpy.angle(M)
    M = xp.abs(xp.expand_dims(xp.asarray(M), 2))
    print('shape: {}, min: {}, max: {}'.format(M.shape, M.min(), M.max()))

    # set parameters
    lmd = 0.03
    rho = 0.8
    max_iter = 1000
    stopcri = 0.05

    # optimize
    rpca = RPCA(xp, lmd, rho, max_iter, stopcri)
    L, S = rpca(M, echo_iter=5)

    M = xp.maximum(M, 0)
    L = xp.maximum(L, 0)
    S = xp.maximum(S, 0)

    # to CPU
    if available:
        M = xp.asnumpy(M)
        L = xp.asnumpy(L)
        S = xp.asnumpy(S)

    # save audio
    L = numpy.squeeze(L) * numpy.exp(1j*phase)
    S = numpy.squeeze(S) * numpy.exp(1j*phase)
    M_recon = numpy.squeeze(L + S) * numpy.exp(1j*phase)
    M = numpy.squeeze(M) * numpy.exp(1j*phase)

    librosa.output.write_wav(
        'L.wav', librosa.istft(L, hop_length=512), 16000)
    librosa.output.write_wav(
        'S.wav', librosa.istft(S, hop_length=512), 16000)
    librosa.output.write_wav(
        'M_recon.wav', librosa.istft(M_recon, hop_length=512),
        16000)
    librosa.output.write_wav(
        'M.wav', librosa.istft(M, hop_length=512), 16000)


if __name__ == '__main__':
    main()
