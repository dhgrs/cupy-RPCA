def main():
    import argparse

    import numpy
    try:
        import cupy
        available = True
    except ImportError:
        available = False
    from PIL import Image

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
    img = Image.open(args.file)
    M = xp.asarray(img.convert('RGB')) / 255
    print('shape: {}, min: {}, max: {}'.format(M.shape, M.min(), M.max()))

    # set parameters
    lmd = 0.03
    rho = 0.8
    max_iter = 1000
    stopcri = 0.001

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

    # save image
    L *= 255
    L = numpy.clip(L, 0, 255)

    S *= 255
    S = numpy.clip(S, 0, 255)

    M_recon = L + S
    M_recon = numpy.clip(M_recon, 0, 255)

    M *= 255
    M = numpy.clip(M, 0, 255)

    Image.fromarray(numpy.uint8(L)).save('L.png')
    Image.fromarray(numpy.uint8(S)).save('S.png')
    Image.fromarray(numpy.uint8(M_recon)).save('M_recon.png')
    Image.fromarray(numpy.uint8(M)).save('M.png')


if __name__ == '__main__':
    main()
