# cupy-RPCA
A cupy(numpy) implementation of Robust PCA( https://statweb.stanford.edu/~candes/papers/RobustPCA.pdf ).
I referenced this( https://www.slideshare.net/skanata2/ss-61220354 ).

# Requirements
- numpy
- cupy(if you use GPU)
- Pillow(if you try image example)
- librosa(if you try audio example)

# Usage
SVD on GPU is too slow and RPCA uses SVD. So you should run without GPU. If you don't use GPU, don't set -g option.
## Image example
```
python image.py <path to image>
python image.py <path to image> -g <GPU ID>
```

## Audio example
```
python audio.py <path to audio>
python audio.py <path to audio> -g <GPU ID>
```
