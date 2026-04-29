# this is my code for set 2 of 166 :D
# trying to follow the prompt step by step and save every file with the exact required name

import numpy as np
import ps_lib
from PIL import Image


# Q1 work

def cross_corr(image, kernel):
    # just made kernel float so the multiply &  sum math is clean
    k = np.float32(kernel)

    # prompt says kernel height & width are odd
    k_h = k.shape[0]
    k_w = k.shape[1]

    # since kernel is odd, half kernel size is the padding amount
    # this is just what keeps the final output the same size as the original image
    u_pad = k_h // 2
    v_pad = k_w // 2

    # use padding from the set library
    padded = ps_lib.pad(image, u_pad, v_pad)

    # output same shape as input image
    output = np.zeros_like(image, dtype=np.float32)

    # manually loop through every px
    for u in range(image.shape[0]):
        for v in range(image.shape[1]):

            # take the image patch under the kernel
            patch = padded[u:u + k_h, v:v + k_w]

            # cross correlation multiply patch & kernel, then sum
            # no flipping the kernel because that would be convolution
            output[u, v] = np.sum(patch * k[:, :, None], axis=(0, 1))

    return output


# read apple & the provided kernels
apple = ps_lib.read_image("apple.png")
gker = np.load("gaussian-kernel.npy")
sv = np.load("sobel-kernel-vertical.npy")
sh = np.load("sobel-kernel-horizontal.npy")

# cross-correlate apple with Gaussian & Sobel kernels
ps_lib.write_image("q1-gaussian.png", cross_corr(apple, gker))
ps_lib.write_image("q1-sobel-vertical.png", cross_corr(apple, sv))
ps_lib.write_image("q1-sobel-horizontal.png", cross_corr(apple, sh))


# Q2 work

k = gker

def gauss(img):
    # start Gaussian pyramid with og image
    pyr = [img.astype(np.float32)]

    # keep making smaller images until one dimension becomes less than the k size
    while pyr[-1].shape[0] >= k.shape[0] and pyr[-1].shape[1] >= k.shape[1]:

        # blur before downsampling, so using Gaussian k
        blur = cross_corr(pyr[-1], k)

        # downsample by factor of 2 
        small = ps_lib.resize(
            blur,
            (pyr[-1].shape[0] // 2, pyr[-1].shape[1] // 2)
        ).astype(np.float32)

        pyr.append(small)

    # everse to smallest first
    return np.array(pyr[::-1], dtype=object)


def lap(img):
    # first make Gaussian pyramid
    gp = gauss(img)

    # smallest level stays as base
    lp = [gp[0].astype(np.float32)]

    # Laplacian level is current image minus smaller level resized back up
    # this gives the detail lost between levels
    for i in range(1, len(gp)):
        up = ps_lib.resize(gp[i - 1], gp[i].shape[:2])
        lp.append((gp[i] - up).astype(np.float32))

    return np.array(lp, dtype=object)


# make required pyramids
apple_p = lap(ps_lib.read_image("apple.png"))
orange_p = lap(ps_lib.read_image("orange.png"))
mask_p = gauss(ps_lib.read_image("mask.png"))

# save pyramid arrays
np.save("q2-apple.npy", apple_p)
np.save("q2-orange.npy", orange_p)
np.save("q2-mask.npy", mask_p)

# render each pyramid level with its own range
for name, pyr in [
    ("q2-apple", apple_p),
    ("q2-orange", orange_p),
    ("q2-mask", mask_p)
]:
    for i in range(len(pyr)):
        img = pyr[i]

        # use min & max for this level so it is visible
        lo = img.min()
        hi = img.max()

        img = (img - lo) / (hi - lo)

        # save the files
        ps_lib.write_image(name + "-" + str(i + 1) + ".png", img)


# Q3 work

def blend(a, b, m):
    out = []

    for i in range(len(a)):
        # mask value near 1 means more of 1st img
        # mask value mear 0 means more of 2nd img
        mixed = m[i] * a[i] + (1 - m[i]) * b[i]
        out.append(mixed.astype(np.float32))

    return out


def rebuild(p):
    # start at smallest level
    img = p[0]

    # resize up & add detail back each time
    for i in range(1, len(p)):
        img = ps_lib.resize(img, p[i].shape[:2]) + p[i]

    return img


# just load Q2 pyramids
apple_p = np.load("q2-apple.npy", allow_pickle=True)
orange_p = np.load("q2-orange.npy", allow_pickle=True)
mask_p = np.load("q2-mask.npy", allow_pickle=True)

# blend apple & orange
img = rebuild(blend(apple_p, orange_p, mask_p))
ps_lib.write_image("q3-apple-and-orange.png", img)


# repeat on my own image, found 2 landscapes online to use for this
land1 = ps_lib.read_image("landscape-1.png")
land2 = ps_lib.resize(ps_lib.read_image("landscape-2.png"), land1.shape[:2])

# using same mask image, resizing it to match my images
# rotated it so the blend direction is different, so we have a diff kind of mask
land_m = ps_lib.read_image("mask.png")[:, :, :3]
land_m = np.rot90(land_m)
land_m = ps_lib.resize(land_m, land1.shape[:2])

img = rebuild(blend(lap(land1), lap(land2), gauss(land_m)))
ps_lib.write_image("q3-new-images.png", img)


# Q4 work

for i in range(1, 5):
    # my first had an exposure of 5, second 10, third 20, and fourth the max 33
    #didn't adjust the gain, bc it said we could do either or/and
    # load q4-shot-i.npz
    d = np.load("q4-shot-" + str(i) + ".npz")

    # img array from capture script is stored in "data"
    img = d["data"].astype(np.float32)

    # reasonable dynamic range for this shot
    img = (img - img.min()) / (img.max() - img.min())

    # save rendered
    ps_lib.write_image("q4-shot-" + str(i) + ".png", img)


# Q5 
#this question killed me bro

def lum(img):
    # one brightness value per px
    # using this instead of separate RGB channels keeps the merge from making weird colors
    #kept coming out weird so had to adjust the brightness
    return 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]


def make_comp(shots):
    # shot 1 is the reference brightness scale
    gains = [1.0]

    s1y = lum(shots[0])

    # compute gain for each other shot relative to shot 1
    for i in range(1, len(shots)):
        sy = lum(shots[i])

        # nonsaturated pxs in both images
        # also avoiding almost-black pxs b/c division gets unstable there
        good = (
            (s1y > 0.01) & (s1y < 0.95) &
            (sy > 0.01) & (sy < 0.95)
        )

        # median ratio gives relative gain factor
        gains.append(np.median(sy[good] / s1y[good]))

    # merge all shots after scaling them to shot 1 brightness scale
    num = np.zeros_like(shots[0], dtype=np.float32)
    den = np.zeros_like(shots[0], dtype=np.float32)

    for i in range(len(shots)):
        # divide by gain to put this shot back on shot 1 scale
        scaled = shots[i] / gains[i]

        # only use nonsaturated pixels when merging
        good = lum(shots[i]) < 0.95

        num += scaled * good[:, :, None]
        den += good[:, :, None]

    comp = num / np.maximum(den, 1e-6)

    return comp.astype(np.float32)


def render(comp, word):
    # five brightness levels  I picked using quantiles
    qs = [
        (0.001, 0.999),
        (0.005, 0.995),
        (0.01, 0.99),
        (0.02, 0.98),
        (0.05, 0.95)
    ]

    vals = comp[np.isfinite(comp)]

    for i in range(5):
        lo = np.quantile(vals, qs[i][0])
        hi = np.quantile(vals, qs[i][1])

        img = (comp - lo) / (hi - lo)
        img = np.clip(img, 0, 1).astype(np.float32)

        ps_lib.write_image("q5-" + word + "-rendering-" + str(i + 1) + ".png", img)


# canyon HDR
canyon = [ps_lib.read_image("canyon-shot-" + str(i) + ".png") for i in range(1, 5)]
canyon_comp = make_comp(canyon)

np.save("q5-canyon-composite.npy", canyon_comp.astype(np.float32))
render(canyon_comp, "canyon")


# captured HDR from Q4
captured = []

for i in range(1, 5):
    d = np.load("q4-shot-" + str(i) + ".npz")

    img = d["data"].astype(np.float32)

    # put captured image on 0 -1 scale before HDR merge
    img = img / img.max()

    captured.append(img)

captured_comp = make_comp(captured)

np.save("q5-captured-composite.npy", captured_comp.astype(np.float32))
render(captured_comp, "captured")


# Q6 work

def gaussian(x, s):
    # Gaussian function G
    return np.exp(-(x ** 2) / (2 * s ** 2))


def bilateral_filter(img, sd, sr):
    # sd is sigma_domain
    # sr is sigma_range

    # radius based on domain sigma
    r = int(3 * sd)

    # pad so edge pixels still have neighbors
    pad = ps_lib.pad(img, r, r)

    # output same size as input
    out = np.zeros_like(img, dtype=np.float32)

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            ctr = pad[y + r, x + r]

            # for Q6 ctr is RGB, for Q7 it can be scalar log luminance
            total = np.zeros_like(ctr, dtype=np.float32)

            # normalization factor k
            norm = 0.0

            # sum over nearby pixels
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    nb = pad[y + r + dy, x + r + dx]

                    # domain Gaussian uses pixel distance
                    wd = gaussian(np.sqrt(dy**2 + dx**2), sd)

                    # range Gaussian uses Euclidean RGB distance for Q6
                    wr = gaussian(np.linalg.norm(ctr - nb), sr)

                    w = wd * wr

                    total += w * nb
                    norm += w

            out[y, x] = total / norm

    return out


noisy = ps_lib.read_image("noisy-image.png")
filt = bilateral_filter(noisy, 1, 0.3)

ps_lib.write_image("q6-filtered-image.png", filt)


# my work forQ7

def tone_map(img):
    # normalize HDR values so chrominance split does not go crazy
    img = img / img.max()

    # luminance is Y channel plus small constant
    y = lum(img) + 1e-6

    # chrominance is original image divided by luminance
    c = img / y[:, :, None]

    # log10 luminance
    log_y = np.log10(y)

    # base image from log10 luminance using bilateral filtering
    # using smaller version so this does not run forever
    small = ps_lib.resize(
        log_y,
        (log_y.shape[0] // 8, log_y.shape[1] // 8)
    )

    base_small = bilateral_filter(small, small.shape[1] / 50, 0.4)

    # resize base back to original size
    base = ps_lib.resize(base_small, log_y.shape)

    # detail is the part left after subtracting smooth base
    detail = log_y - base

    # compute alpha_base for R_target = 100
    base_lin = 10 ** base
    hi = base_lin.max()
    lo = base_lin[base_lin > 0].min()

    a = np.log10(100) / np.log10(hi / lo)

    # beta makes highest base value map to 1 in linear scale
    beta = -a * base.max()

    # merge transformed base and amplified details in log domain
    out_log = a * base + 3 * detail + beta

    # back to linear scale
    out_y = 10 ** out_log

    # merge processed luminance with chrominance
    out = c * out_y[:, :, None]

    return np.clip(out, 0, 1).astype(np.float32)


ps_lib.write_image("q7-canyon.png", tone_map(np.load("q5-canyon-composite.npy")))
ps_lib.write_image("q7-captured.png", tone_map(np.load("q5-captured-composite.npy")))

#q5 and q7 gave me such a headache, I just gave up on making it smoother :(