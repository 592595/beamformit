import numpy as np
import math
from scipy.fftpack import fft, ifft
import matplotlib.pylab as plt
import soundfile as sf

infilenames_cell = ['sample11/M05_443C0207_PED.CH1.wav',
                    'sample11/M05_443C0207_PED.CH3.wav',
                    'sample11/M05_443C0207_PED.CH4.wav',
                    'sample11/M05_443C0207_PED.CH5.wav',
                    'sample11/M05_443C0207_PED.CH6.wav']

outfilename = 'enhanced.wav'


def test(infilenames_cell, outfilename):
    [x, sr, nmic, npair, nsample] = get_x(infilenames_cell)
    plt.figure()
    plt.plot(x[0, :])
    plt.show()

    # make hamming window
    nwin = 16000
    hamming_bfit(nwin)
    win = hamming_bfit(nwin)
    plt.figure()
    plt.plot(win)
    plt.show()
    print(win[1:10])

    # calculate avg_ccorr
    npiece = 200
    nfft = 32768
    nbest = 2
    nmask = 5
    calcuate_avg_ccorr(x, nsample, nmic, npiece, win, nwin, nfft, nbest, nmask)
    # ref_mic = calcuate_avg_ccorr(x, nsample, nmic, npiece, win, nwin, nfft, nbest, nmask)

    # # calculating scaling factor
    # nsegment = 10
    # overall_weight = calculate_scaling_factor(x, sr, nsample, nmic, nsegment)
    #
    # # compute total number of delays
    # nwin = 8000
    # nshift = nwin / 2
    # nframe = math.floor((nsample - nwin) / (nshift))
    #
    # # recreating hamming window
    # win = hamming_bfit(nwin)
    # plt.figure()
    # plt.plot(win)
    # print(16000 * 30 / 1000)
    #
    # # get pair2mic table
    # pair2mic = get_pair2mic(nmic, npair)
    #
    # # compute TDOA
    # nbest = 4
    # nfft = 16384
    # [gcc_nbest, tdoa_nbest] = compute_tdoa(x, npair, ref_mic, pair2mic, nframe, win, nwin, nshift, nfft, nbest, nmask)
    # print(np.squeeze(gcc_nbest[:, :, 1]))
    # print(np.squeeze(gcc_nbest[:, :, 2]))
    #
    # print(np.squeeze(tdoa_nbest[:, :, 1]))
    # print(np.squeeze(tdoa_nbest[:, :, 2]))
    #
    # # find noise threshold
    # threshold = get_noise_threshold(gcc_nbest, nmic, ref_mic, nframe)
    # print(threshold)
    #
    # # noise filtering
    # [gcc_nbest, tdoa_nbest, noise_filter] = get_noise_filter(gcc_nbest, tdoa_nbest, npair, ref_mic, nframe, threshold);
    # print(gcc_nbest[:, :, 1])
    # print(tdoa_nbest[:, :, 1])
    #
    # print(gcc_nbest[:, :, 2])
    # print(tdoa_nbest[:, :, 2])
    #
    # # single channel viterbi
    # [emission1, transition1] = prep_ch_indiv_viterbi(gcc_nbest, tdoa_nbest, npair, nframe, nbest)
    #
    # bestpath1 = np.ones(npair, nframe)
    #
    # bestpath1 = decode_ch_indiv_viterbi(bestpath1, emission1, transition1, npair, nframe, nbest)
    #
    # best2path = decode_ch_indiv_viterbi_best2(bestpath1, gcc_nbest, transition1, npair, nframe, nbest)
    #
    # print(bestpath1[:, :, 1])
    #
    # # multi channel viterbi
    # nbest2 = 2
    # nstate = nbest2 ^ npair
    # g = get_states(nstate, nmic, npair, nbest2)
    #
    # [emission2, transition2] = prep_global_viterbi(best2path, gcc_nbest, tdoa_nbest, g, npair, nbest, nframe, nstate)
    # besttdoa = decode_global_viterbi(best2path, emission2, transition2, tdoa_nbest, g, npair, nframe, nstate)
    #
    # print(g)
    # print(emission2(2, emission2[-1]))
    # print(transition2[2, :, 1])
    # print(besttdoa)
    # # plt.figure()///////////////////
    #
    # # compute local xcorr
    # mic2refpair = get_mic2refpair(pair2mic, ref_mic, nmic, npair)
    # localxcorr = compute_local_xcorr(besttdoa, x, nsample, nmic, npair, nframe, ref_mic, mic2refpair)
    #
    # # compute sum weight
    # alpha = 0.05
    # out_weight = compute_out_weight(localxcorr, nframe, nmic, noise_filter, ref_mic, mic2refpair, alpha)
    #
    # # Channel sum
    # out_x = channel_sum(x, nsample, nframe, nmic, ref_mic, mic2refpair, nwin, nshift, besttdoa, out_weight,
    #                     overall_weight)
    #
    # wavfile.read(outfilename, out_x, sr)


def get_x(infilenames_cell):
    nmic = np.size(infilenames_cell, 0)
    npair = nmic
    # x_ch1,sr = sf.read(infilenames_cell[0])
    # nsample = np.size(x_ch1, 0)
    # x = np.zeros((nmic, nsample))
    # x[0, :] = x_ch1
    # for m in range(1, nmic):
    #     x[m, :] = sf.read(infilenames_cell[m])
    x0, sr = sf.read(infilenames_cell[0])
    x1, sr = sf.read(infilenames_cell[1])
    x2, sr = sf.read(infilenames_cell[2])
    x3, sr = sf.read(infilenames_cell[3])
    x4, sr = sf.read(infilenames_cell[4])
    x = [x0, x1, x2, x3, x4]
    x = np.array(x)
    tuple(x)
    nsample = np.size(x0, 0)
    return [x, sr, nmic, npair, nsample]


def maxk(list, k, nmask):
    candi_list = np.zeros(len(list[:]))

    for i in range(1, (len(list[:]) - 1)):
        if list[i - 1] < list[i] and list[i + 1] < list[i]:
            candi_list[i] = list[i]

    max_val = np.zeros(k)
    idx = np.zeros(k)
    for i in range(0, k):
        [max_val[i], idx[i]] = max(candi_list)
        st = max(idx[i] - nmask + 1, 1)
        ed = min(len(candi_list[:]), idx[i] + nmask - 1)
        candi_list[st:ed] = 0
    return [max_val, idx]


def hamming_bfit(nwin):
    win = np.zeros(nwin)
    for i in range(0, nwin):
        win[i] = 0.54 - 0.46 * math.cos(6.283185307 * (i - 1) / (nwin - 1))
    return win


def calcuate_avg_ccorr(x, nsample, nmic, npiece, win, nwin, nfft, nbest, nmask):
    scroll = math.floor(nsample / (npiece + 2))
    avg_ccorr = np.zeros((nmic, nmic))

    for i in range(0, npiece):
        st = i * scroll
        ed = st + nwin
        if st + nfft / 2 >= nsample:
            break
        for m1 in range(0, nmic - 1):
            avg_ccorr[m1, m1] = 0
            for m2 in range(m1, nmic):
                stft1 = fft(np.append(np.multiply(x[m1, st:ed], win), np.zeros((nfft - nwin))))
                stft2 = fft(np.append(np.multiply(x[m2, st:ed], win), np.zeros((nfft - nwin))))
                numerator = np.multiply(stft1, np.conj(stft2))
                ccorr = (ifft(numerator / (abs(numerator)))).real
                ccorr = np.append(ccorr[- 479:], ccorr[1:480])

                avg_ccorr[m1, m2] = avg_ccorr[m1, m2] + sum(maxk(ccorr, nbest, nmask))
                avg_ccorr[m2, m1] = avg_ccorr[m1, m2]
    avg_ccorr = avg_ccorr / (nbest * npiece)
    [dummy, ref_mic] = max(sum(avg_ccorr))
    print(ref_mic)
    # return ref_mic


def calculate_scaling_factor(x, sr, nsample, nmic, nsegment):
    max_val = np.zeros(nmic, 1)
    if nsample <= 10 * sr:
        for m in range(0, nmic - 1):
            max_val[m] = max(abs(x[m, :]))
    else:
        if nsample < 100 * sr:
            nsegment = math.floor(np.size(x, 1) / 160000)
        scroll = math.floor(np.size(x, 1) / nsegment)
        max_val_candidate = np.zeros(nmic, nsegment)

        for s in range(-1, nsegment - 2):
            st = s * scroll + 1
            ed = st + 160000 - 1
            for m in range(0, nmic - 1):
                max_val_candidate[m, s + 1] = max(abs(x[m, st:ed]))
        for m in range(0, nmic - 1):
            sorted = sorted(max_val_candidate[m, :], 'ascend')
            if len(sorted[:]) > 2:
                max_val[m] = sorted(math.floor(sorted[-1] / 2) + 1)
            else:
                max_val[m] = sorted

    overall_weight = (0.3 * nmic) / sum(max_val)
    return overall_weight


def get_pair2mic(nmic, npair):
    pair2mic = np.zeros(nmic, npair)
    for m in range(0, nmic - 1):
        p = 1
        for i in range(0, nmic - 1):
            pair2mic[m, p] = i
            p = p + 1;
    return pair2mic


def get_mic2refpair(pair2mic, ref_mic, nmic, npair):
    mic2refpair = np.zeros(nmic, 1)
    mic2refpair[ref_mic] = 0
    for p in range(0, npair - 1):
        m = pair2mic(ref_mic, p)
        mic2refpair[m] = p
    return mic2refpair

    # problem


def compute_tdoa(x, npair, ref_mic, pair2mic, nframe, win, nwin, nshift, nfft, nbest, nmask):
    gcc_nbest = np.zeros(npair, nframe, nbest)
    tdoa_nbest = np.zeros(npair, nframe, nbest)

    for t in range(0, (nframe - 1)):
        st = (t - 1) * nshift + 1
        ed = st + nwin - 1
        for p in range(0, npair - 1):
            m = pair2mic(ref_mic, p)
            stft_ref = fft([np.dot(x[ref_mic, st:ed], win), np.zeros(1, nfft - nwin)])
            stft_m = fft([np.dot(x[m, st:ed], win), np.zeros(1, nfft - nwin)])
            numerator = np.dot(stft_m, math.conj(stft_ref))
            # gcc = (ifft(numerator/(eps + abs(numerator)))).real
            gcc = (ifft(numerator / (abs(numerator)))).real
            gcc = [gcc[gcc[-1] - 479:gcc[-1]], gcc[1: 480]]
            [gcc_nbest[p, t, :], tdoa_nbest[p, t, :]] = maxk(gcc, nbest, nmask)
            tdoa_nbest[p, t, :] = tdoa_nbest[p, t, :] - (481)
    return [gcc_nbest, tdoa_nbest]


def get_noise_threshold(gcc_nbest, nmic, ref_mic, nframe):
    th_idx = math.floor((0.1 * nframe)) + 1

    sorted = sorted(sum(gcc_nbest[:, :, 1], 1) - sum(gcc_nbest[ref_mic, :, 1], 1), 'ascend')

    threshold = sorted(th_idx) / (nmic - 1)
    return threshold


def get_noise_filter(gcc_nbest, tdoa_nbest, npair, ref_mic, nframe, threshold):
    noise_filter = np.zeros(npair, nframe)

    for p in range(0, npair - 1):
        for t in range(0, nframe - 1):
            if gcc_nbest(p, t, 1) < threshold:
                noise_filter[p, t] = 1
                if t == 1:
                    gcc_nbest[p, t, :] = 0
                    gcc_nbest[p, t, 1] = 1
                    tdoa_nbest[p, t, :] = 480
                    tdoa_nbest[p, t, 1] = 0
                else:
                    tdoa_nbest[p, t, :] = tdoa_nbest[p, t - 1, :]

            if p == ref_mic:
                gcc_nbest[p, t, :] = 0
                gcc_nbest[p, t, 1] = 1
                tdoa_nbest[p, t, :] = 0

    return [gcc_nbest, tdoa_nbest, noise_filter]


def prep_ch_indiv_viterbi(gcc_nbest, tdoa_nbest, npair, nframe, nbest):
    emission1 = np.zeros(npair, nframe, nbest)
    diff1 = np.zeros(npair, nframe, nbest, nbest)
    transition1 = np.zeros(npair, nframe, nbest, nbest)

    for p in range(0, npair - 1):
        for t in range(0, nframe - 1):
            for n in range(0, nbest - 1):
                if gcc_nbest(p, t, n) > 0:
                    emission1[p, t, n] = math.log10(gcc_nbest(p, t, n) / sum(gcc_nbest[p, t, :]))
                else:
                    emission1[p, t, n] = -1000

    for p in range(0, npair - 1):
        for t in range(1, nframe - 1):
            for n in range(0, nbest - 1):
                for nprev in range(0, nbest - 1):
                    diff1[p, t, n, nprev] = abs(tdoa_nbest(p, t, n) - tdoa_nbest(p, t - 1, nprev))

    for p in range(0, npair):
        for t in range(1, nframe - 1):
            for n in range(0, nbest - 1):
                for nprev in range(0, nbest - 1):
                    maxdiff1 = max(diff1[p, t, :])
                    nume = 1 + maxdiff1 - diff1[p, t, n, nprev]
                    deno = (2 + maxdiff1)
                    transition1[p, t, n, nprev] = math.log10(nume / deno)

    return [emission1, transition1]


def decode_ch_indiv_viterbi(bestpath1, emission1, transition1, npair, nframe, nbest):
    dC = np.ones(npair, nframe, nbest) * -1000
    tC = np.ones(npair, nframe, nbest)
    R = np.ones(npair, nframe)
    F = np.ones(npair, nframe)

    forwardTrans = np.zeros(npair, nframe, nbest)
    selfLoopTrans = np.zeros(npair, nframe, nbest)

    for p in range(0, npair - 1):
        for n in range(0, nbest - 1):
            dC[p, 1, n] = emission1(p, 1, n)

    for p in range(0, npair - 1):
        for t in range(1, nframe):
            for n in range(0, nbest - 1):
                best_n_prev = R[p, t - 1]
                forwardTrans[p, t, n] = dC(p, t - 1, best_n_prev) + 25 * transition1(p, t, n, best_n_prev)
                selfLoopTrans[p, t, n] = dC(p, t - 1, n) + 25 * transition1(p, t, n, n)

                if selfLoopTrans[p, t, n] >= forwardTrans[p, t, n]:
                    dC[p, t, n] = selfLoopTrans[p, t, n] + emission1[p, t, n]
                    tC[p, t, n] = tC[p, t - 1, n]
                else:
                    dC[p, t, n] = forwardTrans[p, t, n] + emission1[p, t, n]
                    tC[p, t, n] = t
            [dummy, R[p, t]] = max(dC[p, t, :])
            F[p, t] = tC[p, t, R[p, t]]

        st = F[p, F[-1]]
        bestpath1[p, st: F[-1]] = R[p, F[-1]]

        while st > 1:
            ed = st - 1
            st = F[p, ed]
            bestpath1[p, st: ed] = R[p, ed]

    return bestpath1


def decode_ch_indiv_viterbi_best2(bestpath1, gcc_nbest, transition1, npair, nframe, nbest):
    best2path = np.zeros(npair, nframe, 2)
    emission1 = np.zeros(npair, nframe, nbest)

    for p in range(0, npair - 1):
        for t in range(0, nframe - 1):
            best1 = bestpath1(p, t)
            gcc_nbest[p, t, best1] = 0

    for p in range(0, npair - 1):
        for t in range(0, nframe - 1):
            for n in range(0, nbest - 1):
                if gcc_nbest(p, t, n) > 0:
                    emission1[p, t, n] = math.log10(gcc_nbest(p, t, n) / sum(gcc_nbest[p, t, :]))
                else:
                    emission1[p, t, n] = -1000

    bestpath2 = decode_ch_indiv_viterbi(bestpath1, emission1, transition1, npair, nframe, nbest)

    for p in range(0, npair - 1):
        for t in range(0, nframe - 1):
            best2path[p, t, 1] = bestpath1(p, t)
            best2path[p, t, 2] = bestpath2(p, t)

    return best2path


def fill_all_comb(ipair, npair, ibest, nbest, tmp_row, table, l):
    tmp_row[ipair] = ibest

    if ipair == npair:
        for j in range(0, npair - 1):
            table[l, j] = tmp_row(j)
        l = l + 1
    else:
        for ibest in range(0, nbest - 1):
            [table, l] = fill_all_comb(ipair + 1, npair, ibest, nbest, tmp_row, table, l)

    return [table, l]


def get_states(nstate, nmic, npair, nbest2):
    g = np.zeros(nstate, npair)
    tmp_row = np.zeros(nmic, 1)
    l = 1
    for ibest in range(0, nbest2 - 1):
        [g, l] = fill_all_comb(1, npair, ibest, nbest2, tmp_row, g, l)
    return g


def prep_global_viterbi(best2path, gcc_nbest, tdoa_nbest, g, npair, nbest, nframe, nstate):
    diff2 = np.zeros(npair, nframe, nbest, nbest)
    emission2 = np.zeros(nframe, nstate)
    transition2 = np.zeros(nframe, nstate, nstate)

    for t in range(0, nframe - 1):
        for l in range(0, nstate - 1):
            for m in range(0, npair - 1):
                ibest = best2path(m, t, g(l, m))

                if gcc_nbest(m, t, ibest) > 0:
                    emission2[t, l] = emission2[t, l] + math.log10(gcc_nbest(m, t, ibest))
                else:
                    emission2[t, l] = emission2[t, l] + -1000

    for t in range(1, nframe - 1):
        maxdiff2 = 0
        for ibest in range(0, nbest - 1):
            for jbest in range(0, nbest - 1):
                for m in range(0, npair - 2):
                    diff2[m, t, ibest, jbest] = abs(tdoa_nbest(m, t, ibest) - tdoa_nbest(m, t - 1, jbest))
                    if maxdiff2 < diff2[m, t, ibest, jbest]:
                        maxdiff2 = diff2[m, t, ibest, jbest]
        diff2[:, t, :, :] = math.log10((1 + maxdiff2 - diff2[:, t, :, :]) / (2 + maxdiff2))
        for l in range(0, nstate - 1):
            for lprev in range(0, nstate - 1):
                for m in range(0, npair - 2):
                    ibest = best2path(m, t, g(l, m))
                    jbest = best2path(m, t - 1, g(lprev, m))
                    transition2[t, l, lprev] = transition2[t, l, lprev] + diff2[m, t, ibest, jbest]

    return [emission2, transition2]


def decode_global_viterbi(best2path, emission2, transition2, tdoa_nbest, g, npair, nframe, nstate):
    besttdoa = np.zeros(npair, nframe)

    dC = np.ones(nframe, nstate) * -1000
    tC = np.ones(nframe, nstate)

    R = np.ones(nframe, 1)
    F = np.ones(nframe, 1)

    forwardTrans = np.zeros(nframe, nstate)
    selfLoopTrans = np.zeros(nframe, nstate)

    for l in range(0, nstate - 1):
        dC[1, l] = emission2(1, l)
    for t in range(1, nframe - 1):
        for l in range(0, nstate - 1):
            best_l_prev = R(t - 1)
            forwardTrans[t, l] = dC(t - 1, best_l_prev) + 25 * transition2(t, l, best_l_prev)
            selfLoopTrans[t, l] = dC(t - 1, l) + 25 * transition2(t, l, l)

            if selfLoopTrans[t, l] >= forwardTrans[t, l]:
                dC[t, l] = selfLoopTrans[t, l] + emission2[t, l]
                tC[t, l] = tC[t - 1, l]
            else:
                dC[t, l] = forwardTrans[t, l] + emission2[t, l]
                tC[t, l] = t

        [dummy, R[t]] = max(dC[t, :])
        F[t] = tC[t, R[t]]
    bestpath2 = np.ones(nframe, 1)
    st = F[F[-1]]
    bestpath2[st: bestpath2[-1]] = R[R[-1]]

    while st > 1:
        ed = st - 1
        st = F[ed]
        bestpath2[st: ed] = R[ed]

    besttdoa = np.zeros(npair, nframe)

    for t in range(0, nframe - 1):
        if bestpath2[t] == 0:
            print('t: %d\n', bestpath2(t))
            print(F)
            print(R)
        for p in range(0, npair - 1):
            l = bestpath2[t]
            ibest = best2path(p, t, g(l, p))
            besttdoa[p, t] = tdoa_nbest(p, t, ibest)

    return besttdoa


def compute_local_xcorr(besttdoa, x, nsample, nmic, npair, nframe, ref_mic, mic2refpair):
    tmp_localxcorr = np.zeros(nmic, nmic, nframe)

    for t in range(0, nframe - 1):
        ref_st = (t - 1) * 4000 + 1
        ref_ed = min(ref_st + 8000 - 1, nsample)
        for m1 in range(0, nmic - 2):
            for m2 in range(m1, nmic - 1):
                if m1 == ref_mic:
                    st1 = ref_st
                    ed1 = ref_ed
                else:
                    p = mic2refpair(m1)
                    st1 = max(1, ref_st + besttdoa(p, t))
                    ed1 = min(nsample, ref_ed + besttdoa(p, t))

                if m2 == ref_mic:
                    st2 = ref_st
                    ed2 = ref_ed
                else:
                    p = mic2refpair(m2)
                    st2 = max(1, ref_st + besttdoa(p, t))
                    ed2 = min(nsample, ref_ed + besttdoa(p, t))

                buf1 = x[m1, st1:ed1]
                buf2 = x[m2, st2:ed2]

                ener1 = sum(buf1[:] ** 2)
                ener2 = sum(buf2[:] ** 2)

                min_ed = min(ed1 - st1, ed2 - st2) + 1
                tmp_localxcorr[m1, m2, t] = sum(np.dot(buf1[1: min_ed], buf2[1: min_ed]) / [ener1 * ener2])

                if tmp_localxcorr[m1, m2, t] < 0:
                    tmp_localxcorr[m1, m2, t] = 0
                tmp_localxcorr[m2, m1, t] = tmp_localxcorr[m1, m2, t]
    localxcorr = np.squeeze(sum(tmp_localxcorr, 1))
    return localxcorr


def compute_out_weight(localxcorr, nframe, nmic, noise_filter, ref_mic, mic2refpair, alpha):
    out_weight = np.ones(nmic, nframe) * (1 / nmic)

    for t in range(0, nframe - 1):
        if sum(localxcorr[:, t]) == 0:
            localxcorr[:, t] = 1 / nmic
        localxcorr[:, t] = localxcorr[:, t] / sum(localxcorr[:, t])
        localxcorr_sum_nonref = 0
        for m in range(0, nmic - 1):
            if m == ref_mic:
                out_weight[m, t] = (1 - alpha) * out_weight(m, max(1, t - 1)) + alpha * localxcorr(m, t)
            else:
                p = mic2refpair(m)
                if noise_filter(p, t) == 0:
                    out_weight[m, t] = (1 - alpha) * out_weight(m, max(1, t - 1)) + alpha * localxcorr(m, t)
                localxcorr_sum_nonref = localxcorr_sum_nonref + localxcorr(m, t);
        if sum(localxcorr[:, t]) == 0:
            out_weight[:, t] = 1
        for m in range(0, nmic - 1):
            if m != ref_mic:
                if (localxcorr(m, t) / localxcorr_sum_nonref) < (1 / (10 * (nmic - 1))):
                    out_weight[m, t] = 0
        out_weight[:, t] = out_weight[:, t] / sum(out_weight[:, t])
    return out_weight

    # problem


def channel_sum(x, nsample, nframe, nmic, ref_mic, mic2refpair, nwin, nshift, besttdoa, out_weight, overall_weight):
    out_x = np.zeros(1, nsample)
    for t in range(0, nframe - 1):
        ref_st = (t - 1) * nshift + 1
        ref_ed = min(ref_st + nwin - 1, nsample)

        for m in range(0, nmic - 1):
            if m == ref_mic:
                st = ref_st
                ed = ref_ed
            else:
                p = mic2refpair(m)
                st = max(1, ref_st + besttdoa(p, t))
                ed = min(nsample, ref_ed + besttdoa(p, t))
            # triwin = np.transpose(triang(nwin))/////////////////
            diff = 0
            if (ref_ed - ref_st) != (ed - st):
                diff = ref_ed - ed
            # out_x[ref_st + diff: ref_ed] = out_x[ref_st + diff: ref_ed] + (
            #     np.dot((np.squeeze(x[m, st:ed]) * out_weight(m, t)),
            #            (triwin[1: min(nwin, ed - st + 1)] * overall_weight)))//////////////////
    while ref_st + 4000 < nsample:
        ref_st = ref_st + 4000
        ref_ed = min(ref_st + nwin - 1, nsample)

        for m in range(0, nmic - 1):
            if m == ref_mic:
                st = ref_st
                ed = ref_ed
            else:
                p = mic2refpair(m)
                st = max(1, ref_st + besttdoa(p, t))
                ed = min(nsample, ref_ed + besttdoa(p, t))
            if st > ed:
                continue

            buf = np.squeeze(x[m, st:ed])
            diff = (ref_ed - ref_st) - (ed - st)
            if diff > 0:
                buf = [buf, np.zeros(1, diff)];
            else:
                buf = buf[1:buf[-1] - diff]
            # triwin = np.transpose(triang(nwin))//////////////////////
    #         out_x[ref_st: ref_ed] = out_x[ref_st: ref_ed] + np.dot(
    #             (buf * out_weight(m, t), triwin[1:min(nwin, np.size(buf, 1))]) * overall_weight)
    #
    return out_x


test(infilenames_cell, outfilename)
