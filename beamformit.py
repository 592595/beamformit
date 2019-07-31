import numpy as np
import math
from scipy.fftpack import fft, ifft
import matplotlib.pylab as plt
import soundfile as sf
import function as fc

infilenames_cell = ('sample11/M05_443C0207_PED.CH1.wav',
                    'sample11/M05_443C0207_PED.CH3.wav',
                    'sample11/M05_443C0207_PED.CH4.wav',
                    'sample11/M05_443C0207_PED.CH5.wav',
                    'sample11/M05_443C0207_PED.CH6.wav')

outfilename = 'enhanced.wav'


def test(infilenames_cell, outfilename):
    [x, sr, nmic, npair, nsample] = fc.get_x(infilenames_cell)
    plt.figure()
    plt.plot(x[0, :])
    plt.show()

    # make hamming window
    nwin = 16000
    win = fc.hamming_bfit(nwin)
    plt.figure()
    plt.plot(win)
    plt.show()
    print(win[1:10])

    # calculate avg_ccorr
    npiece = 200
    nfft = 32768
    nbest = 2
    nmask = 5
    ref_mic = fc.calcuate_avg_ccorr(x, nsample, nmic, npiece, win, nwin, nfft, nbest, nmask)

    # calculating scaling factor
    nsegment = 10
    overall_weight = fc.calculate_scaling_factor(x, sr, nsample, nmic, nsegment)

    # compute total number of delays
    nwin = 8000
    nshift = nwin / 2
    nframe = math.floor((nsample - nwin) / (nshift))
    print(nframe)

    # recreating hamming window
    win = fc.hamming_bfit(nwin)
    plt.figure()
    plt.plot(win)
    plt.show()
    print(16000 * 30 / 1000)

    # get pair2mic table
    pair2mic = fc.get_pair2mic(nmic, npair)

    # compute TDOA
    nbest = 4
    nfft = 16384
    [gcc_nbest, tdoa_nbest] = fc.compute_tdoa(x, npair, ref_mic, pair2mic, nframe, win, nwin, nshift, nfft, nbest, nmask)
    print(np.squeeze(gcc_nbest[:, :, 1]))
    print(np.squeeze(gcc_nbest[:, :, 2]))

    print(np.squeeze(tdoa_nbest[:, :, 1]))
    print(np.squeeze(tdoa_nbest[:, :, 2]))

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


test(infilenames_cell, outfilename)
