# Authors: Edouard Oyallon, Muawiz Chaudhary
# Scientific Ancestry: Edouard Oyallon, Laurent Sifre, Joan Bruna

def scattering2d(x, pad, unpad, backend, J, L, OS, phi, psi, max_order,
        out_type='array', local=False):
    subsample_fourier = backend.subsample_fourier
    modulus = backend.modulus
    fft = backend.fft
    cdgmm = backend.cdgmm
    concatenate = backend.concatenate
    mean = backend.mean

    # Define lists for output.
    out_S_0, out_S_1, out_S_2 = [], [], []

    U_r = pad(x)

    U_0_c = fft(U_r, 'C2C')
    resU0 = 0

    if local:
        # First low pass filter
        U_1_c = cdgmm(U_0_c, phi[0])
        U_1_c = subsample_fourier(U_1_c, k=2 ** max(J - OS, 0))
    
        S_0 = fft(U_1_c, 'C2R', inverse=True)
        S_0 = unpad(S_0)
    else:
        S_0 = fft(U_0_c, 'C2R', inverse=True)
        S_0 = unpad(S_0)
        S_0 = mean(S_0, axis=(-1, -2))

    out_S_0.append({'coef': S_0,
                    'j': (),
                    'theta': ()})

    for n1 in range(len(psi)):
        j1 = psi[n1]['j']
        theta1 = psi[n1]['theta']

        U_1_c = cdgmm(U_0_c, psi[n1][resU0])
        if j1 > 0:
            U_1_c = subsample_fourier(U_1_c, k=2 ** max(j1 - resU0 - OS, 0))
        U_1_c = fft(U_1_c, 'C2C', inverse=True)
        U_1_c = modulus(U_1_c)
        U_1_c = fft(U_1_c, 'C2C')
        resU1 = resU0 + max(j1 - resU0 - OS, 0)

        if local:
            # Second low pass filter
            S_1_c = cdgmm(U_1_c, phi[resU1])
            S_1_c = subsample_fourier(S_1_c, k=2 ** max(J - resU1 - OS, 0))
    
            S_1_r = fft(S_1_c, 'C2R', inverse=True)
            S_1_r = unpad(S_1_r)
        else:
            S_1_r = fft(U_1_c, 'C2R', inverse=True)
            S_1_r = unpad(S_1_r)
            S_1_r = mean(S_1_r, axis=(-1, -2))

        out_S_1.append({'coef': S_1_r,
                        'j': (j1,),
                        'theta': (theta1,)})

        if max_order < 2:
            continue
        for n2 in range(len(psi)):
            j2 = psi[n2]['j']
            theta2 = psi[n2]['theta']

            if j2 <= j1:
                continue

            U_2_c = cdgmm(U_1_c, psi[n2][resU1])
            U_2_c = subsample_fourier(U_2_c, k=2 ** max(j2 - resU1 - OS, 0))
            U_2_c = fft(U_2_c, 'C2C', inverse=True)
            U_2_c = modulus(U_2_c)
            U_2_c = fft(U_2_c, 'C2C')
            resU2 = resU1 + max(j2 - resU1 - OS, 0)

            if local:
                # Third low pass filter
                S_2_c = cdgmm(U_2_c, phi[resU2])
                S_2_c = subsample_fourier(S_2_c, k=2 ** max(J - resU2 - OS, 0))
    
                S_2_r = fft(S_2_c, 'C2R', inverse=True)
                S_2_r = unpad(S_2_r)
            else:
                S_2_r = fft(U_2_c, 'C2R', inverse=True)
                S_2_r = unpad(S_2_r)
                S_2_r = mean(S_2_r, axis=(-1, -2))

            out_S_2.append({'coef': S_2_r,
                            'j': (j1, j2),
                            'theta': (theta1, theta2)})

    out_S = []
    out_S.extend(out_S_0)
    out_S.extend(out_S_1)
    out_S.extend(out_S_2)

    if out_type == 'array':
        if local:
            out_S = concatenate([x['coef'] for x in out_S], -3)
        else:
            out_S = concatenate([x['coef'] for x in out_S], -1)

    return out_S


__all__ = ['scattering2d']
