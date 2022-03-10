def calc_energy(img):
    filter_du = np.array([
        [1.0, 2.0, 1.0],
        [0.0, 0.0, 0.0],
        [-1.0, -2.0, -1.0],
    ])
    
    filter_du = np.stack([filter_du] * 3, axis=2)

    filter_dv = np.array([
        [1.0, 0.0, -1.0],
        [2.0, 0.0, -2.0],
        [1.0, 0.0, -1.0],
    ])
    filter_dv = np.stack([filter_dv] * 3, axis=2)

    convolved = np.sqrt(np.square(convolve(img, filter_du)) + np.square(convolve(img, filter_dv)))

    energy_map = convolved.sum(axis=2)
    return energy_map
    
    print('zuraggui bol ur dungui')