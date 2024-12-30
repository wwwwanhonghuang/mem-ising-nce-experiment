def configuration_int32_encoding(configuration):
    v = 0
    for i in range(len(configuration)):
        v <<= 1
        v |= (int)(configuration[i])

    return v