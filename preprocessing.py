import numpy as np 


def fill_sptr(s, freq):
    """
    Fill Whisper spectrum gaps and remove specific interference points.

    Parameters
    ----------
    s  
        Whisper spectrum values, where missing data may appear as NaN.
    freq 
        Frequency array corresponding to the spectrum.

    Returns
    -------
    ndarray
        The corrected spectrum with internal NaN regions interpolated,
        external NaNs replaced with zeros
    """

    # Find limits of the spectrum
    value = np.argwhere(np.isfinite(s))
    tmp_s = s[value.min():value.max()]
    tmp_f = np.array(freq[value.min():value.max()])
    
    # Find NaNs
    internalNaN = np.isnan(s[value.min():value.max()])
    if internalNaN.sum() > 0:
        # Linear interpolation between values 
        internalVal = np.isfinite(s[value.min():value.max()])
        tmp_s[internalNaN] = np.interp(tmp_f[internalNaN], tmp_f[internalVal], tmp_s[internalVal])
        s[value.min():value.max()] = tmp_s
    
    # Replace remaining NaNs with 0
    s = np.nan_to_num(s)

    return s


def normSptr_db(spectra,freq):
    """
    Get Whisper spectra expressed in dB and normalize them to the range [0, 1].

    Parameters 
    ----------
    spectra 
        Whisper spectra to be processed.
    freq 
        Frequency values associated with the spectra.

    Returns
    -------
    s : ndarray
        The processed spectrum, filled, converted to dB, and normalized
        between 0 and 1.
    """

    for i,s in enumerate(spectra):
        try :   
            # Fill spectra 
            sFill = fill_sptr(s.copy(),freq)
            above0 = sFill > 0         
            sdb = sFill.copy()   
            
            # Amplitude in dB     
            sdb[above0] = 20*np.log10(sdb[above0]/sdb[above0].min())
            if sdb.max() == 0:
                print("Sp number {} is not in dB".format(i))
                sdb = sFill   
            spectra[i,:] = (sdb - sdb.min())/(sdb.max() - sdb.min())

        except ValueError:
            print('error i: {} '.format(i))

    return spectra


def calc_fpe_dens(density):
    """
    Parameters 
    ----------
    density
        electron density in cm^-3

    Returns
    ------- 
    fpe 
        plasma frequency in kHz 
    """
    fpe = 8.98*np.sqrt(density)
    return fpe


def find_nearest_index(array, value):
    """
    Find nearest index

    Parameters
        array: input array
        value: value 
    
    Returns 
        closest index 
    """
    idx = np.searchsorted(array,value)
    idx = np.clip(idx,1,len(array)-1)

    l = array[idx-1]
    r = array[idx]

    idx -= value - l <= r - value

    return idx