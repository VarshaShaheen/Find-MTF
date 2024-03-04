import mtf as mtf

imgArr = mtf.Helper.LoadImageAsArray('img.png')
res = mtf.MTF.CalculateMtf(imgArr, verbose=mtf.Verbosity.DETAIL)